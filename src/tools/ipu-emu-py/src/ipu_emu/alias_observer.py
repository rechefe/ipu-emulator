"""Execution-to-profiler adapter. Instruction decoding stays in the dispatcher."""
from __future__ import annotations

import math
import struct

from ipu_common.activations import ACTIVATION_IDENTITY
from ipu_common.registers import get_lane_count
from ipu_emu.alias_profile import Event
from ipu_emu.ipu_math import dtype_one_byte
from ipu_emu.xmem import XMEM_WIDTH_BYTES

LANES = get_lane_count()


class ExecutionObserver:
    def __init__(self, profile, state):
        self.profile = profile
        self.state = state
        profile.metadata.update(mode=state.wide_vector_arithmetic.value if state.wide_vector_debug else "native",
                                dtype=state.dtype.name, quantize_output=state.wide_vector_quantize_output)
        self.next_id = 0
        self.pc = 0
        self.cycle = 0
        self.current = None
        self.accesses = []
        self.rows = {}
        self.snapshot_rows = {}
        self.stores = {}
        self.memory_versions = {}
        self.memory_epoch = 0
        self.last_reads = {}
        self.acc = None
        self.mult = None
        self.post = None
        self.chain = 0
        self.acc_scale = 0
        self.post_scale = 0
        self.cycle_events = []
        state.xmem._profile_observer = self.memory
        state.regfile._profile_observer = self.register_write

    def register_write(self, name, start, size):
        banks = {"r": "ra", "r_wide_debug": "ra", "r_cyclic": "ring", "r_cyclic_wide_debug": "ring"}
        if name in banks:
            width = LANES * (4 if "wide_debug" in name else 1)
            for index in range(start // width, (start + size - 1) // width + 1):
                self.rows.pop((banks[name], index), None)
        if self.current is None:
            if name == "r_acc":
                self.acc = None
                self.acc_scale = 0
            elif name == "post_aaq_reg":
                self.post = None
                self.post_scale = 0
            elif name == "mult_res":
                self.mult = None

    def diamond(self, ipu, plan, kwargs):
        """Prove the canonical two-arm SET diamond from decoded ISA plans."""
        if plan.inst_name not in ("BEQ", "BNE", "BLT", "BGE"):
            return None
        if kwargs.get("label") != self.pc + 3:
            return None
        from ipu_emu.ipu import _SLOT_PLANS, _SLOT_OPCODE_FIELDS, _LR_PLANS, _LR_OPCODE_FIELDS

        def operations(pc):
            if pc >= len(self.state.inst_mem) or self.state.inst_mem[pc] is None:
                return []
            inst = self.state.inst_mem[pc]
            result = []
            for slot, plans in _SLOT_PLANS.items():
                p = plans[inst[_SLOT_OPCODE_FIELDS[slot]]]
                if p.inst_name != "NOP":
                    result.append((slot, p.inst_name, {n: inst[k] for n, k in p.fields}))
            for i, plans in enumerate(_LR_PLANS):
                p = plans[inst[_LR_OPCODE_FIELDS[i]]]
                if p.inst_name != "NOP":
                    result.append(("lr", p.inst_name, {n: inst[k] for n, k in p.fields}))
            return result

        a, jump, b = (operations(self.pc + i) for i in (1, 2, 3))
        if not (len(a) == len(b) == len(jump) == 1):
            return None
        if a[0][:2] != ("lr", "SET") or b[0][:2] != ("lr", "SET"):
            return None
        if a[0][2]["reg"] != b[0][2]["reg"]:
            return None
        slot, name, args = jump[0]
        if slot == "cond" and name == "BEQ" and args["reg1"] == args["reg2"] and args["label"] == self.pc + 4:
            return a[0][2]["reg"], self.pc + 4
        return None

    def memory(self, kind, address, data):
        end = address + len(data)
        blocks = range(address // XMEM_WIDTH_BYTES, (end + XMEM_WIDTH_BYTES - 1) // XMEM_WIDTH_BYTES)
        origin = None
        if kind == "write":
            self.memory_epoch += 1
            for block in blocks:
                self.stores.pop(block, None)
                self.memory_versions[block] = self.memory_epoch
        else:
            first = self.stores.get(address // XMEM_WIDTH_BYTES)
            if (first and first[0] <= address and end <= first[0] + first[1]
                    and all(self.stores.get(block) is first for block in blocks)):
                origin = first[2]
        if self.current is not None:
            self.accesses.append((kind, address, bytes(data), origin))

    def begin(self, ipu):
        self.cycle += 1
        self.profile.cycles += 1
        self.pc = self.state.program_counter
        self.snapshot_rows = self.rows.copy()
        self.cycle_events = []

    def start(self, ipu, slot, plan, kwargs, encoded):
        if plan.inst_name == "NOP":
            return None
        facts = {"operands": dict(kwargs), "encoded": encoded, "chain": self.chain}
        # Do not serialize resolved vectors; only small execution facts escape.
        facts["operands"] = {k: v for k, v in kwargs.items() if isinstance(v, int)}
        facts["lr_reads"] = {read.name: encoded[read.name] for read in plan.reads
                             if read.op_type == "LrIdx" or
                             (read.op_type == "LcrIdx" and encoded[read.name] < 16)}
        facts["snapshot_reads"] = tuple(read.name for read in plan.reads if read.from_snapshot)
        if slot == "lr" and plan.write_operand:
            facts["lr_writes"] = (ipu._lrd_lr_indices(kwargs[plan.write_operand])
                                   if plan.write_is_lrd else (kwargs[plan.write_operand],))
        if slot == "mult":
            cr = kwargs.get("dstructure_cr_idx", kwargs.get("cr_idx"))
            mask, ds = ipu._effective_mult_mask(kwargs["mask_offset"], kwargs["mask_shift"], cr)
            width = ds.valid_elements or LANES
            facts.update(active_lanes=(mask & ((1 << width) - 1)).bit_count(),
                         mask=mask, declared_width=ds.valid_elements, partition=int(ds.partition),
                         identity=ipu._mult_identity,
                         wide_fp32=self.state.wide_vector_debug and self.state.wide_vector_arithmetic.value == "fp32")
            sources = []
            if "rc_idx" in kwargs:
                sources += self.window(ipu, "ring", kwargs["rc_idx"], LANES)
            if "ra" in encoded:
                sources += self.window(ipu, "ra", encoded["ra"] * LANES, LANES)
            if "ra_idx" in kwargs:
                sources += self.window(ipu, "ra", kwargs["ra_idx"],
                                       1 if plan.inst_name == "MULT.EE" else LANES)
            if plan.inst_name == "MULT.RC.VE" and kwargs["src"] < 16:
                sources += self.window(ipu, "ra", self.state.regfile.get_lr(kwargs["src"]), 1)
                facts["lr_reads"]["src"] = kwargs["src"]
            facts["sources"] = tuple(sources)
            complete = [s for s in sources if plan.inst_name == "MULT.RC.VV" and (
                (s.facts.get("bank") == "ra" and s.facts.get("bank_index") == encoded.get("ra")) or
                (s.facts.get("bank") == "ring" and kwargs["rc_idx"] % LANES == 0 and
                 s.facts.get("bank_index") == kwargs["rc_idx"] % (4 * LANES) // LANES))]
            facts["constants"] = self.constants(ipu, complete)
            facts["scale_lanes"] = self.mult_scale(ipu, plan.inst_name, kwargs, encoded, facts)
        if slot == "acc":
            facts["mult"] = self.mult
            facts["previous_acc_id"] = self.acc.id if self.acc else None
            facts["scale_lanes"] = self.accumulator_scale(ipu, plan.inst_name, kwargs)
        if slot == "aaq":
            facts["acc"] = self.acc
            facts["lossy"] = not (self.state.wide_vector_debug and not self.state.wide_vector_quantize_output)
            active = min(LANES, self.state.get_dstructure_for(kwargs["cr_idx"]).valid_elements)
            active_mask = (1 << active) - 1
            facts["input_scale_verified"] = bool(active) and self.acc_scale & active_mask == active_mask
            facts["scale_lanes"] = ((self.post_scale & ~active_mask) | (self.acc_scale & active_mask)
                                    if kwargs["activation_fn"] == ACTIVATION_IDENTITY and not facts["lossy"] else 0)
        if slot == "store":
            if self.post:
                # Transfers need the old accumulator version, not its recursive
                # expression tree. EXP uses the current AAQ event directly.
                post_facts = self.post.facts.copy()
                old = post_facts.get("acc")
                post_facts["acc"] = Event(old.id, old.cycle, old.pc, old.slot, old.instruction) if old else None
                facts["post"] = Event(self.post.id, self.post.cycle, self.post.pc,
                                      self.post.slot, self.post.instruction, post_facts)
                facts["scale_lanes"] = self.post_scale
        if slot == "cond":
            diamond = self.diamond(ipu, plan, kwargs)
            if diamond:
                facts["diamond"] = diamond
        self.current = (slot, plan.inst_name, facts)
        self.accesses = []
        return self.current

    def scaled_window(self, ipu, bank, start, length):
        """Lane provenance follows snapshot loads, never numerical equality alone."""
        sources = {e.facts["bank_index"]: e for e in self.window(ipu, bank, start, length)}
        if not any(e.facts.get("scale_lanes", 0) for e in sources.values()):
            return 0
        capacity = LANES * (4 if bank == "ring" else 2)
        result = 0
        for i in range(length):
            pos = (start + i) % capacity
            source = sources.get(pos // LANES)
            if source and source.facts.get("scale_lanes", 0) & (1 << (pos % LANES)):
                result |= 1 << i
        return result

    def mult_scale(self, ipu, name, kw, encoded, facts):
        """Prove a common log2(e) factor; unknown operations erase the proof."""
        if not facts["wide_fp32"]:
            return 0
        full = (1 << LANES) - 1
        scaled = 0
        if name == "MULT.RC.VV":
            for constant in facts["constants"]:
                if constant["log2e"] and constant["declared"]:
                    scaled = full
                    break
                if constant["uniform"] and constant["value"] == 1 and constant["declared"]:
                    source = constant["source"]
                    bank = "ra" if source.facts["bank"] == "ring" else "ring"
                    start = encoded["ra"] * LANES if bank == "ra" else kw["rc_idx"]
                    scaled = self.scaled_window(ipu, bank, start, LANES)
        elif name in ("MULT.VE", "MULT.EE") and ipu._wide_cr_scalar_byte_as_int32(kw["cr_idx"]) == 1:
            length = 1 if name == "MULT.EE" else LANES
            scaled = self.scaled_window(ipu, "ra", kw["ra_idx"], length)
            if length == 1:
                scaled = full if scaled else 0
        elif name == "MULT.RC.VE" and kw["src"] >= 16 and ipu._wide_cr_scalar_byte_as_int32(kw["src"] - 16) == 1:
            scaled = self.scaled_window(ipu, "ring", kw["rc_idx"], LANES)
        # Mask padding is zero or infinity, both invariant under positive scaling.
        return scaled | (full ^ facts["mask"])

    def accumulator_scale(self, ipu, name, kw):
        mult = self.mult.facts.get("scale_lanes", 0) if self.mult else 0
        if name in ("ACC.ADD.FIRST", "ACC.SUB.FIRST", "ACC.MAX.FIRST"):
            return mult
        if name in ("ACC.ADD", "ACC.SUB", "ACC.MAX"):
            return self.acc_scale & mult
        if name.startswith(("AGG.SUM", "AGG.MAX")):
            active = min(LANES, self.state.get_dstructure_for(kw["cr_idx"]).valid_elements)
            mask = (1 << active) - 1
            dest = 1 << (kw["dest_slot"] % LANES)
            proven = mult & mask == mask and (name.endswith(".FIRST") or self.acc_scale & dest)
            return (self.acc_scale & ~dest) | (dest if proven else 0)
        return 0

    def window(self, ipu, bank, start, length):
        """Associate snapshot operands with loads, including cyclic slot crossings."""
        capacity = LANES * (4 if bank == "ring" else 2)
        slots = {(start + i) % capacity // LANES for i in range(length)}
        result = []
        wide = self.state.wide_vector_debug
        name = ("r_cyclic_wide_debug" if wide else "r_cyclic") if bank == "ring" else ("r_wide_debug" if wide else "r")
        raw = ipu._snapshot_buffer.raw_readonly(name)
        if name in self.state.regfile._mutable_exports:
            return []
        size = ipu._row_size_bytes()
        for slot in sorted(slots):
            item = self.snapshot_rows.get((bank, slot))
            if item and raw[slot * size:(slot + 1) * size] == item[1]:
                result.append(item[0])
        return result

    def constants(self, ipu, sources):
        # A property of a complete loaded row. Partial/cross-row operands cannot
        # prove that a complete MULT operand is a broadcast constant.
        result = []
        for source in sources:
            if "constant_properties" in source.facts:
                props = source.facts["constant_properties"]
                if props is not None:
                    result.append(dict(props, source=source))
                continue
            data = source.facts.get("data", b"")
            if len(data) != ipu._row_size_bytes():
                continue
            element = ipu._element_width_bytes()
            one = struct.pack("<f" if self.state.wide_vector_arithmetic.value == "fp32" else "<i", 1) if self.state.wide_vector_debug else bytes([dtype_one_byte(self.state.dtype)])
            zero = bytes(element)
            if data != data[:element] * LANES and data.replace(one, b"").replace(zero, b""):
                source.facts["constant_properties"] = None
                continue
            if self.state.wide_vector_debug:
                fmt = "<f" if self.state.wide_vector_arithmetic.value == "fp32" else "<i"
                values = [x[0] for x in struct.iter_unpack(fmt, data)]
            else:
                from ipu_emu.ipu_math import DType, fp8_bytes_to_fp32
                values = ([x if x < 128 else x - 256 for x in data]
                          if self.state.dtype == DType.INT8 else fp8_bytes_to_fp32(data, self.state.dtype).tolist())
            if len(values) != LANES or not all(math.isfinite(x) for x in values):
                continue
            uniform = all(x == values[0] for x in values)
            binary = all(x in (0, 1) for x in values) and 0 in values and 1 in values
            properties = {"source": source, "uniform": uniform, "value": values[0], "binary": binary,
                           "declared": source.facts.get("declared", False),
                           "fractional": uniform and values[0] != int(values[0]),
                           "log2e": uniform and data[:4] == struct.pack("<f", math.log2(math.e))
                           if self.state.wide_vector_debug and self.state.wide_vector_arithmetic.value == "fp32" else False}
            # Cache scalar properties only (no self-reference).
            source.facts["constant_properties"] = {k: v for k, v in properties.items() if k != "source"}
            result.append(properties)
        return tuple(result)

    def finish(self, ipu, token, alias=None):
        if token is None:
            return
        slot, name, facts = token
        if alias:
            facts["alias"] = alias
        facts["accesses"] = tuple((kind, addr, len(data)) for kind, addr, data, _ in self.accesses)
        facts["read_bytes"] = sum(len(data) for kind, _, data, _ in self.accesses if kind == "read")
        facts["write_bytes"] = sum(len(data) for kind, _, data, _ in self.accesses if kind == "write")
        if slot == "load" and self.accesses:
            _, address, data, origin = self.accesses[0]
            blocks = range(address // XMEM_WIDTH_BYTES,
                           (address + len(data) + XMEM_WIDTH_BYTES - 1) // XMEM_WIDTH_BYTES)
            versions = tuple(self.memory_versions.get(block, 0) for block in blocks)
            previous = self.last_reads.get((address, len(data)))
            if previous and previous[2] != versions:
                previous = None
            # A row the program never wrote was placed by setup and is immutable
            # for the whole run: that is what makes it a constant. The observer
            # only attaches once execution begins, so an all-zero version tuple
            # means exactly that, and no harness-side declaration is needed.
            facts.update(address=address, data=data, origin=origin,
                         previous_read=previous,
                         declared=not any(versions))
            if origin and origin.instruction == "STR_POST_AAQ_REG":
                store_address = origin.facts["accesses"][0][1]
                if (address - store_address) % 4 == 0:
                    facts["scale_lanes"] = (origin.facts.get("scale_lanes", 0) >> ((address - store_address) // 4)) & ((1 << (len(data) // 4)) - 1)
            if name == "LDR_MULT_REG":
                facts.update(bank="ra", bank_index=facts["operands"]["dest"])
            elif name == "LDR_CYCLIC_MULT_REG":
                facts.update(bank="ring", bank_index=facts["operands"]["index"] // LANES)
        self.next_id += 1
        event = Event(self.next_id, self.cycle, self.pc, slot, name, facts)
        self.current = None
        if slot == "load" and self.accesses:
            self.last_reads[(facts["address"], len(facts["data"]))] = (event.cycle, self.chain, versions)
            kw = facts["operands"]
            if name == "LDR_MULT_REG":
                self.rows[("ra", kw["dest"])] = (event, facts["data"])
            elif name == "LDR_CYCLIC_MULT_REG":
                self.rows[("ring", kw["index"] // LANES)] = (event, facts["data"])
        elif slot == "mult":
            self.mult = event
        elif slot == "acc":
            # Only keep the latest accumulator producer. Detectors retain their
            # own bounded sequence state instead of an unbounded expression DAG.
            facts.pop("previous_acc", None)
            if name.endswith(".FIRST"):
                self.chain += 1
            self.acc = event
            self.acc_scale = facts.get("scale_lanes", 0)
        elif slot == "aaq":
            self.post = event
            self.post_scale = facts.get("scale_lanes", 0)
        for kind, address, data, _ in self.accesses:
            if kind == "write":
                stored = (address, len(data), event)
                for block in range(address // XMEM_WIDTH_BYTES,
                                   (address + len(data) + XMEM_WIDTH_BYTES - 1) // XMEM_WIDTH_BYTES):
                    self.stores[block] = stored
        self.cycle_events.append(event)
        self.profile.emit(event)

    def end(self):
        self.next_id += 1
        event = Event(self.next_id, self.cycle, self.pc, "cycle", "CYCLE",
                      {"events": tuple(self.cycle_events), "next_pc": self.state.program_counter})
        # Cycle summaries are control evidence, not retired instructions.
        self.profile.observe(event)
