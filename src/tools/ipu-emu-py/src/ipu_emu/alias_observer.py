"""Execution-to-profiler adapter. Instruction decoding stays in the dispatcher.

The profiler sees the emulator only through public records (E04-11): a
:class:`~ipu_emu.cycle.CycleObserver` per cycle, one
:class:`~ipu_emu.issue.IssueFacts` per op, the ranged register and XMEM events,
the cycle-start snapshot through ``Cycle.snapshot`` and the shared helpers of
:mod:`ipu_emu.geometry`. It imports nothing private from ``ipu_emu``.
"""

from __future__ import annotations

import math
import struct
import weakref
from typing import TYPE_CHECKING, Any

from ipu_emu.alias_profile import Event
from ipu_emu.cycle import decode_word
from ipu_emu.geometry import cr_scalar, element_width_bytes, row_size_bytes, vector_registers
from ipu_emu.xmem import XMEM_WIDTH_BYTES
from ipu_isa.activations import ACTIVATION_IDENTITY
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.operand_types import OperandType
from ipu_isa.registers import REGISTER_DEFINITIONS

if TYPE_CHECKING:
    from ipu_emu.cycle import Cycle
    from ipu_emu.issue import IssueFacts

_LR_NAMES = frozenset(REGISTER_DEFINITIONS["lr"]["assembler_values"])


def _names_an_lr(operand_type: OperandType, encoded: int) -> bool:
    """Whether an operand of ``operand_type`` encoded as ``encoded`` names an LR."""
    members = operand_type.domain.members
    return 0 <= encoded < len(members) and members[encoded] in _LR_NAMES


class _CycleBinding:
    """An :class:`ExecutionObserver` attached to one cycle.

    One profile observes one state, but every engine built on that state has
    its own cycle, so each attaches its own binding: ``begin`` points the
    observer at the cycle whose snapshot it reads, and every other hook goes
    straight to the observer.
    """

    __slots__ = ("_cycle", "_observer", "end", "issued", "retired")

    def __init__(self, observer: ExecutionObserver, cycle: Cycle) -> None:
        self._observer = observer
        # Weak: the cycle holds this binding.
        self._cycle = weakref.ref(cycle)
        self.issued = observer.issued
        self.retired = observer.retired
        self.end = observer.end

    def begin(self, pc: int) -> None:
        self._observer.cycle_source = self._cycle
        self._observer.begin(pc)


#: The emulator-only 4-byte formats (the profile's ``mode`` names them).
_WIDE_FORMATS = (ElementFormat.FP32, ElementFormat.INT32)


class ExecutionObserver:
    def __init__(self, profile, state):
        self.profile = profile
        # Weak: state -> alias_profile -> _bridge -> observer, so a strong
        # reference back would make every profiled state (and its XMEM) wait
        # for the cyclic garbage collector instead of being freed on release.
        self._state = weakref.ref(state)
        fmt = state.fmt
        wide = fmt in _WIDE_FORMATS
        profile.metadata.update(
            mode=fmt.value if wide else "native",
            # The 8-bit format; FP32 and INT32 report INT8, as before E04-17.
            dtype=ElementFormat.INT8.name if wide else fmt.name,
            quantize_output=state.aaq.quantize_output,
        )
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
        # Version of a block absent from memory_versions: 0 (never written)
        # until a clear() stamps the whole memory at once.
        self.cleared_epoch = 0
        self.last_reads = {}
        self.acc = None
        self.mult = None
        self.post = None
        self.chain = 0
        self.acc_scale = 0
        self.post_scale = 0
        self.cycle_events = []
        # The cycle being observed (weakly), whose snapshot the operand windows
        # read; set by the cycle's binding (observing) at each cycle start.
        self.cycle_source = lambda: None
        self._removers = (
            state.xmem.add_observer(self.memory),
            state.regfile.add_observer(self.register_write),
        )

    def observing(self, cycle: Cycle) -> _CycleBinding:
        """This observer as a ``CycleObserver`` of *cycle*, for ``Cycle.add_observer``."""
        return _CycleBinding(self, cycle)

    def detach(self):
        """Stop observing the state's XMEM and register file."""
        for remove in self._removers:
            remove()
        self._removers = ()

    @property
    def state(self):
        return self._state()

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

    def diamond(self, plan, kwargs):
        """Prove the canonical two-arm SET diamond from decoded ISA plans."""
        if plan.inst_name not in ("BEQ", "BNE", "BLT.S", "BGE.S"):
            return None
        if kwargs.get("label") != self.pc + 3:
            return None

        def operations(pc):
            if pc >= len(self.state.inst_mem) or self.state.inst_mem[pc] is None:
                return []
            return [
                (op.slot, op.plan.inst_name, dict(op.operands))
                for op in decode_word(self.state.inst_mem[pc])
                if op.plan.inst_name != "NOP"
            ]

        a, jump, b = (operations(self.pc + i) for i in (1, 2, 3))
        if not (len(a) == len(b) == len(jump) == 1):
            return None
        if a[0][:2] != ("lr", "SET") or b[0][:2] != ("lr", "SET"):
            return None
        if a[0][2]["reg"] != b[0][2]["reg"]:
            return None
        slot, name, args = jump[0]
        if (
            slot == "cond"
            and name == "BEQ"
            and args["reg1"] == args["reg2"]
            and args["label"] == self.pc + 4
        ):
            return a[0][2]["reg"], self.pc + 4
        return None

    def memory(self, kind, address, size, data):
        if kind == "clear":
            # Every block now holds a fresh zero written at this epoch. Drop the
            # tables and remember the epoch as the version of every untouched
            # block, instead of stamping each of the 4M blocks.
            self.memory_epoch += 1
            self.stores.clear()
            self.memory_versions.clear()
            self.cleared_epoch = self.memory_epoch
            return
        end = address + size
        blocks = range(
            address // XMEM_WIDTH_BYTES, (end + XMEM_WIDTH_BYTES - 1) // XMEM_WIDTH_BYTES
        )
        origin = None
        if kind == "write":
            self.memory_epoch += 1
            for block in blocks:
                self.stores.pop(block, None)
                self.memory_versions[block] = self.memory_epoch
        else:
            first = self.stores.get(address // XMEM_WIDTH_BYTES)
            if (
                first
                and first[0] <= address
                and end <= first[0] + first[1]
                and all(self.stores.get(block) is first for block in blocks)
            ):
                origin = first[2]
        if self.current is not None:
            if data is None:  # a fill: the event carries the range, not the bytes
                data = self.state.xmem.snapshot_bytes(address, size)
            self.accesses.append((kind, address, bytes(data), origin))

    def begin(self, pc):
        self.cycle += 1
        self.profile.cycles += 1
        self.pc = pc
        self.snapshot_rows = self.rows.copy()
        self.cycle_events = []

    def issued(self, issued: IssueFacts) -> None:
        """Open the event of one op about to execute, from its ``IssueFacts``."""
        slot, plan, kwargs, encoded = issued.slot, issued.plan, issued.operands, issued.encoded
        if plan.inst_name == "NOP" or slot == "break":
            return
        # Do not serialize resolved vectors; only small execution facts escape.
        facts = {
            "operands": {k: v for k, v in kwargs.items() if isinstance(v, int)},
            "encoded": dict(encoded),
            "chain": self.chain,
        }
        facts["lr_reads"] = {
            read.name: encoded[read.name]
            for read in plan.reads
            if _names_an_lr(read.op_type, encoded[read.name])
        }
        facts["snapshot_reads"] = tuple(read.name for read in plan.reads if read.from_snapshot)
        if issued.lr_writes:
            facts["lr_writes"] = issued.lr_writes
        if slot == "mult":
            ds = self.state.get_dstructure_for(kwargs[plan.mask_cr_operand])
            width = ds.valid_elements or LANES
            facts.update(
                active_lanes=(issued.mask & ((1 << width) - 1)).bit_count(),
                mask=issued.mask,
                declared_width=ds.valid_elements,
                partition=int(ds.partition),
                identity=issued.mult_identity.any,
                wide_fp32=self.state.fmt is ElementFormat.FP32,
            )
            sources = []
            if "rc_idx" in kwargs:
                sources += self.window("ring", kwargs["rc_idx"], LANES)
            if "ra" in encoded:
                sources += self.window("ra", encoded["ra"] * LANES, LANES)
            if "ra_idx" in kwargs:
                sources += self.window(
                    "ra", kwargs["ra_idx"], 1 if plan.inst_name == "MULT.EE" else LANES
                )
            if plan.inst_name == "MULT.RC.VE" and kwargs["src"] < 16:
                sources += self.window("ra", self.state.regfile.get_lr(kwargs["src"]), 1)
                facts["lr_reads"]["src"] = kwargs["src"]
            facts["sources"] = tuple(sources)
            complete = [
                s
                for s in sources
                if plan.inst_name == "MULT.RC.VV"
                and (
                    (s.facts.get("bank") == "ra" and s.facts.get("bank_index") == encoded.get("ra"))
                    or (
                        s.facts.get("bank") == "ring"
                        and kwargs["rc_idx"] % LANES == 0
                        and s.facts.get("bank_index") == kwargs["rc_idx"] % (4 * LANES) // LANES
                    )
                )
            ]
            facts["constants"] = self.constants(complete)
            facts["scale_lanes"] = self.mult_scale(plan.inst_name, kwargs, encoded, facts)
        if slot == "acc":
            facts["mult"] = self.mult
            facts["previous_acc_id"] = self.acc.id if self.acc else None
            facts["scale_lanes"] = self.accumulator_scale(plan.inst_name, kwargs)
        if slot == "aaq":
            facts["acc"] = self.acc
            facts["lossy"] = not (
                self.state.fmt in _WIDE_FORMATS and not self.state.aaq.quantize_output
            )
            active = min(LANES, self.state.get_dstructure_for(kwargs["cr_idx"]).valid_elements)
            active_mask = (1 << active) - 1
            facts["input_scale_verified"] = (
                bool(active) and self.acc_scale & active_mask == active_mask
            )
            facts["scale_lanes"] = (
                (self.post_scale & ~active_mask) | (self.acc_scale & active_mask)
                if kwargs["activation_fn"] == ACTIVATION_IDENTITY and not facts["lossy"]
                else 0
            )
        if slot == "store":
            if self.post:
                # Transfers need the old accumulator version, not its recursive
                # expression tree. EXP uses the current AAQ event directly.
                post_facts = self.post.facts.copy()
                old = post_facts.get("acc")
                post_facts["acc"] = (
                    Event(old.id, old.cycle, old.pc, old.slot, old.instruction) if old else None
                )
                facts["post"] = Event(
                    self.post.id,
                    self.post.cycle,
                    self.post.pc,
                    self.post.slot,
                    self.post.instruction,
                    post_facts,
                )
                facts["scale_lanes"] = self.post_scale
        if slot == "cond":
            diamond = self.diamond(plan, kwargs)
            if diamond:
                facts["diamond"] = diamond
        self.current = (slot, plan.inst_name, facts)
        self.accesses = []

    def retired(self, issued: IssueFacts, result: Any) -> None:
        """Close the event :meth:`issued` opened for this op."""
        if issued.plan.inst_name == "NOP" or issued.slot == "break":
            return
        self.finish(self.current, issued.alias)

    def scaled_window(self, bank, start, length):
        """Lane provenance follows snapshot loads, never numerical equality alone."""
        sources = {e.facts["bank_index"]: e for e in self.window(bank, start, length)}
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

    def mult_scale(self, name, kw, encoded, facts):
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
                    scaled = self.scaled_window(bank, start, LANES)
        elif name in ("MULT.VE", "MULT.EE") and cr_scalar(self.state.regfile, kw["cr_idx"]) == 1:
            length = 1 if name == "MULT.EE" else LANES
            scaled = self.scaled_window("ra", kw["ra_idx"], length)
            if length == 1:
                scaled = full if scaled else 0
        elif (
            name == "MULT.RC.VE"
            and kw["src"] >= 16
            and cr_scalar(self.state.regfile, kw["src"] - 16) == 1
        ):
            scaled = self.scaled_window("ring", kw["rc_idx"], LANES)
        # Mask padding is zero or infinity, both invariant under positive scaling.
        return scaled | (full ^ facts["mask"])

    def accumulator_scale(self, name, kw):
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

    def window(self, bank, start, length):
        """Associate snapshot operands with loads, including cyclic slot crossings."""
        capacity = LANES * (4 if bank == "ring" else 2)
        slots = {(start + i) % capacity // LANES for i in range(length)}
        result = []
        registers = vector_registers(self.state)
        name = registers.r_cyclic if bank == "ring" else registers.ra
        raw = self.cycle_source().snapshot.raw_readonly(name)
        if self.state.regfile.has_mutable_export(name):
            return []
        size = row_size_bytes(self.state.fmt)
        for slot in sorted(slots):
            item = self.snapshot_rows.get((bank, slot))
            if item and raw[slot * size : (slot + 1) * size] == item[1]:
                result.append(item[0])
        return result

    def constants(self, sources):
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
            element_fmt = self.state.fmt
            if len(data) != row_size_bytes(element_fmt):
                continue
            element = element_width_bytes(element_fmt)
            codec = self.state.codec
            one = codec.one()
            zero = bytes(element)
            if data != data[:element] * LANES and data.replace(one, b"").replace(zero, b""):
                source.facts["constant_properties"] = None
                continue
            values = codec.decode(data, LANES).tolist()
            if len(values) != LANES or not all(math.isfinite(x) for x in values):
                continue
            uniform = all(x == values[0] for x in values)
            binary = all(x in (0, 1) for x in values) and 0 in values and 1 in values
            properties = {
                "source": source,
                "uniform": uniform,
                "value": values[0],
                "binary": binary,
                "declared": source.facts.get("declared", False),
                "fractional": uniform and values[0] != int(values[0]),
                "log2e": uniform and data[:4] == struct.pack("<f", math.log2(math.e))
                if self.state.fmt is ElementFormat.FP32
                else False,
            }
            # Cache scalar properties only (no self-reference).
            source.facts["constant_properties"] = {
                k: v for k, v in properties.items() if k != "source"
            }
            result.append(properties)
        return tuple(result)

    def finish(self, token, alias=None):
        if token is None:
            return
        slot, name, facts = token
        if alias:
            facts["alias"] = alias
        facts["accesses"] = tuple((kind, addr, len(data)) for kind, addr, data, _ in self.accesses)
        facts["read_bytes"] = sum(len(data) for kind, _, data, _ in self.accesses if kind == "read")
        facts["write_bytes"] = sum(
            len(data) for kind, _, data, _ in self.accesses if kind == "write"
        )
        if slot == "load" and self.accesses:
            _, address, data, origin = self.accesses[0]
            blocks = range(
                address // XMEM_WIDTH_BYTES,
                (address + len(data) + XMEM_WIDTH_BYTES - 1) // XMEM_WIDTH_BYTES,
            )
            versions = tuple(
                self.memory_versions.get(block, self.cleared_epoch) for block in blocks
            )
            previous = self.last_reads.get((address, len(data)))
            if previous and previous[2] != versions:
                previous = None
            # A row the program never wrote was placed by setup and is immutable
            # for the whole run: that is what makes it a constant. The observer
            # only attaches once execution begins, so an all-zero version tuple
            # means exactly that, and no harness-side declaration is needed.
            facts.update(
                address=address,
                data=data,
                origin=origin,
                previous_read=previous,
                declared=not any(versions),
            )
            if origin and origin.instruction == "ST.AAQ":
                store_address = origin.facts["accesses"][0][1]
                if (address - store_address) % 4 == 0:
                    facts["scale_lanes"] = (
                        origin.facts.get("scale_lanes", 0) >> ((address - store_address) // 4)
                    ) & ((1 << (len(data) // 4)) - 1)
            if name == "LD.MULT":
                facts.update(bank="ra", bank_index=facts["operands"]["dest"])
            elif name == "LD.MULT.CYCLIC":
                facts.update(bank="ring", bank_index=facts["operands"]["index"] // LANES)
        self.next_id += 1
        event = Event(self.next_id, self.cycle, self.pc, slot, name, facts)
        self.current = None
        if slot == "load" and self.accesses:
            self.last_reads[(facts["address"], len(facts["data"]))] = (
                event.cycle,
                self.chain,
                versions,
            )
            kw = facts["operands"]
            if name == "LD.MULT":
                self.rows[("ra", kw["dest"])] = (event, facts["data"])
            elif name == "LD.MULT.CYCLIC":
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
                for block in range(
                    address // XMEM_WIDTH_BYTES,
                    (address + len(data) + XMEM_WIDTH_BYTES - 1) // XMEM_WIDTH_BYTES,
                ):
                    self.stores[block] = stored
        self.cycle_events.append(event)
        self.profile.emit(event)

    def end(self):
        self.next_id += 1
        event = Event(
            self.next_id,
            self.cycle,
            self.pc,
            "cycle",
            "CYCLE",
            {"events": tuple(self.cycle_events), "next_pc": self.state.program_counter},
        )
        # Cycle summaries are control evidence, not retired instructions.
        self.profile.observe(event)
