"""Built-in detectors. Additions register here; the profiling engine has no IDs."""

from __future__ import annotations

from ipu_emu.alias_profile import AliasDefinition, DetectorRegistry, Match
from ipu_emu.isa_alias_spec import ISA_ALIAS_SPEC
from ipu_isa.activations import AAQ_EXP2, ACTIVATION_EXP2, ACTIVATION_IDENTITY
from ipu_isa.geometry import LANES, R_CYCLIC_ELEMENTS

#: A MULT mask with every lane on.
_ALL_LANES = (1 << LANES) - 1


#: The AaQ store instructions, and the encoding of ``exp2`` in each.
_AAQ_STORES = ("ACTIVATE.QUANTIZE", "ACTIVATE.EXACT")
_EXP2 = {"ACTIVATE.QUANTIZE": AAQ_EXP2, "ACTIVATE.EXACT": ACTIVATION_EXP2}


def transfer(load):
    """``(store, load)`` when *load* reads the row an identity AaQ store wrote."""
    if load.instruction not in ("LD.MULT", "LD.MULT.CYCLIC"):
        return ()
    store = load.facts.get("origin")
    if store is None or store.instruction not in _AAQ_STORES:
        return ()
    stores = store.facts.get("accesses", ())
    if (
        store.facts["operands"].get("activation_fn") == ACTIVATION_IDENTITY
        and stores
        and load.facts["address"] == stores[0][1]
    ):
        return (store, load)
    return ()


class Identity:
    slots = ("cycle",)

    def __init__(self, alias):
        self.alias = alias

    def observe(self, event):
        if event.facts.get("alias") == self.alias:
            yield Match(self.alias, (event,))
        if event.slot == "cycle":
            issued = event.facts["events"]
            for mult in issued:
                if mult.facts.get("alias") == self.alias and mult.slot == "mult":
                    consumers = tuple(
                        e
                        for e in issued
                        if e.slot == "acc"
                        and e.facts.get("mult") is mult
                        and e.instruction.startswith(
                            ("ACC.ADD", "ACC.SUB", "ACC.MAX", "AGG.", "ACC.STRIDE")
                        )
                    )
                    yield Match(self.alias, (mult, *consumers))


class Transfer:
    slots = ("load",)

    def observe(self, event):
        if event.slot == "load" and (evidence := transfer(event)):
            yield Match(
                "A8_MOV_ACC", evidence, metrics={"lossy_transfers": int(evidence[0].facts["lossy"])}
            )


class AccMultiply:
    slots = ("acc",)

    def observe(self, event):
        if event.instruction != "ACC.ADD.FIRST":
            return
        mult = event.facts.get("mult")
        if not mult or mult.facts.get("identity") or mult.instruction == "MULT.RC.VS":
            return
        for source in mult.facts.get("sources", ()):
            evidence = transfer(source)
            kw = mult.facts["operands"]
            enc = mult.facts["encoded"]
            full_vector = (
                source.facts.get("bank") == "ring" and "rc_idx" in kw and kw["rc_idx"] % LANES == 0
            ) or (
                source.facts.get("bank") == "ra"
                and (
                    ("ra" in enc and enc["ra"] == source.facts["bank_index"])
                    or (mult.instruction == "MULT.VE" and kw["ra_idx"] % LANES == 0)
                )
            )
            if evidence and full_vector:
                old_acc = evidence[0].facts.get("acc")
                if old_acc and old_acc.id == event.facts.get("previous_acc_id"):
                    yield Match("A11_ACC_MUL", (*evidence, mult, event))
                    return


class Exponential:
    slots = ("acc", "aaq", "aaq_exact")

    def __init__(self):
        self.chain = ()
        self.constant = None

    def observe(self, event):
        if event.slot == "acc":
            mult = event.facts.get("mult")
            if event.instruction == "ACC.ADD.FIRST":
                self.chain, self.constant = (), None
                if mult and mult.instruction == "MULT.RC.VV":
                    for constant in mult.facts.get("constants", ()):
                        if constant["log2e"]:
                            self.constant = constant
                            self.chain = (constant["source"], mult, event)
                            break
            elif (
                self.chain
                and event.instruction in ("ACC.ADD", "ACC.SUB")
                and event.facts["previous_acc_id"] == self.chain[-1].id
                and len(self.chain) < 16
            ):
                self.chain += ((mult,) if mult else ()) + (event,)
            else:
                self.chain, self.constant = (), None
            return
        exp2 = _EXP2.get(event.instruction)
        if exp2 is None or event.facts["operands"].get("activation_fn") != exp2:
            return
        acc = event.facts.get("acc")
        if not acc or not self.chain or self.chain[-1].id != acc.id:
            return
        offset = len(self.chain) > 3
        yield Match(
            "A9_EXP",
            (*self.chain, event),
            subtype="rebased_with_offset" if offset else "direct",
            status="verified"
            if self.constant["declared"] and (not offset or event.facts.get("input_scale_verified"))
            else "candidate",
            reason=(
                "Rebase and offset share proven log2(e) scale"
                if event.facts.get("input_scale_verified")
                else "Rebased value feeds exp2; additional offset provenance unproven"
            )
            if offset
            else "FP32 log2(e) broadcast feeds exp2 from a row the program never wrote",
        )


class Fractional:
    slots = ("mult",)

    def observe(self, event):
        if event.instruction != "MULT.RC.VV" or not event.facts.get("wide_fp32"):
            return
        for constant in event.facts.get("constants", ()):
            if constant["fractional"]:
                yield Match(
                    "A10_FRACTIONAL_SCALAR",
                    (constant["source"], event),
                    status="verified" if constant["declared"] else "candidate",
                    metrics={"vector_constant_uses": 1},
                    reason="Uniform fractional vector operand",
                )


class Lanes:
    slots = ("mult",)

    def observe(self, event):
        if event.slot == "mult":
            f = event.facts
            yield Match(
                "A13_NARROW_MULT",
                (event,),
                subtype=f"lanes_{f['active_lanes']}",
                metrics={
                    "declared_lane_capacity": f["declared_width"] or LANES,
                    "masked_physical_lanes": LANES - f["mask"].bit_count(),
                    "undeclared_width_uses": int(f["declared_width"] == 0),
                },
            )


class Reuse:
    slots = ("load",)

    def observe(self, event):
        if event.slot == "load" and (previous := event.facts.get("previous_read")):
            yield Match(
                "A14_MULTIPLE_ACC",
                (event,),
                subtype="unchanged_region_reread",
                metrics={
                    "repeated_read_bytes": event.facts["read_bytes"],
                    "read_distance_cycles": event.cycle - previous[0],
                    "different_accumulator_chain": int(event.facts["chain"] != previous[1]),
                },
                reason="Repeated read of the same unchanged region; no hardware savings implied",
            )


class Segmented:
    """Observe disjoint cyclic source windows folded into one accumulator.

    This is the catalogue's partition fold, not a claim that CR.partition itself
    performs a segmented reduction. Irregular windows remain candidates.
    """

    slots = ("acc",)

    def __init__(self):
        self.parts = []
        self.origin = None
        self.windows = set()
        self.rotation = []
        self.rotation_op = None

    def observe(self, event):
        if event.slot != "acc":
            return
        mult = event.facts.get("mult")
        if event.instruction in ("ACC.ADD.FIRST", "ACC.MAX.FIRST"):
            self.parts, self.windows, self.origin = [], set(), None
            self.rotation = []
            self.rotation_op = event.instruction.removesuffix(".FIRST")
        elif event.instruction not in ("ACC.ADD", "ACC.MAX"):
            self.parts, self.windows, self.origin = [], set(), None
            self.rotation = []
            return
        if not mult:
            self.parts, self.windows, self.origin = [], set(), None
            self.rotation = []
            return
        sources = [s for s in mult.facts.get("sources", ()) if s.facts.get("bank") == "ring"]
        moves = [(source, transfer(source) or (source,)) for source in sources]
        if not moves:
            return
        width = mult.facts["active_lanes"]
        raw_start = mult.facts["operands"].get("rc_idx", 0) % R_CYCLIC_ELEMENTS
        moves.sort(key=lambda item: item[0].facts["bank_index"] != raw_start // LANES)
        source, evidence = moves[0]
        if source.facts["bank_index"] != raw_start // LANES:
            self.parts, self.windows, self.rotation = [], set(), []
            return
        if self.parts and self.parts[-1].instruction.removesuffix(
            ".FIRST"
        ) != event.instruction.removesuffix(".FIRST"):
            self.parts, self.windows = [], set()
        start = raw_start % LANES
        origin = evidence[-2].id if len(evidence) > 1 else source.id
        if self.origin is not None and self.origin != origin:
            self.parts, self.windows = [], set()
            self.rotation = []
        self.origin = origin
        if width == LANES and mult.facts["mask"] == _ALL_LANES:
            if not mult.facts.get("identity") or len(evidence) == 1:
                self.rotation = []
                return
            raw_start = mult.facts["operands"].get("rc_idx", 0) % R_CYCLIC_ELEMENTS
            needed = {(raw_start + i) % R_CYCLIC_ELEMENTS // LANES for i in range(LANES)}
            proven = {
                s.facts["bank_index"]
                for s, path in moves
                if len(path) > 1 and path[-2].id == origin
            }
            if (
                proven != needed
                or self.rotation_op != event.instruction.removesuffix(".FIRST")
                or (not self.rotation and start != 0)
            ):
                self.rotation = []
                return
            if self.rotation:
                step = start if len(self.rotation) == 1 else self.rotation[1][0]
                if not step or LANES % step or start != len(self.rotation) * step:
                    self.rotation = []
                    return
            self.rotation.append(
                (start, tuple(e for _, path in moves for e in path) + (mult, event))
            )
            if len(self.rotation) > 1 and len(self.rotation) * self.rotation[1][0] == LANES:
                yield Match(
                    "A7_REDUCE_SEG",
                    tuple(e for _, path in self.rotation for e in path),
                    subtype="rotating_partition_fold",
                    metrics={"partitions": len(self.rotation)},
                    reason="Complete uniform rotation of duplicated accumulator data, reduced and broadcast",
                )
                self.rotation = []
            return
        if not width or width == LANES or LANES % width or start % width:
            if event.instruction == "ACC.ADD":
                yield Match(
                    "A7_REDUCE_SEG",
                    (*evidence, mult, event),
                    status="candidate",
                    reason="Accumulator round trip followed by a fold; disjoint partition geometry unproven",
                )
            return
        lanes = {(start + i) % LANES for i in range(width)}
        # Require an unshifted contiguous output mask, not merely the same popcount.
        if mult.facts["mask"] & _ALL_LANES != (1 << width) - 1 or lanes & self.windows:
            self.parts, self.windows = [], set()
            return
        self.windows |= lanes
        self.parts.extend((mult, event))
        if len(self.windows) == LANES and len(self.parts) > 2:
            yield Match(
                "A7_REDUCE_SEG",
                (*evidence, *self.parts),
                subtype="partition_fold",
                metrics={"partitions": len(self.parts) // 2},
                reason="One input row accumulated across disjoint source partitions; real multiplies remain counted",
            )
            self.parts, self.windows, self.origin = [], set(), None


# The slots whose address operands an earlier LR update can feed (A12).
# A set of slots, not an execution order: that is ipu_isa.timing.SLOT_EXECUTION_ORDER.
_ADDRESS_UPDATE_CONSUMERS = frozenset({"mult", "load"})


class Addressing:
    slots = ("cycle",)

    def __init__(self):
        self.updates = {}

    def observe(self, event):
        if event.slot != "cycle":
            return
        events = event.facts["events"]
        written = {reg for e in events for reg in e.facts.get("lr_writes", ())}
        for consumer in events:
            reads = {
                reg
                for name, reg in consumer.facts.get("lr_reads", {}).items()
                if name in ("offset", "base", "index", "rc_idx", "ra_idx", "src")
                and (reg not in written or name in consumer.facts.get("snapshot_reads", ()))
            }
            if consumer.slot in _ADDRESS_UPDATE_CONSUMERS:
                for reg in reads:
                    prior = self.updates.get(reg)
                    if prior and prior.cycle < event.cycle:
                        yield Match(
                            "A12_ADDRESSING",
                            (prior, consumer),
                            subtype="address_update",
                            metrics={"intervening_cycles": event.cycle - prior.cycle - 1},
                        )
                        self.updates.pop(reg, None)
            if consumer.slot == "mult":
                for source in consumer.facts.get("sources", ()):
                    if not source.facts.get("first_mult_use_recorded"):
                        source.facts["first_mult_use_recorded"] = True
                        yield Match(
                            "A12_ADDRESSING",
                            (source, consumer),
                            subtype="load_to_mult",
                            metrics={"load_use_distance_cycles": consumer.cycle - source.cycle},
                            reason="Observed dependency distance, not attributed pipeline stalls",
                        )
        # Only isolated LR bundles establish the post-increment observation.
        isolated = bool(events) and all(e.slot == "lr" for e in events)
        for e in events:
            for reg in e.facts.get("lr_writes", ()):
                self.updates.pop(reg, None)
                enc = e.facts["encoded"]
                if isolated and (
                    e.instruction == "INC"
                    or (e.instruction == "ADD" and reg in (enc["src_a"], enc["src_b"]))
                ):
                    self.updates[reg] = e


class Selection:
    slots = ("mult", "lr", "cycle")

    def __init__(self):
        self.adds = {}
        self.branch = None

    def observe(self, event):
        if event.instruction == "MULT.RC.VV":
            for c in event.facts.get("constants", ()):
                if c["binary"]:
                    yield Match(
                        "A15_SELECT",
                        (c["source"], event),
                        subtype="vector_mask",
                        status="verified" if c["declared"] else "candidate",
                        reason="Mixed zero/one operand from a row the program never wrote",
                    )
        if event.slot == "lr":
            enc = event.facts["encoded"]
            dest = enc.get("dest")
            for reg in event.facts.get("lr_writes", ()):
                previous = self.adds.pop(reg, None)
                if event.instruction == "ADD" and enc.get("src_a") == dest:
                    if (
                        previous
                        and previous.pc + 1 == event.pc
                        and previous.cycle + 1 == event.cycle
                        and enc.get("src_b", 16) < 16
                        and enc["src_b"] != dest
                        and previous.facts["encoded"] == enc
                        and previous.facts["operands"].get("src_b")
                        == event.facts["operands"].get("src_b")
                    ):
                        yield Match(
                            "A15_SELECT",
                            (previous, event),
                            subtype="scalar_add_chain",
                            metrics={"additional_additions": 1},
                            reason="Two dependent additions of an unchanged scalar, modulo LR width",
                        )
                    self.adds[reg] = event
        if event.slot == "cycle":
            events = event.facts["events"]
            if self.branch:
                branch, destination, join, evidence = self.branch
                if event.pc == join:
                    if any(
                        e.slot == "lr" and destination in e.facts.get("lr_writes", ())
                        for e in evidence
                    ):
                        yield Match("A15_SELECT", (branch, *evidence), subtype="conditional_select")
                    self.branch = None
                elif len(evidence) >= 8:
                    self.branch = None
                else:
                    evidence.extend(events)
            for e in events:
                if "diamond" in e.facts:
                    dest, join = e.facts["diamond"]
                    self.branch = (e, dest, join, [])


def default_registry():
    registry = DetectorRegistry()
    for key, spec in ISA_ALIAS_SPEC.items():
        registry.register(
            AliasDefinition(key, spec["missing_op"], description=spec["summary"]),
            lambda key=key: Identity(key),
        )
    for key, name, factory in (
        ("A7_REDUCE_SEG", "REDUCE.SEG", Segmented),
        ("A8_MOV_ACC", "MOV.ACC", Transfer),
        ("A9_EXP", "EXP", Exponential),
        ("A10_FRACTIONAL_SCALAR", "Fractional CR scalar", Fractional),
        ("A11_ACC_MUL", "ACC.MUL", AccMultiply),
        ("A12_ADDRESSING", "Addressing and forwarding", Addressing),
        ("A13_NARROW_MULT", "Narrow MULT", Lanes),
        ("A14_MULTIPLE_ACC", "Multiple accumulators / MULT.OUTER", Reuse),
        ("A15_SELECT", "Mask, select and scalar multiply", Selection),
    ):
        registry.register(
            AliasDefinition(
                key, name, modes=("fp32",) if factory in (Exponential, Fractional) else ()
            ),
            factory,
        )
    return registry
