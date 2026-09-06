"""ISA alias specification — declarative table of missing-instruction stand-ins.

The ISA has no vector move, add, subtract, broadcast or gather, and every
``ACC.*``/``AGG.*`` op consumes ``MULT_RES``. There is therefore no path into the
accumulator that avoids the multiplier, so kernels fake those operations with a
multiply against a hardwired 1.0. Those cycles bill as "MULT active" while
retiring zero MACs, which is why ``mult%`` is not MAC utilization.

This table declares those idioms so the emulator can count them. It holds **no
instruction metadata** — only references into ``INSTRUCTION_SPEC``, checked by
``validate_isa_alias_spec()`` at import. See ``docs/content/isa-alias-catalogue.md``
for the full prose audit and the cycle costs.

Structure:

    ISA_ALIAS_SPEC = {
        "<alias id>": {
            "missing_op": "MOV.RC",   # what the ISA should have
            "summary":    "...",      # one line, for humans reading the table
            "matches": [
                {"slot": "mult", "instruction": "MULT.RC.VE", "where": [...]},
                ...
            ],
        },
    }

Constraint vocabulary for ``where`` (a closed set — extend deliberately):

    {"operand": "src", "scalar_is": 1}
        That operand supplies a constant 1.0 from a CR. Dtype-aware: the
        emulator compares against the dtype's own encoding of 1.0. Only 1 is
        expressible — see ``_SCALAR_VALUES``. An operand naming an LR selects
        an element of R0/R1, which is *data*, and never matches: a quantized
        weight that happens to equal 1 is a real MAC, not a pass-through.

    {"operand": "mask_offset", "immediate_ne": 0}
        Raw immediate field differs from the literal. Only operands declared
        as MultMaskOffsetImmediate are supported in v1; register operands are
        rejected, even when they have no automatic read.

    {"vector_ones": True}
        A MULT.RC.VV or MULT.RC.VE vector operand comes from an explicitly
        declared ONES load. Provenance follows register snapshots and is
        revoked on overwrites; operand bytes must still encode numeric one.

    {"masked_window": True}
        A nonzero mask slot, or a nonempty mask-zero window narrower than the
        instruction's declared destination span. A mask matching valid_elements
        is ordinary padding, not a routing window.

    {"co_slot": "acc", "instruction_in": ("ACC.ADD",)}
        Another slot **in the same bundle** issues one of these. This is what
        separates the aliases: A1/A2/A3/A4/A6 all use an identical
        ``MULT.RC.VE`` with a 1.0 scalar, and only the co-issued acc op says
        which missing operation is being emulated.

    {"hardware_only": True}
        The instruction's *slot* is flagged ``"hardware": False`` in
        ``SLOT_METADATA`` — simulation-only, no silicon encoding. Matches
        unconditionally; no runtime value is read.

Every match must carry exactly one of ``scalar_is``, ``vector_ones`` or
``hardware_only``: these are the ways the emulator can decide a match, and
the validator rejects anything else so a new entry cannot silently be treated
as an identity test it never asked for.
Scalar constraints must name the instruction's supported multiplicand. A match
can have at most one immediate constraint and one co-slot constraint, matching
the compiled representation. ``hardware_only`` must stand alone.

**Order matters: first match wins.** Entries are ordered most-specific first so
that each identity multiply is attributed to exactly one alias, which makes the
per-alias hit counts a partition of ``mult_identity_cycles`` rather than
overlapping tallies.
"""

from __future__ import annotations

from .instruction_spec import INSTRUCTION_SPEC, SLOT_METADATA

# Only 1 is implementable today: the emulator tests a multiplicand against the
# dtype's encoding of 1.0, which is the whole point of the table. Widening this
# means teaching the emulator to encode other constants per dtype.
_SCALAR_VALUES = (1,)

_CONSTRAINT_KEYS = {
    frozenset({"operand", "scalar_is"}),
    frozenset({"operand", "immediate_ne"}),
    frozenset({"co_slot", "instruction_in"}),
    frozenset({"hardware_only"}),
    frozenset({"vector_ones"}),
    frozenset({"masked_window"}),
}

# The acc-slot ops that consume MULT_RES, grouped by the operation the
# surrounding idiom is really performing.
_ACC_ADD_RUNNING = ("ACC.ADD",)
_ACC_SUB = ("ACC.SUB", "ACC.SUB.FIRST")
_ACC_REDUCE = ("AGG.SUM", "AGG.SUM.FIRST", "AGG.MAX", "AGG.MAX.FIRST",
               "ACC.MAX", "ACC.MAX.FIRST")

# Every mult form that takes a scalar multiplicand, and the operand carrying it.
# The emulator reads this too, to know which operand to probe for a constant
# 1.0 — keep it the only place that mapping is written down. The operand's
# *type* is not restated here; it is read from INSTRUCTION_SPEC.
SCALAR_MULT_OPERANDS: dict[str, str] = {
    "MULT.RC.VE": "src",
    "MULT.VE": "cr_idx",
    "MULT.EE": "cr_idx",
}

_SCALAR_MULTS = tuple(SCALAR_MULT_OPERANDS.items())
VECTOR_IDENTITY_MULTS = ("MULT.RC.VE", "MULT.RC.VV")


def _identity_mult_matches(*, where_extra=(), only=None, exclude=()):
    """Build CR-scalar and declared-vector identity matches in precedence order."""
    scalar = [
        {
            "slot": "mult",
            "instruction": inst,
            "where": [{"operand": operand, "scalar_is": 1}, *where_extra],
        }
        for inst, operand in _SCALAR_MULTS
        if (only is None or inst in only) and inst not in exclude
    ]
    vector = [
        {"slot": "mult", "instruction": inst,
         "where": [{"vector_ones": True}, *where_extra]}
        for inst in VECTOR_IDENTITY_MULTS
        if (only is None or inst in only) and inst not in exclude
    ]
    return scalar + vector


ISA_ALIAS_SPEC: dict[str, dict] = {
    # Order is precedence. Constraint-carrying entries come first so that the
    # operation actually being emulated wins over the mechanism used to emulate
    # it: a MULT.EE co-issued with ACC.SUB is a vector subtract (A3) that happens
    # to broadcast, not a bare broadcast (A5).
    "A6_GATHER_SCATTER": {
        "missing_op": "GATHER / SCATTER",
        "summary": "Masked pass-through: rc_idx picks the read window, mask_offset the write window.",
        "matches": _identity_mult_matches(
            where_extra=({"masked_window": True},),
        ) + [{"slot": "mult", "instruction": "MULT.RC.VV",
              "where": [{"vector_ones": True},
                        {"co_slot": "acc", "instruction_in": ("ACC.STRIDE",)}]}],
    },
    "A4_AGG_RC": {
        "missing_op": "AGG.SUM.RC / AGG.MAX.RC",
        "summary": "Reduce a vector register; the reduction can only read MULT_RES.",
        "matches": _identity_mult_matches(
            where_extra=({"co_slot": "acc", "instruction_in": _ACC_REDUCE},),
        ),
    },
    "A3_SUB_VV": {
        "missing_op": "SUB.VV / NEG",
        "summary": "Vector subtract; ACC.SUB exists but its operand must cross the multiplier.",
        "matches": _identity_mult_matches(
            where_extra=({"co_slot": "acc", "instruction_in": _ACC_SUB},),
        ),
    },
    "A2_ADD_VV": {
        "missing_op": "ADD.VV",
        "summary": "Vector add, one addend per hit (running ACC.ADD half of the idiom).",
        "matches": _identity_mult_matches(
            where_extra=({"co_slot": "acc", "instruction_in": _ACC_ADD_RUNNING},),
        ),
    },
    # --- catch-alls for a 1.0 multiply no constraint above claimed ----------
    "A5_BCAST": {
        "missing_op": "BCAST",
        "summary": "Broadcast one element to 128 lanes; MULT.EE is the only splat path.",
        "matches": _identity_mult_matches(only=("MULT.EE",)),
    },
    "A1_MOV_RC": {
        "missing_op": "MOV.RC",
        "summary": "Move a vector into R_ACC; nothing reaches the accumulator without a multiply.",
        "matches": _identity_mult_matches(exclude=("MULT.EE",)),
    },
    "A16_STR_ACC_REG": {
        "missing_op": "STR_ACC (hardware)",
        "summary": "Simulation-only accumulator store; has no silicon encoding.",
        "matches": [
            {
                "slot": "acc_store",
                "instruction": "STR_ACC_REG",
                "where": [{"hardware_only": True}],
            },
        ],
    },
}


def _operand_names(slot: str, instruction: str) -> set[str]:
    return {op["name"] for op in INSTRUCTION_SPEC[slot][instruction]["operands"]}


def validate_isa_alias_spec() -> None:
    """Check every reference into ``INSTRUCTION_SPEC`` resolves. Called at import."""
    signatures = set()
    for alias_id, entry in ISA_ALIAS_SPEC.items():
        for required in ("missing_op", "summary", "matches"):
            if required not in entry:
                raise ValueError(f"alias {alias_id!r} is missing {required!r}")
        if not entry["matches"]:
            raise ValueError(f"alias {alias_id!r} declares no matches")

        for match in entry["matches"]:
            slot, instruction = match.get("slot"), match.get("instruction")
            if slot not in INSTRUCTION_SPEC:
                raise ValueError(f"alias {alias_id!r} names unknown slot {slot!r}")
            if instruction not in INSTRUCTION_SPEC[slot]:
                raise ValueError(
                    f"alias {alias_id!r} names unknown instruction "
                    f"{instruction!r} in slot {slot!r}"
                )
            operands = _operand_names(slot, instruction)

            # The emulator decides a match by testing the multiplicand against
            # 1.0, except for hardware_only entries which fire unconditionally.
            # A match with neither would silently be treated as the former.
            gates = sum(
                1
                for c in match["where"]
                if "scalar_is" in c or c.get("hardware_only") or "vector_ones" in c
            )
            if gates != 1:
                raise ValueError(
                    f"alias {alias_id!r} match on {slot}/{instruction} must carry "
                    "exactly one of scalar_is, vector_ones or hardware_only "
                    f"(found {gates})"
                )

            seen_keys = set()
            for constraint in match["where"]:
                keys = frozenset(constraint)
                if keys not in _CONSTRAINT_KEYS:
                    raise ValueError(
                        f"alias {alias_id!r} uses unsupported constraint "
                        f"{sorted(keys)}; supported: "
                        f"{[sorted(k) for k in _CONSTRAINT_KEYS]}"
                    )
                if keys in seen_keys:
                    raise ValueError(
                        f"alias {alias_id!r} repeats constraint kind {sorted(keys)}"
                    )
                seen_keys.add(keys)
                if "vector_ones" in constraint and (
                    constraint["vector_ones"] is not True or slot != "mult"
                    or instruction not in VECTOR_IDENTITY_MULTS
                ):
                    raise ValueError(f"alias {alias_id!r}: unsupported vector_ones constraint")
                if "masked_window" in constraint and (
                    constraint["masked_window"] is not True or slot != "mult"
                    or not {"mask_offset", "mask_shift", "cr_idx"} <= operands
                ):
                    raise ValueError(f"alias {alias_id!r}: unsupported masked_window constraint")
                if "hardware_only" in constraint and (
                    constraint["hardware_only"] is not True or len(match["where"]) != 1
                ):
                    raise ValueError(
                        f"alias {alias_id!r}: hardware_only must be True and stand alone"
                    )
                if "operand" in constraint:
                    name = constraint["operand"]
                    if name not in operands:
                        raise ValueError(
                            f"alias {alias_id!r}: {instruction} has no operand "
                            f"{name!r} (has {sorted(operands)})"
                        )
                    if "immediate_ne" in constraint:
                        operand = next(
                            op for op in INSTRUCTION_SPEC[slot][instruction]["operands"]
                            if op["name"] == name
                        )
                        # This is the only immediate type on the supported
                        # scalar multiplies. Require its spec type, not just
                        # the absence of a read: CR indices also lack reads.
                        if operand["type"] != "MultMaskOffsetImmediate" or "read" in operand:
                            raise ValueError(
                                f"alias {alias_id!r}: {slot}/{instruction}.{name} "
                                "is not a supported raw immediate operand"
                            )
                    if (
                        "scalar_is" in constraint
                        and (slot != "mult" or name != SCALAR_MULT_OPERANDS.get(instruction))
                    ):
                        raise ValueError(
                            f"alias {alias_id!r}: {slot}/{instruction}.{name} "
                            "is not a supported scalar multiplicand"
                        )
                    if (
                        "scalar_is" in constraint
                        and constraint["scalar_is"] not in _SCALAR_VALUES
                    ):
                        raise ValueError(
                            f"alias {alias_id!r}: scalar_is must be one of "
                            f"{_SCALAR_VALUES}"
                        )
                if "co_slot" in constraint:
                    co_slot = constraint["co_slot"]
                    if co_slot not in INSTRUCTION_SPEC:
                        raise ValueError(
                            f"alias {alias_id!r} names unknown co_slot {co_slot!r}"
                        )
                    for name in constraint["instruction_in"]:
                        if name not in INSTRUCTION_SPEC[co_slot]:
                            raise ValueError(
                                f"alias {alias_id!r} names unknown instruction "
                                f"{name!r} in co_slot {co_slot!r}"
                            )
                if constraint.get("hardware_only") and SLOT_METADATA[slot].get(
                    "hardware", True
                ):
                    raise ValueError(
                        f"alias {alias_id!r} uses hardware_only on slot {slot!r}, "
                        "which is not flagged \"hardware\": False"
                    )

            signature = (slot, instruction, frozenset(
                frozenset(
                    (key, frozenset(value) if key == "instruction_in" else value)
                    for key, value in constraint.items()
                )
                for constraint in match["where"]
            ))
            if signature in signatures:
                raise ValueError(f"alias {alias_id!r} repeats an identical match")
            signatures.add(signature)


def aliases_for(slot: str, instruction: str) -> tuple[tuple[str, dict], ...]:
    """Return ``(alias_id, match)`` pairs for one instruction, in table order."""
    return tuple(
        (alias_id, match)
        for alias_id, entry in ISA_ALIAS_SPEC.items()
        for match in entry["matches"]
        if match["slot"] == slot and match["instruction"] == instruction
    )


validate_isa_alias_spec()
