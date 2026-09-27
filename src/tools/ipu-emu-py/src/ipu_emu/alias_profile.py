"""Extensible, opt-in ISA measurements. No detector changes architectural stats.

Detectors consume completed execution events and return evidence-backed matches.
Register a factory on a registry copy to add an alias without editing the runner.
An observation is a measured motif, not a claim that its cycles can be removed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class AliasDefinition:
    id: str
    name: str
    unit: str = "occurrences"
    description: str = ""
    modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Event:
    id: int
    cycle: int
    pc: int
    slot: str
    instruction: str
    facts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Match:
    alias: str
    events: tuple[Event, ...]
    subtype: str = "default"
    status: str = "verified"
    metrics: dict[str, int] = field(default_factory=dict)
    reason: str = ""


class Detector(Protocol):
    def observe(self, event: Event) -> Iterable[Match]: ...


class DetectorRegistry:
    def __init__(self):
        self.entries: dict[str, tuple[AliasDefinition, Callable[[], Detector]]] = {}

    def register(self, definition: AliasDefinition, factory: Callable[[], Detector]):
        if not definition.id or definition.id in self.entries:
            raise ValueError(f"duplicate or empty alias id: {definition.id!r}")
        if not callable(factory):
            raise TypeError("detector factory must be callable")
        self.entries[definition.id] = (definition, factory)
        return self

    def copy(self):
        result = DetectorRegistry()
        result.entries.update(self.entries)
        return result


class AliasProfile:
    """Per-state profiler; independent detector instances, bounded example storage.

    Membership uses an integer bitset per event, not full event retention. This
    preserves exact overlap and union counts even for sequences completed later.
    Event membership is linear in executed instructions; examples are bounded.
    """
    def __init__(self, registry: DetectorRegistry | None = None, *, examples: int = 3,
                 metadata: dict | None = None):
        if registry is None:
            from ipu_emu.alias_detectors import default_registry
            registry = default_registry()
        if examples < 0:
            raise ValueError("examples must be nonnegative")
        self.registry = registry.copy()
        self.detectors = [(key, factory()) for key, (_, factory) in self.registry.entries.items()]
        self.listeners = {}
        for owner, detector in self.detectors:
            for slot in getattr(detector, "slots", ("*",)):
                self.listeners.setdefault(slot, []).append((owner, detector))
        self.example_limit = examples
        self.metadata = dict(metadata or {})
        self.results: dict[tuple[str, str, str], dict] = {}
        self.membership: dict[int, int] = {}
        self.cycle_membership: dict[int, int] = {}
        self.bits = {key: 1 << i for i, key in enumerate(self.registry.entries)}
        self.cycles = 0
        self.events = 0
        self.read_bytes = 0
        self.write_bytes = 0
        self.read_regions = set()
        self._bridge = None

    def emit(self, event: Event):
        self.events += 1
        self.read_bytes += event.facts.get("read_bytes", 0)
        self.write_bytes += event.facts.get("write_bytes", 0)
        self.read_regions.update((address, size) for kind, address, size in event.facts.get("accesses", ()) if kind == "read")
        self.observe(event)

    def observe(self, event):
        for owner, detector in (*self.listeners.get(event.slot, ()), *self.listeners.get("*", ())):
            for match in detector.observe(event):
                if match.alias != owner:
                    raise ValueError(f"detector {owner} emitted another alias: {match.alias}")
                self.record(match)

    def record(self, match: Match):
        if match.alias not in self.bits or not match.events:
            raise ValueError("match needs a registered alias and execution evidence")
        if match.status not in ("verified", "candidate"):
            raise ValueError(f"invalid match status: {match.status}")
        key = (match.alias, match.subtype, match.status)
        row = self.results.setdefault(key, dict(count=0, metrics=Counter(), sites=Counter(), examples=[]))
        evidence = {e.id: e for original in match.events
                    for e in (original.facts.get("events", ()) if original.slot == "cycle" else (original,))}
        cycles = {e.cycle for e in (*match.events, *evidence.values())}
        row["count"] += 1
        row["sites"][str(match.events[-1].pc)] += 1
        row["metrics"].update(match.metrics)
        row["metrics"].update({"participating_instructions": len(evidence),
                               "participating_cycles": len(cycles),
                               "elapsed_cycles": max(cycles) - min(cycles) + 1,
                               "read_bytes": sum(e.facts.get("read_bytes", 0) for e in evidence.values()),
                               "write_bytes": sum(e.facts.get("write_bytes", 0) for e in evidence.values()),
                               "active_lanes": sum(e.facts.get("active_lanes", 0) for e in evidence.values())})
        if match.status == "verified":
            bit = self.bits[match.alias]
            for e in evidence.values():
                self.membership[e.id] = self.membership.get(e.id, 0) | bit
            for cycle in cycles:
                self.cycle_membership[cycle] = self.cycle_membership.get(cycle, 0) | bit
        if len(row["examples"]) < self.example_limit:
            row["examples"].append({"reason": match.reason, "events": [
                {"id": e.id, "cycle": e.cycle, "pc": e.pc, "slot": e.slot,
                 "instruction": e.instruction} for e in evidence.values()]})

    def to_dict(self):
        aliases = {}
        for key, (definition, _) in self.registry.entries.items():
            rows = [{"subtype": subtype, "status": status, "count": row["count"],
                     "metrics": dict(row["metrics"]), "sites": dict(row["sites"]),
                     "examples": row["examples"]}
                    for (alias, subtype, status), row in self.results.items() if alias == key]
            aliases[key] = {"name": definition.name, "unit": definition.unit,
                            "description": definition.description,
                            "supported_modes": list(definition.modes),
                            "status": ("unsupported_mode" if definition.modes and self.metadata.get("mode") not in definition.modes
                                       else "observed" if rows else "no_observed_match"), "measurements": rows}
        overlaps = Counter(self.membership.values())
        unique_read_bytes, end = 0, 0
        for address, size in sorted(self.read_regions):
            unique_read_bytes += max(0, address + size - max(address, end))
            end = max(end, address + size)
        return {"schema_version": 1, "metadata": self.metadata,
                "totals": {"cycles": self.cycles, "instructions": self.events,
                           "read_bytes": self.read_bytes, "write_bytes": self.write_bytes,
                           "unique_read_bytes": unique_read_bytes,
                           "unique_profiled_instructions": len(self.membership),
                           "unique_profiled_cycles": len(self.cycle_membership)},
                "aliases": aliases, "overlap_instruction_groups": [
                    {"aliases": [key for key, bit in self.bits.items() if mask & bit], "instructions": count}
                    for mask, count in sorted(overlaps.items())]}

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)

    def format_summary(self):
        lines = ["--- Full ISA measurements (overlapping; no savings estimates) ---"]
        for key, entry in self.to_dict()["aliases"].items():
            if not entry["measurements"]:
                lines.append(f"{key}: {entry['status'].replace('_', ' ')}")
            for row in entry["measurements"]:
                lines.append(f"{key}/{row['subtype']}: {row['count']} {entry['unit']} "
                             f"({row['status']}), {row['metrics']['participating_cycles']} participating cycles")
        return "\n".join(lines)
