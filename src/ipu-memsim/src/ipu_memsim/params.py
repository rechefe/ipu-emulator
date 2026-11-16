"""The parameter registry: the one list of every tunable parameter.

Each parameter is one :class:`Param` entry in :data:`REGISTRY`: its name,
type, unit, default, allowed range, a one-sentence description, and a status
label saying how much the value can be trusted:

- ``CONFIRMED``: the default is what a document in this repository says;
- ``ASSUMED``: the tool's own choice where the documents are silent;
- ``UNKNOWN``: a placeholder for a number only the architects can give.

Everything that handles parameters is driven by this list, so adding a
parameter here is enough for it to be accepted in a config file
(:func:`validate_config`), usable in a sweep (``ipu_memsim.experiment``) and
shown in the menu (``ipu_memsim.ui``):

- :func:`params_for` lists the parameters of one model;
- :func:`build_config` turns ``{name: value}`` into the ``[model]`` / ``[dram]``
  tables a config file has;
- :func:`placeholders_in_use` names the ASSUMED and UNKNOWN parameters behind
  a configuration, for the "toy model" banner.

A parameter belongs to a section: ``model`` (a key of ``[model]``), ``dram``
(a key of ``[dram]``, only for ``cache_unit``) or ``derived`` (not a key
itself: it is computed into other keys; ``xmem_mb`` is the only one).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ASSUMED",
    "CONFIRMED",
    "MODEL_KINDS",
    "REGISTRY",
    "ROW_BYTES",
    "UNKNOWN",
    "Param",
    "build_config",
    "coerce",
    "param",
    "params_for",
    "placeholders_in_use",
    "validate_config",
]

CONFIRMED, ASSUMED, UNKNOWN = "CONFIRMED", "ASSUMED", "UNKNOWN"

#: Bytes of one hardware XMEM row (1024 bits; cache-unit.md section 4).
ROW_BYTES = 128
MB = 2**20

#: The memory models, in menu order.
MODEL_KINDS = ("cache_unit", "ideal", "all_miss", "page_first_touch", "banked", "cache")

_SPEC = "docs/content/specs/cache-unit.md"


@dataclass(frozen=True)
class Param:
    """One tunable parameter.

    Attributes:
        name: Its name in sweeps, the menu and the command line. Unique among
            the parameters of any one model.
        models: The model kinds it belongs to.
        section: ``"model"``, ``"dram"`` or ``"derived"``.
        type: ``int``, ``float``, ``bool`` or ``str``.
        unit: What the number counts (``"cycles"``, ``"rows"``, ...).
        default: The value used when none is given; ``None`` = leave the key out.
        low, high: The allowed range of a number (``None`` = no limit).
        choices: The allowed values of a ``str`` parameter.
        specials: Text values a numeric parameter also takes (``"auto"``).
        description: One plain sentence, shown as the menu's tooltip.
        status: ``CONFIRMED``, ``ASSUMED`` or ``UNKNOWN``.
        source: Where the default comes from, or which question asks for it.
        key: The key in the config table, when it differs from ``name``.
        dram_kinds: For a ``dram`` parameter, the ``dram_kind`` values it
            applies to.
        sweepable: Whether a sweep over it makes sense (numbers and switches).
    """

    name: str
    models: tuple[str, ...]
    section: str
    type: type
    unit: str
    default: Any
    description: str
    status: str
    source: str
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    specials: tuple[str, ...] = ()
    key: str = ""
    dram_kinds: tuple[str, ...] = ()
    sweepable: bool = True

    @property
    def config_key(self) -> str:
        return self.key or self.name


_CU = ("cache_unit",)

REGISTRY: tuple[Param, ...] = (
    # ---- XMEM geometry ------------------------------------------------------
    Param("num_banks", ("cache_unit", "banked"), "model", int, "banks", 16,
          "Number of banks XMEM is divided into.",
          CONFIRMED, f"{_SPEC}:47 (NUM_BANKS = 16); whether it is final is Q6", low=1, high=4096),
    Param("bank_rows", ("cache_unit", "banked"), "model", int, "rows", 1024,
          "Rows in one bank, which is also the page size (one row is 128 bytes).",
          CONFIRMED, f"{_SPEC}:48 (1024 rows x 1024 bits); whether it is final is Q6", low=1, high=1 << 20),
    Param("xmem_mb", ("cache_unit", "cache"), "derived", float, "MB", None,
          "Total XMEM size; sets the number of banks (cache_unit) or the cache size (cache).",
          CONFIRMED, f"{_SPEC}:47-48 (16 x 128 KB = 2 MB); set a value to override", low=0.001, high=4096),
    # ---- cache_unit behaviour -----------------------------------------------
    Param("bank_allocation", _CU, "model", str, "", "proportional",
          "How banks are shared between a kernel's tables: by table size, or a list per table in the TOML file.",
          ASSUMED, "Q4: who assigns banks to tables is not specified", choices=("proportional",),
          sweepable=False),
    Param("jump_back", _CU, "model", int, "rows", "auto",
          "Row inside a page at which the previous page's bank is given back; 'auto' derives it from the trace.",
          ASSUMED, "Q8: who sets jump_back per kernel", low=0, high=1 << 20, specials=("auto", "never")),
    Param("prefetch", _CU, "model", bool, "", True,
          "The DMA fills input pages ahead of the IPU while the IPU computes.",
          CONFIRMED, f"{_SPEC}:115 and section 8.1 (the DMA fills any free bank in the background)"),
    Param("prefetch_distance", _CU, "model", int, "banks", 1,
          "How many pages ahead of the current one the DMA may fill (the overlap of fetching and computing).",
          ASSUMED, f"{_SPEC} section 8.1 sets no limit except the table's free banks; 1 is the cautious choice",
          low=0, high=4096),
    Param("prefetch_before_start", _CU, "model", bool, "", False,
          "Input pages that fit are already in XMEM when the kernel starts.",
          ASSUMED, "Q3: CU-16 allows it; how much is realistic is open"),
    Param("dma_channels", _CU, "model", int, "transfers", 1,
          "DMA transfers that can run at the same time.",
          ASSUMED, "Q10", low=1, high=64),
    Param("fill_whole_bank", _CU, "model", bool, "", False,
          "A transfer always moves a full bank, even for a table with fewer rows.",
          ASSUMED, "Q9"),
    Param("residency", _CU, "model", str, "", "dram_roundtrip",
          "Between two layers: activations go to DRAM and come back, or stay in XMEM for the next kernel.",
          ASSUMED, "Q2 (the repo's OQ-11); dram_roundtrip is the pessimistic case",
          choices=("dram_roundtrip", "stays_in_xmem")),
    Param("weight_tables", _CU, "model", str, "", "w,filters,weights,gamma,beta,bias",
          "Names of input tables that hold weights (always fetched from DRAM), comma separated.",
          ASSUMED, "by region name; a network file can set it per layer (weights = [...])",
          sweepable=False),
    # ---- the DRAM side (cache_unit only) --------------------------------------
    Param("dram_kind", _CU, "dram", str, "", "bank_fill",
          "How DMA transfer time is given: one fill time per bank, latency plus bandwidth, or a measured table.",
          ASSUMED, "a modelling choice", choices=("bank_fill", "fixed", "table"), key="kind",
          sweepable=False),
    Param("fill_cycles_per_bank", _CU, "dram", float, "cycles", 1024,
          "Cycles the DMA needs to fill one 1024-row bank from DRAM/HBM.",
          UNKNOWN, "Q1; anchor point: DRAMsim3 HBM2 gives 2111 cycles at a 1 ns clock",
          low=0, high=1e9, dram_kinds=("bank_fill",)),
    Param("drain_cycles_per_bank", _CU, "dram", float, "cycles", None,
          "Cycles to drain one 1024-row bank to DRAM; the fill time when not set.",
          UNKNOWN, "Q1", low=0, high=1e9, dram_kinds=("bank_fill",)),
    Param("fill_ns_per_bank", _CU, "dram", float, "ns", None,
          "The bank fill time in nanoseconds, used instead of fill_cycles_per_bank when set.",
          UNKNOWN, "Q1", low=0, high=1e12, dram_kinds=("bank_fill",)),
    Param("clock_ns", _CU, "dram", float, "ns", 1.0,
          "The chip's clock period, used only to convert a fill time in nanoseconds to cycles.",
          ASSUMED, "the emulator defines no clock; OQ-16 speaks of 1 GHz", low=0.001, high=1000,
          dram_kinds=("bank_fill",)),
    Param("latency_cycles", _CU, "dram", int, "cycles", 100,
          "Fixed start-up cost of every DMA transfer.",
          UNKNOWN, "Q1", low=0, high=1e9, dram_kinds=("fixed",)),
    Param("rows_per_cycle", _CU, "dram", float, "rows/cycle", 1.0,
          "DRAM bandwidth: 128-byte rows moved per cycle.",
          UNKNOWN, "Q1", low=1e-6, high=1e6, dram_kinds=("fixed",)),
    Param("fill", _CU, "dram", dict, "cycles", None,
          "Measured fill times by transfer size (rows = cycles), for example from DRAMsim3.",
          UNKNOWN, "memsim dram-measure", dram_kinds=("table",), sweepable=False),
    Param("drain", _CU, "dram", dict, "cycles", None,
          "Measured drain times by transfer size; the fill table when not set.",
          UNKNOWN, "memsim dram-measure", dram_kinds=("table",), sweepable=False),
    # ---- toy models -----------------------------------------------------------
    Param("penalty", ("all_miss", "page_first_touch"), "model", int, "cycles", 10,
          "Stall cycles of one miss.",
          UNKNOWN, "toy model", low=0, high=1e9),
    Param("page_rows", ("page_first_touch",), "model", int, "rows", 1024,
          "Rows in one page; the first touch of each page stalls.",
          CONFIRMED, f"{_SPEC}:48", low=1, high=1 << 20),
    Param("busy_cycles", ("banked",), "model", int, "cycles", 2,
          "Cycles a bank is busy after each access.",
          UNKNOWN, "toy model", low=0, high=1e6),
    Param("mapping", ("banked",), "model", str, "", "interleave",
          "Row-to-bank mapping: neighbouring rows in different banks, or each bank a block of rows.",
          ASSUMED, "toy model; the spec's layout is 'blocked'", choices=("interleave", "blocked")),
    Param("size_rows", ("cache",), "model", int, "rows", 16384,
          "Cache capacity in rows ('inf' = unlimited).",
          ASSUMED, "toy model; 16384 rows = 2 MB", low=0, high=1 << 30, specials=("inf",)),
    Param("line_rows", ("cache",), "model", int, "rows", 1,
          "Rows fetched together on a miss.",
          ASSUMED, "toy model", low=1, high=1 << 20),
    Param("assoc", ("cache",), "model", int, "lines", 0,
          "Lines per set (0 = fully associative).",
          ASSUMED, "toy model", low=0, high=1 << 20),
    Param("miss_penalty", ("cache",), "model", int, "cycles", 10,
          "Stall cycles of one cache miss.",
          UNKNOWN, "toy model", low=0, high=1e9),
    Param("preloaded", ("cache",), "model", bool, "", False,
          "The first touch of a line is free; only a line that comes back after eviction misses.",
          ASSUMED, "toy model"),
    Param("write_allocate_free", ("cache",), "model", bool, "", False,
          "A write miss costs nothing (nothing is fetched).",
          ASSUMED, "toy model"),
)  # fmt: skip


def params_for(kind: str) -> list[Param]:
    """The parameters of model ``kind``, in registry order."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r} (kinds: {', '.join(MODEL_KINDS)})")
    return [p for p in REGISTRY if kind in p.models]


def param(kind: str, name: str) -> Param:
    """Parameter ``name`` of model ``kind``.

    Raises:
        ValueError: The model has no such parameter (the message lists them).
    """
    for p in params_for(kind):
        if p.name == name:
            return p
    known = ", ".join(p.name for p in params_for(kind)) or "none"
    raise ValueError(f"model {kind!r} has no parameter {name!r} (parameters: {known})")


def coerce(p: Param, value: Any) -> Any:  # noqa: C901, PLR0912 (one branch per type)
    """``value`` as parameter ``p`` takes it, checked against its range.

    Text from a menu or the command line is converted (``"16"`` to 16,
    ``"true"`` to True); a special value (``"auto"``) is kept as text.

    Raises:
        ValueError: Wrong type, out of range, or not one of the choices.
    """
    if isinstance(value, str) and value in p.specials:
        return value
    if p.type is dict:
        if not isinstance(value, Mapping):
            raise ValueError(f"{p.name} must be a table")
        return dict(value)
    if p.type is bool:
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        if isinstance(value, bool) or value in (0, 1):
            return bool(value)
        raise ValueError(f"{p.name} must be true or false, not {value!r}")
    if p.type is str:
        if p.name == "bank_allocation" and isinstance(value, Mapping):
            return dict(value)  # an explicit banks-per-table list
        if p.choices and value not in p.choices:
            raise ValueError(f"{p.name} must be one of {', '.join(p.choices)}, not {value!r}")
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        extra = f" or {', '.join(p.specials)}" if p.specials else ""
        raise ValueError(f"{p.name} must be a number{extra}, not {value!r}") from None
    if isinstance(value, bool):
        raise ValueError(f"{p.name} must be a number, not {value!r}")
    if p.type is int:
        if number != int(number):
            raise ValueError(f"{p.name} must be a whole number, not {value!r}")
        number = int(number)
    if (p.low is not None and number < p.low) or (p.high is not None and number > p.high):
        raise ValueError(f"{p.name} = {value!r} is outside {p.low:g}..{p.high:g} {p.unit}".rstrip())
    return number


def build_config(kind: str, values: Mapping[str, Any] | None = None) -> dict[str, Any]:  # noqa: C901 (defaults, sections, derived size)
    """The config (``{"model": {...}, "dram": {...}}``) of model ``kind`` with
    ``values`` over the registry's defaults.

    Only the ``[dram]`` keys of the chosen ``dram_kind`` are kept, so a config
    never carries a bandwidth next to a bank fill time.

    Raises:
        ValueError: An unknown parameter name or a bad value.
    """
    given = dict(values or {})
    for name in given:
        param(kind, name)  # raises on an unknown name
    chosen: dict[str, Any] = {}
    for p in params_for(kind):
        value = given.get(p.name, p.default)
        if value is not None:
            chosen[p.name] = coerce(p, value)

    model: dict[str, Any] = {"kind": kind}
    dram: dict[str, Any] = {}
    dram_kind = chosen.get("dram_kind")
    for p in params_for(kind):
        if p.name not in chosen or p.section == "derived":
            continue
        if p.section == "model":
            model[p.config_key] = chosen[p.name]
        elif p.name == "dram_kind" or dram_kind in p.dram_kinds:
            dram[p.config_key] = chosen[p.name]
    if "fill_ns_per_bank" in dram:
        dram.pop("fill_cycles_per_bank", None)  # nanoseconds were given: they win

    size_mb = chosen.get("xmem_mb")
    if size_mb is not None:
        rows = int(size_mb * MB) // ROW_BYTES
        if kind == "cache_unit":
            # Whole banks only: 0.5 MB of 1024-row banks is 4 banks.
            model["num_banks"] = max(1, rows // model["bank_rows"])
        else:
            model["size_rows"] = rows
    config = {"model": model}
    if dram:
        config["dram"] = dram
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Check a config's keys and values against the registry.

    Raises:
        ValueError: No ``[model]``, an unknown kind, a key the registry does
            not list for that model, or a value outside its range.
    """
    if "model" not in config:
        raise ValueError("the config has no [model] table")
    kind = config["model"].get("kind")
    known = {(p.section, p.config_key): p for p in params_for(kind)}  # raises on a bad kind
    for section in ("model", "dram"):
        for key, value in config.get(section, {}).items():
            if section == "model" and key == "kind":
                continue
            if (section, key) not in known:
                keys = ", ".join(k for s, k in known if s == section) or "none"
                raise ValueError(
                    f"[{section}] of a {kind} config has no key {key!r} (keys: {keys})"
                )
            coerce(known[(section, key)], value)
    unknown = set(config) - {"model", "dram"}
    if unknown:
        raise ValueError(f"unknown config table(s): {', '.join(sorted(unknown))}")


def placeholders_in_use(config: Mapping[str, Any]) -> list[Param]:
    """The ASSUMED and UNKNOWN parameters a config's result depends on: every
    such parameter of its model (and of its ``dram_kind``), whether the config
    sets it or leaves the default."""
    kind = config["model"]["kind"]
    dram_kind = config.get("dram", {}).get("kind", "bank_fill")
    found = []
    for p in params_for(kind):
        if p.status == CONFIRMED or p.section == "derived" or p.name == "dram_kind":
            continue
        if p.section == "dram":
            if dram_kind not in p.dram_kinds:
                continue
            if p.default is None and p.config_key not in config.get("dram", {}):
                continue  # an optional key that is not set plays no part
        found.append(p)
    return found
