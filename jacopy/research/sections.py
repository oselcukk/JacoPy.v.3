"""
Generalized sections as FIRST-CLASS objects (research interface,
Phase 8 brick 4): a section of ``E = A ⊕ Z`` — a vector field together
with one or more differential forms — used to be a bare Python tuple
``(U, ω)``; here it is a typed object whose *type* records the slots.

    T = SectionType.generalized_tangent()          # TM ⊕ Λ¹T*M
    e = T.section(U, om)                            # U ⊕ ω
    T3 = SectionType.exceptional()                  # TM ⊕ Λ²T*M ⊕ Λ⁵T*M

Sections add, negate and scale slot-wise (the module ``C∞``-structure
of ``Γ(E)``); brackets and block matrices act on them. The type is what
lets the axiom suite know *how to evaluate* each component (a vector on
a probe function, a ``k``-form on ``k`` vector slots).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from jacopy.core.expr import Expr, Integer, Neg, Product, Sum


@dataclass(frozen=True)
class Slot:
    """One summand of ``E``: ``kind`` is ``"vector"`` (a section of
    ``TM``), ``"form"`` (a ``degree``-form) or ``"function"``."""

    kind: str
    degree: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("vector", "form", "function"):
            raise ValueError(f"unknown slot kind {self.kind!r}")
        if self.kind != "form" and self.degree != 0:
            raise ValueError("only form slots carry a degree")

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.kind == "vector":
            return "TM"
        if self.kind == "function":
            return "C∞(M)"
        return f"Λ{_sup(self.degree)}T*M"


def _sup(n: int) -> str:
    return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(c)] for c in str(n))


class SectionType:
    """The slot list of ``E`` (``TM ⊕ Λ² ⊕ Λ⁵`` and the like)."""

    def __init__(self, *slots: Slot) -> None:
        if not slots:
            raise ValueError("a SectionType needs at least one slot")
        self._slots: Tuple[Slot, ...] = tuple(slots)

    @property
    def slots(self) -> Tuple[Slot, ...]:
        return self._slots

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self):
        return iter(self._slots)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SectionType) and self._slots == other._slots

    def __hash__(self) -> int:
        return hash(("section-type", self._slots))

    def __repr__(self) -> str:
        return " ⊕ ".join(s.label for s in self._slots)

    # ---- common types --------------------------------------------- #

    @classmethod
    def generalized_tangent(cls, p: int = 1) -> "SectionType":
        """``TM ⊕ ΛᵖT*M`` (``p = 1``: the standard generalized
        tangent bundle)."""
        return cls(Slot("vector"), Slot("form", p))

    @classmethod
    def exceptional(cls) -> "SectionType":
        """``TM ⊕ Λ²T*M ⊕ Λ⁵T*M`` — the E₆ exceptional bundle."""
        return cls(Slot("vector"), Slot("form", 2), Slot("form", 5))

    # ---- constructors --------------------------------------------- #

    def section(self, *components: Expr) -> "GeneralizedSection":
        return GeneralizedSection(self, *components)

    def zero(self) -> "GeneralizedSection":
        return GeneralizedSection(
            self, *(Integer(0) for _ in self._slots)
        )


class GeneralizedSection:
    """``U ⊕ ω₂ ⊕ ω₅`` — a section of ``E`` with one ``Expr`` per slot."""

    __slots__ = ("_type", "_components")

    def __init__(self, type_: SectionType, *components: Expr) -> None:
        if not isinstance(type_, SectionType):
            raise TypeError("GeneralizedSection needs a SectionType")
        if len(components) != len(type_):
            raise ValueError(
                f"{type_} has {len(type_)} slots, got "
                f"{len(components)} components"
            )
        for c in components:
            if not isinstance(c, Expr):
                raise TypeError(
                    "section components must be Expr; got "
                    f"{type(c).__name__}"
                )
        self._type = type_
        self._components = tuple(components)

    @property
    def type(self) -> SectionType:
        return self._type

    @property
    def components(self) -> Tuple[Expr, ...]:
        return self._components

    def __iter__(self):
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def __getitem__(self, i: int) -> Expr:
        return self._components[i]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GeneralizedSection)
            and self._type == other._type
            and self._components == other._components
        )

    def __hash__(self) -> int:
        return hash(("section", self._type, self._components))

    def __repr__(self) -> str:
        return " ⊕ ".join(c._repr_inner() for c in self._components)

    # ---- module structure ----------------------------------------- #

    def _same_type(self, other: "GeneralizedSection") -> None:
        if not isinstance(other, GeneralizedSection):
            raise TypeError("expected a GeneralizedSection")
        if other._type != self._type:
            raise TypeError(
                f"section types differ: {self._type} vs {other._type}"
            )

    def __add__(self, other: "GeneralizedSection") -> "GeneralizedSection":
        self._same_type(other)
        return GeneralizedSection(
            self._type,
            *(_add(a, b) for a, b in zip(self, other)),
        )

    def __sub__(self, other: "GeneralizedSection") -> "GeneralizedSection":
        return self + (-other)

    def __neg__(self) -> "GeneralizedSection":
        return GeneralizedSection(
            self._type, *(_neg(c) for c in self._components)
        )

    def scale(self, f: Expr) -> "GeneralizedSection":
        """``f · e`` slot-wise (``f`` a scalar function)."""
        return GeneralizedSection(
            self._type, *(_scale(f, c) for c in self._components)
        )

    def __rmul__(self, f: Expr) -> "GeneralizedSection":
        return self.scale(f)

    def map(self, fn) -> "GeneralizedSection":
        """Apply ``fn(component, slot)`` slot-wise."""
        return GeneralizedSection(
            self._type,
            *(fn(c, s) for c, s in zip(self._components, self._type)),
        )


# ---- slot arithmetic that keeps zeros out of the way ----------------- #


def _is_zero(x: Expr) -> bool:
    return x == Integer(0)


def _add(a: Expr, b: Expr) -> Expr:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    return Sum(a, b)


def _neg(a: Expr) -> Expr:
    if _is_zero(a):
        return a
    if isinstance(a, Neg):
        return a.arg
    return Neg(a)


def _scale(f: Expr, a: Expr) -> Expr:
    if _is_zero(a):
        return a
    return Product(f, a)


def sum_components(terms: Iterable[Expr]) -> Expr:
    """``Σ terms`` with zeros dropped (``0`` when nothing remains)."""
    kept = [t for t in terms if not _is_zero(t)]
    if not kept:
        return Integer(0)
    if len(kept) == 1:
        return kept[0]
    return Sum(*kept)
