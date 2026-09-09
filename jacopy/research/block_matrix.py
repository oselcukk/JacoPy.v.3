"""
Block-operator matrices (research interface, Phase 8 brick 1): write a
bundle automorphism the way the papers do,

    Ψ_Π = ( 1  Π )          Π = ( Π₃ ,  Π₆ + Π₃ ⊛ Π₃ )
          ( 0  1 )

and let the *action*, the *product* and the *inverse* be derived
instead of typed. Entries are :class:`Operator` objects — maps between
slots (``θ♯ : Λ¹ → TM``, ``ι_·B : TM → Λᵖ``, …) with an additive
structure (sums, scalar multiples, negation, composition); ``0`` and
``1`` are accepted as shorthand for the zero and identity entries.

Inverses are computed exactly for UNIPOTENT block matrices
(identity diagonal, strictly block-triangular off-diagonal): with
``N = Ψ − 1`` nilpotent, ``Ψ⁻¹ = 1 − N + N² − … ± N^{n−1}``. This covers
every twist of [2409.11973 §7] and the E₆ rotation (4.10)–(4.12); a
non-unipotent matrix must be inverted by the user (``inverse=``).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

from jacopy.core.expr import Expr, Integer, Neg, Product
from jacopy.research.sections import (
    GeneralizedSection,
    SectionType,
    sum_components,
)


# ------------------------------------------------------------------ #
# Operators (matrix entries)                                          #
# ------------------------------------------------------------------ #


class Operator:
    """A map between slots, ``Expr → Expr``, with an additive
    structure. Subclasses implement :meth:`apply`."""

    name: str = "?"

    def apply(self, x: Expr) -> Expr:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, x: Expr) -> Expr:
        if x == Integer(0):
            return Integer(0)
        return self.apply(x)

    # ---- algebra -------------------------------------------------- #

    @property
    def is_zero(self) -> bool:
        return False

    @property
    def is_identity(self) -> bool:
        return False

    def same(self, other: "Operator") -> bool:
        """Structural sameness good enough for cancellation:
        identical objects, or both identities, or both zeros."""
        return (
            self is other
            or (self.is_identity and other.is_identity)
            or (self.is_zero and other.is_zero)
            or (
                isinstance(self, NegOperator)
                and isinstance(other, NegOperator)
                and self.inner.same(other.inner)
            )
        )

    def __add__(self, other) -> "Operator":
        other = as_operator(other)
        if self.is_zero:
            return other
        if other.is_zero:
            return self
        # a + (−a) = 0 (the cancellation the unipotent inverse needs)
        if isinstance(other, NegOperator) and other.inner.same(self):
            return Zero()
        if isinstance(self, NegOperator) and self.inner.same(other):
            return Zero()
        return SumOperator(self, other)

    __radd__ = __add__

    def __neg__(self) -> "Operator":
        if self.is_zero:
            return self
        if isinstance(self, NegOperator):
            return self.inner
        return NegOperator(self)

    def __sub__(self, other) -> "Operator":
        return self + (-as_operator(other))

    def __rmul__(self, scalar: Expr) -> "Operator":
        if self.is_zero:
            return self
        return ScaledOperator(scalar, self)

    def __matmul__(self, other) -> "Operator":
        """Composition ``self ∘ other``."""
        other = as_operator(other)
        if self.is_zero or other.is_zero:
            return Zero()
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        return ComposedOperator(self, other)

    def __repr__(self) -> str:
        return self.name


class Zero(Operator):
    name = "0"

    def apply(self, x: Expr) -> Expr:
        return Integer(0)

    @property
    def is_zero(self) -> bool:
        return True


class Identity(Operator):
    name = "1"

    def apply(self, x: Expr) -> Expr:
        return x

    @property
    def is_identity(self) -> bool:
        return True


class Map(Operator):
    """An operator given by a Python callable, e.g.
    ``Map(N.sharp_vf, "θ♯")``."""

    def __init__(self, fn: Callable[[Expr], Expr], name: str) -> None:
        if not callable(fn):
            raise TypeError("Map needs a callable")
        self._fn = fn
        self.name = name

    def apply(self, x: Expr) -> Expr:
        return self._fn(x)


class SumOperator(Operator):
    def __init__(self, a: Operator, b: Operator) -> None:
        self.a, self.b = a, b
        self.name = f"({a.name} + {b.name})"

    def apply(self, x: Expr) -> Expr:
        return sum_components((self.a(x), self.b(x)))


class NegOperator(Operator):
    def __init__(self, inner: Operator) -> None:
        self.inner = inner
        self.name = f"−{inner.name}"

    def apply(self, x: Expr) -> Expr:
        out = self.inner(x)
        if out == Integer(0):
            return out
        if isinstance(out, Neg):
            return out.arg
        return Neg(out)


class ScaledOperator(Operator):
    def __init__(self, scalar: Expr, inner: Operator) -> None:
        self.scalar, self.inner = scalar, inner
        self.name = f"{scalar._repr_inner()}·{inner.name}"

    def apply(self, x: Expr) -> Expr:
        out = self.inner(x)
        if out == Integer(0):
            return out
        return Product(self.scalar, out)


class ComposedOperator(Operator):
    def __init__(self, outer: Operator, inner: Operator) -> None:
        self.outer, self.inner = outer, inner
        self.name = f"{outer.name}∘{inner.name}"

    def apply(self, x: Expr) -> Expr:
        return self.outer(self.inner(x))


EntryLike = Union[Operator, int, Callable[[Expr], Expr]]


def as_operator(x: EntryLike) -> Operator:
    """``0``/``1`` → zero/identity; an :class:`Operator` passes
    through; a bare callable becomes an anonymous :class:`Map`."""
    if isinstance(x, Operator):
        return x
    if isinstance(x, bool):
        raise TypeError("use 0/1, not booleans, for matrix entries")
    if isinstance(x, int):
        if x == 0:
            return Zero()
        if x == 1:
            return Identity()
        if x == -1:
            return NegOperator(Identity())
        raise ValueError("integer entries must be 0, 1 or -1")
    if isinstance(x, Integer):
        return as_operator(int(str(x)))
    if callable(x):
        return Map(x, getattr(x, "__name__", "map"))
    raise TypeError(f"cannot use {type(x).__name__} as a matrix entry")


# ------------------------------------------------------------------ #
# Block matrices                                                      #
# ------------------------------------------------------------------ #


class BlockMatrix:
    """A square block matrix acting on sections of ``type_``:
    ``(Ψe)ᵢ = Σⱼ Ψᵢⱼ eⱼ``. Rows are given top to bottom, entries as
    :class:`Operator` objects or ``0``/``1``."""

    def __init__(
        self,
        type_: SectionType,
        rows: Sequence[Sequence[EntryLike]],
        *,
        name: str = "Ψ",
    ) -> None:
        n = len(type_)
        if len(rows) != n or any(len(r) != n for r in rows):
            raise ValueError(
                f"{type_} has {n} slots: need a {n}×{n} block matrix"
            )
        self._type = type_
        self._rows: List[List[Operator]] = [
            [as_operator(x) for x in r] for r in rows
        ]
        self.name = name

    @property
    def type(self) -> SectionType:
        return self._type

    @property
    def rows(self) -> List[List[Operator]]:
        return [list(r) for r in self._rows]

    def __getitem__(self, ij) -> Operator:
        i, j = ij
        return self._rows[i][j]

    @property
    def size(self) -> int:
        return len(self._rows)

    # ---- action --------------------------------------------------- #

    def apply(self, e: GeneralizedSection) -> GeneralizedSection:
        if not isinstance(e, GeneralizedSection):
            raise TypeError("a BlockMatrix acts on GeneralizedSection")
        if e.type != self._type:
            raise TypeError(
                f"matrix on {self._type} applied to a section of "
                f"{e.type}"
            )
        comps = []
        for row in self._rows:
            comps.append(
                sum_components(
                    op(c) for op, c in zip(row, e) if not op.is_zero
                )
            )
        return GeneralizedSection(self._type, *comps)

    __call__ = apply

    # ---- algebra -------------------------------------------------- #

    @classmethod
    def identity(cls, type_: SectionType) -> "BlockMatrix":
        n = len(type_)
        return cls(
            type_,
            [[1 if i == j else 0 for j in range(n)] for i in range(n)],
            name="1",
        )

    def __matmul__(self, other: "BlockMatrix") -> "BlockMatrix":
        if not isinstance(other, BlockMatrix):
            raise TypeError("BlockMatrix @ BlockMatrix only")
        if other._type != self._type:
            raise TypeError("block matrices of different types")
        n = self.size
        rows = []
        for i in range(n):
            row = []
            for j in range(n):
                acc: Operator = Zero()
                for k in range(n):
                    acc = acc + (self._rows[i][k] @ other._rows[k][j])
                row.append(acc)
            rows.append(row)
        return BlockMatrix(
            self._type, rows, name=f"{self.name}·{other.name}"
        )

    def __add__(self, other: "BlockMatrix") -> "BlockMatrix":
        if other._type != self._type:
            raise TypeError("block matrices of different types")
        return BlockMatrix(
            self._type,
            [
                [a + b for a, b in zip(ra, rb)]
                for ra, rb in zip(self._rows, other._rows)
            ],
            name=f"({self.name} + {other.name})",
        )

    def __neg__(self) -> "BlockMatrix":
        return BlockMatrix(
            self._type,
            [[-a for a in r] for r in self._rows],
            name=f"−{self.name}",
        )

    def __sub__(self, other: "BlockMatrix") -> "BlockMatrix":
        return self + (-other)

    # ---- unipotent inverse ---------------------------------------- #

    def _strictly_triangular(self) -> Optional[str]:
        n = self.size
        upper = all(
            self._rows[i][j].is_zero
            for i in range(n)
            for j in range(n)
            if j < i
        )
        lower = all(
            self._rows[i][j].is_zero
            for i in range(n)
            for j in range(n)
            if j > i
        )
        if upper:
            return "upper"
        if lower:
            return "lower"
        return None

    @property
    def is_unipotent(self) -> bool:
        """Identity diagonal and strictly block-triangular off the
        diagonal — the shape whose inverse is a finite Neumann
        series."""
        n = self.size
        return (
            all(self._rows[i][i].is_identity for i in range(n))
            and self._strictly_triangular() is not None
        )

    def inverse(self) -> "BlockMatrix":
        """``Ψ⁻¹ = 1 − N + N² − …`` for unipotent ``Ψ = 1 + N``
        (``N`` nilpotent of order ≤ size); exact, no assumption on the
        entries beyond linearity."""
        if not self.is_unipotent:
            raise ValueError(
                f"{self.name} is not unipotent (identity diagonal + "
                "strictly triangular): supply its inverse explicitly"
            )
        one = BlockMatrix.identity(self._type)
        N = self - one
        result = one
        power = one
        sign = -1
        for _ in range(1, self.size):
            power = power @ N
            result = result + (power if sign > 0 else -power)
            sign = -sign
        result.name = f"{self.name}⁻¹"
        return result

    # ---- display -------------------------------------------------- #

    def __repr__(self) -> str:
        cells = [[op.name for op in r] for r in self._rows]
        width = max(len(c) for r in cells for c in r)
        lines = [
            "( " + "  ".join(c.ljust(width) for c in r) + " )"
            for r in cells
        ]
        return f"{self.name} =\n" + "\n".join(lines)
