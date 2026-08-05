"""
Invertible bundle endomorphisms (Phase 6.E.3) — the primitive behind
the frame-change twist ``Ψ_m = [[m, 0], [0, m̃]]`` [2409.11973 eq
(7.11)].

An endomorphism pair is identified by NAME: ``EndoVF("m", X)`` is
``m(X) ∈ TM`` and ``EndoForm("m", ω)`` is ``m̃(ω) ∈ Λᵖ``; the
``inverted`` flag marks ``m⁻¹`` / ``m̃⁻¹``. Two definitional rule
families give them semantics:

* :class:`EndoLinearityDefinition` — ``m`` and ``m̃`` are
  C∞-linear bundle maps: sums, negations, zeros and scalar factors
  pull out of the argument slot (tensoriality — unlike the Lie
  direction, a bundle map has no derivative anomaly).
* :class:`EndoInverseDefinition` — ``m⁻¹(m(x)) → x`` and
  ``m(m⁻¹(x)) → x`` (same name, opposite flags), on both sides.

Everything else about ``Ψ_m`` (the conjugated brackets and calculus,
(7.13)-(7.16)) is a theorem in :mod:`jacopy.packages.drinfeld.twist`.
The atoms are inert otherwise — an opaque automorphism has nothing
more to expand into (honest-fail policy)."""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Derivation, degree_of
from jacopy.core.expr import Atom, Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


class EndoVF(Derivation):
    """``m(X)`` — the image of a vector field under the bundle
    endomorphism named ``name`` (``m⁻¹(X)`` when ``inverted``). A
    vector field itself: degree-0 derivation, wedge degree 1."""

    __slots__ = ("_endo_name", "_arg", "_inverted")

    def __init__(
        self, name: str, X: Expr, *, inverted: bool = False
    ) -> None:
        if not isinstance(X, Expr):
            raise TypeError("EndoVF requires an Expr vector field")
        tag = f"{name}⁻¹" if inverted else name
        super().__init__(f"{tag}({X._repr_inner()})", degree=0)
        self._endo_name = name
        self._arg = X
        self._inverted = inverted

    @property
    def endo_name(self) -> str:
        return self._endo_name

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def inverted(self) -> bool:
        return self._inverted

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._arg,)

    def with_slots(self, X: Expr) -> "EndoVF":
        return EndoVF(self._endo_name, X, inverted=self._inverted)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._endo_name,
            self._arg,
            self._inverted,
        )


class EndoForm(Atom):
    """``m̃(ω)`` — the image of a form under the ``Λᵖ``-side of the
    endomorphism pair named ``name`` (``m̃⁻¹(ω)`` when ``inverted``).
    Degree follows the argument."""

    __slots__ = ("_endo_name", "_arg", "_inverted")

    def __init__(
        self, name: str, omega: Expr, *, inverted: bool = False
    ) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("EndoForm requires an Expr form")
        self._endo_name = name
        self._arg = omega
        self._inverted = inverted

    @property
    def endo_name(self) -> str:
        return self._endo_name

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def inverted(self) -> bool:
        return self._inverted

    @property
    def degree(self) -> Degree:
        deg = getattr(self._arg, "degree", None)
        if isinstance(deg, Degree):
            return deg
        try:
            return degree_of(self._arg, None)
        except ValueError:
            raise AttributeError("EndoForm degree undetermined")

    @property
    def rewritable_slots(self):
        return (self._arg,)

    def with_slots(self, omega: Expr) -> "EndoForm":
        return EndoForm(
            self._endo_name, omega, inverted=self._inverted
        )

    def _key(self) -> Any:
        return (
            self._endo_name,
            self._arg,
            self._inverted,
        )

    def _repr_inner(self) -> str:
        tag = (
            f"{self._endo_name}̃⁻¹"
            if self._inverted
            else f"{self._endo_name}̃"
        )
        return f"{tag}({self._arg._repr_inner()})"


class EndoLinearityDefinition(Definition):
    """C∞-linearity of a bundle endomorphism in its argument slot:
    ``m(X + Y) → m(X) + m(Y)``, ``m(−X) → −m(X)``, ``m(0) → 0``,
    ``m(f·X) → f·m(X)`` — tensoriality of a bundle map (both the
    vector and the form side)."""

    name = "endomorphism linearity: m(fX + Y) = f·m(X) + m(Y)"
    anchor = (EndoVF, EndoForm)

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def _splits(self, arg: Expr) -> bool:
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )

        if isinstance(arg, (Sum, Neg)) or arg == Integer(0):
            return True
        if isinstance(arg, Product) and len(arg.children) >= 2:
            return any(
                is_scalar_function(c, self._registry)
                for c in arg.children
            )
        return False

    def matches(self, expr: Expr) -> bool:
        return isinstance(
            expr, (EndoVF, EndoForm)
        ) and self._splits(expr.arg)

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )

        arg = expr.arg
        if arg == Integer(0):
            return Integer(0)
        if isinstance(arg, Sum):
            return Sum(*(expr.with_slots(c) for c in arg.children))
        if isinstance(arg, Neg):
            return Neg(expr.with_slots(arg.arg))
        scalars = [
            c
            for c in arg.children
            if is_scalar_function(c, self._registry)
        ]
        rest = [
            c
            for c in arg.children
            if not is_scalar_function(c, self._registry)
        ]
        core = rest[0] if len(rest) == 1 else Product(*rest)
        return Product(*scalars, expr.with_slots(core))


class EndoInverseDefinition(Definition):
    """``m⁻¹(m(x)) → x`` and ``m(m⁻¹(x)) → x`` — the defining laws
    of an invertible pair (matched by name, opposite flags), on both
    the vector and the form side."""

    name = "endomorphism inverse: m⁻¹(m(x)) = x = m(m⁻¹(x))"
    anchor = (EndoVF, EndoForm)

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, (EndoVF, EndoForm)):
            return False
        inner = expr.arg
        return (
            type(inner) is type(expr)
            and inner.endo_name == expr.endo_name
            and inner.inverted != expr.inverted
        )

    def rewrite(self, expr: Expr) -> Expr:
        return expr.arg.arg
