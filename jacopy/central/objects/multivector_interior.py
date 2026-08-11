"""
Interior product by a MULTIVECTOR (Phase 6.H.2) — the primitive the
exceptional twist needs [2409.11973 / E6 paper eqs (4.13)-(4.14)]:

    ι_P : Λ^q T*M → Λ^{q−p} T*M,   P ∈ Λ^p TM,

defined on decomposable multivectors by

    ι_{X₁∧…∧X_p} := ι_{X_p} ∘ ⋯ ∘ ι_{X₁}

and extended C∞-linearly. PARTIAL contraction — the result is a
form, not a vector — which is what distinguishes it from the Nambu
sharp (full contraction). On an ATOMIC multivector the operator is
inert (honest-fail policy: nothing to expand into); its semantics
come from two definitional rules:

* :class:`MultivectorInteriorLinearityDefinition` — tensoriality in
  the multivector slot (sums, negations, zeros, scalar factors);
* :class:`MultivectorInteriorDecomposableDefinition` — the
  decomposable case unfolds into the iterated ordinary interior.

This node closes the 6.H "kapsam notu" gap: with it the
``Π₃ ⊛ Π₃`` bundle map of the exceptional Ψ_Π twist is buildable
(``(Π₃⊛Π₃)(ω₅) = ½ Π₃(ι_{Π₃} ω₅)``, see
:mod:`jacopy.packages.drinfeld.examples`)."""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.proof.expansion import Definition


def _multivector_degree(P: Expr) -> Degree:
    """The multivector degree of ``P`` (wedge-lift aware). In a
    ``Product``, factors with no determinable degree are scalar
    coefficients (degree 0) — ``f·Π₃`` grades like ``Π₃``."""
    lift = getattr(P, "wedge_degree", None)
    if isinstance(lift, Degree):
        return lift
    deg = getattr(P, "degree", None)
    if isinstance(deg, Degree):
        return deg
    if isinstance(P, (Wedge, Product)):
        total = Degree.const(0)
        for c in P.children:
            try:
                total = total + _multivector_degree(c)
            except ValueError:
                pass  # scalar coefficient
        return total
    if isinstance(P, (Sum, Neg)):
        arg = P.children[0] if isinstance(P, Sum) else P.arg
        return _multivector_degree(arg)
    return degree_of(P, None)


class MultivectorInterior(Derivation):
    """``ι_P`` — interior product by the multivector ``P``; a graded
    derivation-slot operator of degree ``−p``."""

    __slots__ = ("_multivector",)

    def __init__(
        self, P: Expr, *, name: Optional[str] = None
    ) -> None:
        if not isinstance(P, Expr):
            raise TypeError(
                "MultivectorInterior requires an Expr multivector"
            )
        display = (
            name
            if name is not None
            else f"ι_{P._repr_inner()}"
        )
        k = _multivector_degree(P)
        super().__init__(display, degree=Degree.const(0) - k)
        self._multivector = P

    @property
    def multivector(self) -> Expr:
        return self._multivector

    @property
    def rewritable_slots(self):
        return (self._multivector,)

    def with_slots(self, P: Expr) -> "MultivectorInterior":
        return MultivectorInterior(P)

    def _key(self) -> Any:
        return (self._name, self._degree, self._multivector)


class MultivectorInteriorLinearityDefinition(Definition):
    """Tensoriality of ``ι_P`` in the MULTIVECTOR slot:
    ``ι_{f·P + Q} = f·ι_P + ι_Q`` (Act level)."""

    name = "multivector interior linearity: ι_{fP+Q} = f·ι_P + ι_Q"
    anchor = Act

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def _splits(self, P: Expr) -> bool:
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )

        if isinstance(P, (Sum, Neg)) or P == Integer(0):
            return True
        if isinstance(P, Product) and len(P.children) >= 2:
            return any(
                is_scalar_function(c, self._registry)
                for c in P.children
            )
        return False

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, MultivectorInterior)
            and self._splits(expr.op.multivector)
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )

        P = expr.op.multivector
        x = expr.arg
        if P == Integer(0):
            return Integer(0)
        if isinstance(P, Sum):
            return Sum(
                *(
                    Act(MultivectorInterior(c), x)
                    for c in P.children
                )
            )
        if isinstance(P, Neg):
            return Neg(Act(MultivectorInterior(P.arg), x))
        scalars = [
            c
            for c in P.children
            if is_scalar_function(c, self._registry)
        ]
        rest = [
            c
            for c in P.children
            if not is_scalar_function(c, self._registry)
        ]
        core = rest[0] if len(rest) == 1 else Product(*rest)
        return Product(
            *scalars, Act(MultivectorInterior(core), x)
        )


class MultivectorInteriorDecomposableDefinition(Definition):
    """The DECOMPOSABLE case [eq (4.13)]:
    ``ι_{X₁∧…∧X_p}(ω) → ι_{X_p}(⋯ ι_{X₁}(ω) ⋯)`` — nested ordinary
    interiors (rightmost factor contracts first). Fires when every
    wedge factor is a wedge-degree-1 vector."""

    name = (
        "multivector interior on decomposables: "
        "ι_{X₁∧…∧X_p} = ι_{X_p} ∘ ⋯ ∘ ι_{X₁}"
    )
    anchor = Act

    def _vectors(self, P: Expr):
        if not (
            isinstance(P, Wedge) and len(P.children) >= 2
        ):
            return None
        for c in P.children:
            lift = getattr(c, "wedge_degree", None)
            if not (
                isinstance(lift, Degree)
                and lift == Degree.const(1)
            ):
                return None
        return P.children

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, MultivectorInterior)
            and self._vectors(expr.op.multivector) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.objects.interior import Interior

        vectors = self._vectors(expr.op.multivector)
        out = expr.arg
        for X in vectors:  # X₁ first (innermost)
            out = Act(Interior(X), out)
        return out
