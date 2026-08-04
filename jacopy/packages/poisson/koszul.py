"""
The Koszul bracket on 1-forms (Phase 5.B) — PDF item 12.

Canonical definition (definition policy §1):

    [α, β]_π := L_{π♯α} β − L_{π♯β} α − d(π(α, β)),

with ``L``/``d`` the ordinary TM Cartan operators (Phase 2) and
``π♯α`` the sharp vector field of 5.A. The bracket is a 1-form.

Phase 5.B ships the node, the definitional expansion, the sharp's
``C^∞``-linearity, and the first mechanical theorems:

* antisymmetry ``[α, β]_π = −[β, α]_π``,
* the exact-forms identity ``[df, dg]_π = d{f, g}`` (evaluated
  against a generic vector field) — the bracket restricted to exact
  forms IS the Poisson bracket's differential.

The algebroid-axiom proofs (right-Leibniz, anchor morphism, Jacobi
under ``[π,π]=0``) are the Phase 5.C showcase.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Atom, Expr, Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.packages.poisson.core import (
    PoissonStructure,
    SharpVF,
)


class KoszulBracket(Atom):
    """``[α, β]_π`` — the Koszul bracket of two 1-forms; itself a
    1-form (degree 1, slot protocol)."""

    __slots__ = ("_pi", "_alpha", "_beta")

    def __init__(self, pi, alpha: Expr, beta: Expr) -> None:
        for s in (alpha, beta):
            if not isinstance(s, Expr):
                raise TypeError("KoszulBracket requires Expr 1-forms")
        self._pi = pi
        self._alpha = alpha
        self._beta = beta

    @property
    def pi(self):
        return self._pi

    @property
    def alpha(self) -> Expr:
        return self._alpha

    @property
    def beta(self) -> Expr:
        return self._beta

    @property
    def degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._alpha, self._beta)

    def with_slots(self, alpha: Expr, beta: Expr) -> "KoszulBracket":
        return KoszulBracket(self._pi, alpha, beta)

    def _key(self) -> Any:
        return (self._pi, self._alpha, self._beta)

    def _repr_inner(self) -> str:
        return (
            f"[{self._alpha._repr_inner()},{self._beta._repr_inner()}]_"
            f"{self._pi._repr_inner()}"
        )


def koszul_bracket(
    P: PoissonStructure, alpha: Expr, beta: Expr
) -> KoszulBracket:
    """``[α, β]_π`` for the given Poisson structure — 1-FORMS ONLY.

    The Koszul bracket does extend to all form degrees
    (Koszul-Brylinski, ``[Ω^p, Ω^q]_π → Ω^{p+q−1}`` via the generator
    ``∂_π = [d, ι_π]``), and its p-form version on a Nambu-Poisson
    structure is a Leibniz algebroid bracket — both are Phase 5.E
    material. The degree-1 formula implemented here is WRONG for
    higher degrees, so anything determinably not a 1-form is refused
    rather than silently mis-expanded.
    """
    from jacopy.algebra.derivation import degree_of

    if not isinstance(P, PoissonStructure):
        raise TypeError("koszul_bracket expects a PoissonStructure")
    # ℝ-bilinearity: a zero argument gives the zero bracket (the zero
    # form inhabits every degree, so the degree guard must not reject
    # it — edge-case audit fix).
    if alpha == Integer(0) or beta == Integer(0):
        return Integer(0)
    for leg in (alpha, beta):
        try:
            deg = degree_of(leg, None)
        except ValueError:
            continue  # undeterminable: caller-asserted, allow
        if deg != Degree.const(1):
            raise ValueError(
                "the Phase 5.B Koszul bracket is the DEGREE-1 "
                f"(cotangent algebroid) bracket; got a degree-{deg} "
                "argument. The higher-degree Koszul-Brylinski bracket "
                "and the Nambu-Poisson p-form bracket arrive in "
                "Phase 5.E."
            )
    return KoszulBracket(P.pi, alpha, beta)


class KoszulBracketDefinition(Definition):
    """``[α, β]_π → L_{π♯α} β − L_{π♯β} α − d(π(α, β))`` — the
    canonical definition."""

    anchor = KoszulBracket

    def __init__(self, structure: PoissonStructure) -> None:
        self._structure = structure
        self.name = (
            f"Koszul bracket ({structure.pi._repr_inner()}): "
            "[α,β]_π = L_{π♯α}β − L_{π♯β}α − d(π(α,β))"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.algebra.derivation import degree_of

        if not (
            isinstance(expr, KoszulBracket)
            and expr.pi == self._structure.pi
        ):
            return False
        # The degree-1 formula only: a node carrying determinably
        # higher-degree slots stays inert (honest) instead of
        # mis-expanding.
        for leg in (expr.alpha, expr.beta):
            try:
                if degree_of(leg, None) != Degree.const(1):
                    return False
            except ValueError:
                continue
        return True

    def rewrite(self, expr: Expr) -> Expr:
        pi = self._structure.pi
        alpha, beta = expr.alpha, expr.beta
        return Sum(
            Act(CARTAN_TM.lie(SharpVF(pi, alpha)), beta),
            Neg(Act(CARTAN_TM.lie(SharpVF(pi, beta)), alpha)),
            Neg(
                d(
                    MultiEval(
                        pi,
                        alpha,
                        beta,
                        alternating=True,
                        slot_kind="covector",
                    )
                )
            ),
        )


class SharpVFLinearityDefinition(Definition):
    """``C^∞``-linearity of the sharp vector field (definitional —
    the sharp is tensorial): ``π♯(fα + β) = f·π♯α + π♯β``."""

    name = "sharp linearity: π♯(fα + β) = f·π♯(α) + π♯(β)"
    anchor = SharpVF

    def __init__(
        self,
        structure: PoissonStructure,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._structure = structure
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            scalars = [
                c
                for c in expr.children
                if is_scalar_function(c, self._registry)
            ]
            rest = [
                c
                for c in expr.children
                if not is_scalar_function(c, self._registry)
            ]
            if scalars and len(rest) == 1:
                return scalars, rest[0]
        return None

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, SharpVF)
            and expr.pi == self._structure.pi
        ):
            return False
        a = expr.alpha
        if isinstance(a, (Sum, Neg)) or a == Integer(0):
            return True
        return self._scalar_split(a) is not None

    def rewrite(self, expr: Expr) -> Expr:
        a = expr.alpha
        if a == Integer(0):
            return Integer(0)
        if isinstance(a, Sum):
            return Sum(*(expr.with_slots(c) for c in a.children))
        if isinstance(a, Neg):
            return Neg(expr.with_slots(a.arg))
        scalars, core = self._scalar_split(a)
        return Product(*scalars, expr.with_slots(core))


# --------------------------------------------------------------------- #
# Theorems                                                               #
# --------------------------------------------------------------------- #


def _engine(structures, registry):
    from jacopy.packages.poisson.core import poisson_engine

    engine = poisson_engine(registry=registry, structures=structures)
    for P in structures:
        engine.register(KoszulBracketDefinition(P))
        engine.register(SharpVFLinearityDefinition(P, registry))
    return engine


def prove_koszul_antisymmetry(
    P: PoissonStructure,
    alpha: Expr,
    beta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``[α, β]_π = −[β, α]_π`` — from the definitional expansion and
    the alternating evaluation of ``π``."""
    return ExpandAndSimplify().prove(
        koszul_bracket(P, alpha, beta),
        Neg(koszul_bracket(P, beta, alpha)),
        registry=registry,
        engine=_engine((P,), registry),
    )


def prove_koszul_on_exact_forms(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨[df, dg]_π, Y⟩ = ⟨d{f, g}, Y⟩`` — the Koszul bracket on
    exact forms is the Poisson bracket's differential (evaluated
    against the generic vector field ``Y``)."""
    return ExpandAndSimplify().prove(
        Pairing(koszul_bracket(P, d(f), d(g)), Y),
        Pairing(d(P.bracket(f, g)), Y),
        registry=registry,
        engine=_engine((P,), registry),
    )
