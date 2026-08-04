"""
The Poisson core (Phase 5.A) — PDF item 12.

A **Poisson structure** is a bivector ``π`` with ``[π, π]_SN = 0``.
Canonical definitions (definition policy §1):

    {f, g}  := π(df, dg)          (the Poisson bracket),
    X_f(g)  := {f, g}             (the Hamiltonian vector field),
    ⟨β, π♯(α)⟩ := π(α, β)         (the sharp evaluation).

``[π, π]_SN = 0`` is the DECLARED axiom of the structure (the
``poisson`` declaration — opt-in engine rule, the package analogue of
the algebroid declaration system); everything ℝ/C∞-structural about
``{·,·}`` is a THEOREM of the definitions:

* antisymmetry ``{f, g} = −{g, f}`` (the alternating evaluation),
* Leibniz ``{f, g·h} = g·{f, h} + h·{f, g}``,
* ``{f, c} = 0`` for constants,
* the sharp/Hamiltonian coincidence ``⟨dg, π♯(df)⟩ = {f, g}``.

The Jacobi identity of ``{·,·}`` is NOT free: it is equivalent to
``[π,π]_SN = 0`` and arrives with the cotangent-algebroid showcase
(Phase 5.C).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.core.expr import Atom, Expr, Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.multivector import PVector, bivector
from jacopy.central.objects.musical import Sharp
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.schouten import SN


class PoissonBracketValue(Atom):
    """``{f, g}`` — a Poisson bracket value, a scalar function
    (canonical definition ``{f,g} = π(df, dg)``)."""

    __slots__ = ("_pi", "_f", "_g")

    def __init__(self, pi: PVector, f: Expr, g: Expr) -> None:
        if not isinstance(pi, PVector):
            raise TypeError("PoissonBracketValue requires a PVector")
        for s in (f, g):
            if not isinstance(s, Expr):
                raise TypeError(
                    "PoissonBracketValue requires Expr functions"
                )
        self._pi = pi
        self._f = f
        self._g = g

    @property
    def pi(self) -> PVector:
        return self._pi

    @property
    def f(self) -> Expr:
        return self._f

    @property
    def g(self) -> Expr:
        return self._g

    @property
    def degree(self) -> Degree:
        return Degree.const(0)

    @property
    def rewritable_slots(self):
        return (self._f, self._g)

    def with_slots(self, f: Expr, g: Expr) -> "PoissonBracketValue":
        return PoissonBracketValue(self._pi, f, g)

    def _key(self) -> Any:
        return (self._pi, self._f, self._g)

    def _repr_inner(self) -> str:
        return (
            "{" + self._f._repr_inner() + "," + self._g._repr_inner() + "}"
        )


class HamiltonianVF(Derivation):
    """``X_f`` — the Hamiltonian vector field of ``f`` (degree-0
    derivation, wedge degree 1); canonical action ``X_f(g) = {f,g}``."""

    __slots__ = ("_pi", "_f")

    def __init__(
        self, pi: PVector, f: Expr, *, name: Optional[str] = None
    ) -> None:
        if not isinstance(pi, PVector):
            raise TypeError("HamiltonianVF requires a PVector")
        if not isinstance(f, Expr):
            raise TypeError("HamiltonianVF requires an Expr function")
        display = (
            name if name is not None else f"X_{f._repr_inner()}"
        )
        super().__init__(display, degree=0)
        self._pi = pi
        self._f = f

    @property
    def pi(self) -> PVector:
        return self._pi

    @property
    def f(self) -> Expr:
        return self._f

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._f,)

    def with_slots(self, f: Expr) -> "HamiltonianVF":
        return HamiltonianVF(self._pi, f)

    def _key(self) -> Any:
        return (self._name, self._degree, self._pi, self._f)


class SharpVF(Derivation):
    """``π♯(α)`` — the sharp image of a 1-form AS A VECTOR FIELD
    (degree-0 derivation, wedge degree 1) — the exact
    :class:`AnchoredVF` pattern.

    The raw ``Act(Sharp, α)`` node has form-degree 0 and would be
    mistaken for a scalar by degree-based heuristics (caught in 5.A);
    the derivation atom carries the vector-field nature structurally.
    Canonical action: ``(π♯α)(g) = π(α, dg)``.
    """

    __slots__ = ("_pi", "_alpha")

    def __init__(
        self, pi: PVector, alpha: Expr, *, name: Optional[str] = None
    ) -> None:
        if not isinstance(pi, PVector):
            raise TypeError("SharpVF requires a PVector")
        if not isinstance(alpha, Expr):
            raise TypeError("SharpVF requires an Expr 1-form")
        display = (
            name
            if name is not None
            else f"{pi._repr_inner()}♯({alpha._repr_inner()})"
        )
        super().__init__(display, degree=0)
        self._pi = pi
        self._alpha = alpha

    @property
    def pi(self) -> PVector:
        return self._pi

    @property
    def alpha(self) -> Expr:
        return self._alpha

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._alpha,)

    def with_slots(self, alpha: Expr) -> "SharpVF":
        return SharpVF(self._pi, alpha)

    def _key(self) -> Any:
        return (self._name, self._degree, self._pi, self._alpha)


class PoissonStructure:
    """``(M, π)`` — a Poisson structure context.

    The bivector is the data; ``[π, π]_SN = 0`` is the structure's
    axiom, entering proofs ONLY through the ``poisson`` declaration
    (``poisson_engine(structures=(P,))``).
    """

    __slots__ = ("_pi",)

    def __init__(self, pi: PVector) -> None:
        if not isinstance(pi, PVector):
            raise TypeError("PoissonStructure requires a PVector")
        if pi.degree != Degree.const(2):
            raise ValueError(
                "a Poisson structure needs a bivector (degree 2)"
            )
        self._pi = pi

    @property
    def pi(self) -> PVector:
        return self._pi

    def bracket(self, f: Expr, g: Expr) -> PoissonBracketValue:
        """``{f, g}``."""
        return PoissonBracketValue(self._pi, f, g)

    def hamiltonian(self, f: Expr) -> HamiltonianVF:
        """``X_f``."""
        return HamiltonianVF(self._pi, f)

    def sharp(self) -> Sharp:
        """``π♯`` — the musical operator atom (form-level use)."""
        return Sharp(self._pi)

    def sharp_vf(self, alpha: Expr) -> SharpVF:
        """``π♯(α)`` as a genuine vector field (the anchor of the
        cotangent algebroid, Phase 5.C)."""
        return SharpVF(self._pi, alpha)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PoissonStructure) and self._pi == other._pi
        )

    def __hash__(self) -> int:
        return hash(("poisson-structure", self._pi))

    def __repr__(self) -> str:
        return f"PoissonStructure({self._pi!r})"


def poisson_structure(name: str = "π") -> PoissonStructure:
    """Create a Poisson structure with a fresh bivector ``π``."""
    return PoissonStructure(bivector(name))


# --------------------------------------------------------------------- #
# Definitional rules                                                     #
# --------------------------------------------------------------------- #


class PoissonBracketDefinition(Definition):
    """``{f, g} → π(df, dg)`` — the canonical definition."""

    anchor = PoissonBracketValue

    def __init__(self, structure: PoissonStructure) -> None:
        self._structure = structure
        self.name = (
            f"Poisson bracket ({structure.pi._repr_inner()}): "
            "{f,g} = π(df, dg)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, PoissonBracketValue)
            and expr.pi == self._structure.pi
        )

    def rewrite(self, expr: Expr) -> Expr:
        return MultiEval(
            self._structure.pi,
            d(expr.f),
            d(expr.g),
            alternating=True,
            slot_kind="covector",
        )


class HamiltonianActionDefinition(Definition):
    """``X_f(g) → {f, g}`` for a scalar ``g`` — the canonical action
    of the Hamiltonian vector field."""

    anchor = Act

    def __init__(
        self,
        structure: PoissonStructure,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._structure = structure
        self._registry = registry
        self.name = (
            f"Hamiltonian action ({structure.pi._repr_inner()}): "
            "X_f(g) = {f,g}"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, HamiltonianVF)
            and expr.op.pi == self._structure.pi
            and is_scalar_function(expr.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return PoissonBracketValue(
            self._structure.pi, expr.op.f, expr.arg
        )


class SharpEvaluationDefinition(Definition):
    """``⟨β, π♯(α)⟩ → π(α, β)`` — the defining evaluation of the
    sharp vector field."""

    anchor = Pairing

    def __init__(self, structure: PoissonStructure) -> None:
        self._structure = structure
        self.name = (
            f"sharp evaluation ({structure.pi._repr_inner()}): "
            "⟨β, π♯(α)⟩ = π(α, β)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Pairing)
            and isinstance(expr.X, SharpVF)
            and expr.X.pi == self._structure.pi
        )

    def rewrite(self, expr: Expr) -> Expr:
        return MultiEval(
            self._structure.pi,
            expr.X.alpha,
            expr.alpha,
            alternating=True,
            slot_kind="covector",
        )


class SharpActionDefinition(Definition):
    """``(π♯α)(g) → π(α, dg)`` for scalar ``g`` — the canonical action
    of the sharp vector field on functions."""

    anchor = Act

    def __init__(
        self,
        structure: PoissonStructure,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._structure = structure
        self._registry = registry
        self.name = (
            f"sharp action ({structure.pi._repr_inner()}): "
            "(π♯α)(g) = π(α, dg)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, SharpVF)
            and expr.op.pi == self._structure.pi
            and is_scalar_function(expr.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return MultiEval(
            self._structure.pi,
            expr.op.alpha,
            d(expr.arg),
            alternating=True,
            slot_kind="covector",
        )


class PVectorEvalLinearityDefinition(Definition):
    """Multilinearity of a p-vector evaluation in its FORM slots:
    sums split and certainly-scalar factors pull out of
    ``π(…, Σᵢ sᵢ·αᵢ, …)`` (order-robust). Definitional — a p-vector
    is a multilinear map on 1-forms."""

    name = "p-vector evaluation linearity: π(…, fα + β, …) = f·π(…,α,…) + π(…,β,…)"
    anchor = MultiEval

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, slot: Expr):
        if not (isinstance(slot, Product) and len(slot.children) >= 2):
            return None
        scalars = [
            c
            for c in slot.children
            if is_scalar_function(c, self._registry)
        ]
        rest = [
            c
            for c in slot.children
            if not is_scalar_function(c, self._registry)
        ]
        if scalars and len(rest) == 1:
            return scalars, rest[0]
        return None

    def _splittable(self, slot: Expr) -> bool:
        if isinstance(slot, Sum):
            return True
        return self._scalar_split(slot) is not None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and isinstance(expr.head, PVector)
            and any(self._splittable(a) for a in expr.args)
        )

    def rewrite(self, expr: Expr) -> Expr:
        args = expr.args
        for i, slot in enumerate(args):
            if not self._splittable(slot):
                continue

            def rebuild(sub: Expr) -> Expr:
                new = args[:i] + (sub,) + args[i + 1 :]
                return MultiEval(
                    expr.head,
                    *new,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )

            if isinstance(slot, Sum):
                return Sum(*(rebuild(c) for c in slot.children))
            scalars, core = self._scalar_split(slot)
            return Product(*scalars, rebuild(core))
        raise AssertionError("rewrite called without a splittable slot")


class PoissonSNDeclaration(Definition):
    """``[π, π]_SN → 0`` — the DECLARED Poisson axiom (the structure's
    integrability), never assumed silently."""

    anchor = BracketApply

    def __init__(self, structure: PoissonStructure) -> None:
        self._structure = structure
        self.name = (
            f"Poisson ({structure.pi._repr_inner()}): [π, π]_SN = 0"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, BracketApply)
            and expr.bracket is SN
            and expr.a == self._structure.pi
            and expr.b == self._structure.pi
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


def poisson_engine(
    *,
    registry: Optional[PropertyRegistry] = None,
    mode: str = "efficient",
    structures=(),
) -> ExpansionEngine:
    """The tangent engine extended with the Poisson layer for the
    given structures (definitional rules always; the ``[π,π]_SN = 0``
    declaration for each structure passed — the opt-in axiom)."""
    from jacopy.central.tangent.engine import tangent_engine

    engine = tangent_engine(registry=registry, mode=mode)
    engine.register(PVectorEvalLinearityDefinition(registry))
    for P in structures:
        if not isinstance(P, PoissonStructure):
            raise TypeError(
                "poisson_engine structures must be PoissonStructure "
                f"instances, got {P!r}"
            )
        engine.register(PoissonSNDeclaration(P))
        engine.register(PoissonBracketDefinition(P))
        engine.register(HamiltonianActionDefinition(P, registry))
        engine.register(SharpEvaluationDefinition(P))
        engine.register(SharpActionDefinition(P, registry))
    return engine


# --------------------------------------------------------------------- #
# Theorems (mechanical, from the definitions)                            #
# --------------------------------------------------------------------- #


def prove_poisson_antisymmetry(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``{f, g} = −{g, f}`` — from the alternating evaluation."""
    return ExpandAndSimplify().prove(
        P.bracket(f, g),
        Neg(P.bracket(g, f)),
        registry=registry,
        engine=poisson_engine(registry=registry, structures=(P,)),
    )


def prove_poisson_leibniz(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``{f, g·h} = g·{f, h} + h·{f, g}`` — from the ``d`` Leibniz
    rule and the multilinearity of the evaluation."""
    return ExpandAndSimplify().prove(
        P.bracket(f, Product(g, h)),
        Sum(
            Product(g, P.bracket(f, h)),
            Product(h, P.bracket(f, g)),
        ),
        registry=registry,
        engine=poisson_engine(registry=registry, structures=(P,)),
    )


def prove_poisson_constant(
    P: PoissonStructure,
    f: Expr,
    c: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``{f, c} = 0`` for a numeric constant ``c``."""
    return ExpandAndSimplify().prove(
        P.bracket(f, c),
        Integer(0),
        registry=registry,
        engine=poisson_engine(registry=registry, structures=(P,)),
    )


def prove_hamiltonian_is_sharp(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨dg, π♯(df)⟩ = X_f(g)`` — the sharp and Hamiltonian
    definitions meet in ``{f, g}``."""
    return ExpandAndSimplify().prove(
        Pairing(d(g), P.sharp_vf(d(f))),
        Act(P.hamiltonian(f), g),
        registry=registry,
        engine=poisson_engine(registry=registry, structures=(P,)),
    )
