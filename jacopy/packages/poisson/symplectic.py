"""
The symplectic special case (Phase 5.E.1) — Midterm Q2 (a)-(b).

A **symplectic structure** is a 2-form ``ω`` that is CLOSED and
nondegenerate. Canonical definitions (definition policy §1):

    ι_{X_f} ω := df            (the Hamiltonian vector field of f),

with ``dω = 0`` the structure's DECLARED axiom (the ``closed``
declaration — opt-in engine rule, same pattern as the ``poisson``
declaration). Nondegeneracy enters only through the *existence* of
``X_f`` for every ``f`` (the defining relation posits the solution);
no rewrite rule assumes it.

The bridge to the Poisson world, ``π = ω⁻¹``, is a DECLARED
compatibility between two independently-given structures:

    π♯(ι_X ω) = X              (sharp ∘ flat = id, the primitive),
    π(df, dg) = ω(X_g, X_f)    (the exact-generator view),

registered only via ``symplectic_engine(..., poisson_bridge=P)``.

Sign convention note: this package's interior inserts into the FIRST
slot, ``(ι_X ω)(Y) = ω(X, Y)``, and 5.A fixes ``{f,g} := π(df, dg)``
with ``X_f(g) = {f,g}``. Under these conventions the Q2(b) chain
reads ``{f, g} = ω(X_g, X_f) = X_f(g) = ι_{X_f} dg
= ι_{X_f} ι_{X_g} ω``; a statement of the form ``{f,g} = ω(X_f,X_g)``
belongs to the opposite interior convention and is mechanically
REFUSED here (tested).

Mechanical theorems:

* Q2(a) ``(L_{X_f} ω)(Y, Z) = 0`` — the Hamiltonian flow preserves
  ``ω``; needs the DECLARED closedness (cited evaluated instance,
  two-leg tactic); honest fail without.
* Q2(b) the bracket chain above (bridge + defining relation).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.objects import forms
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import d
from jacopy.packages.poisson.core import PoissonStructure, SharpVF


class SymplecticHamiltonianVF(Derivation):
    """``X_f`` — the Hamiltonian vector field of ``f`` DEFINED BY the
    symplectic form (``ι_{X_f}ω = df``); a degree-0 derivation of
    wedge degree 1. Distinct from the Poisson-side
    :class:`~jacopy.packages.poisson.core.HamiltonianVF` (whose
    canonical definition is ``X_f(g) = {f,g}``): the bridge THEOREMS
    identify them, definitions stay separate (policy §1)."""

    __slots__ = ("_omega", "_f")

    def __init__(
        self, omega: Expr, f: Expr, *, name: Optional[str] = None
    ) -> None:
        if not isinstance(omega, Expr) or not isinstance(f, Expr):
            raise TypeError(
                "SymplecticHamiltonianVF requires Expr omega and f"
            )
        display = (
            name if name is not None else f"X^ω_{f._repr_inner()}"
        )
        super().__init__(display, degree=0)
        self._omega = omega
        self._f = f

    @property
    def omega(self) -> Expr:
        return self._omega

    @property
    def f(self) -> Expr:
        return self._f

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._f,)

    def with_slots(self, f: Expr) -> "SymplecticHamiltonianVF":
        return SymplecticHamiltonianVF(self._omega, f)

    def _key(self) -> Any:
        return (self._name, self._degree, self._omega, self._f)


class SymplecticStructure:
    """``(M, ω)`` — a symplectic structure context.

    The 2-form is the data; ``dω = 0`` is the structure's axiom,
    entering proofs ONLY through the ``closed`` declaration
    (``symplectic_engine(declare_closed=True)``)."""

    __slots__ = ("_omega",)

    def __init__(self, omega: Expr) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("SymplecticStructure requires an Expr")
        deg = degree_of(omega, None)
        if deg != Degree.const(2):
            raise ValueError(
                "a symplectic structure needs a 2-form "
                f"(got degree {deg})"
            )
        self._omega = omega

    @property
    def omega(self) -> Expr:
        return self._omega

    def hamiltonian(self, f: Expr) -> SymplecticHamiltonianVF:
        """``X_f`` with ``ι_{X_f}ω = df``."""
        return SymplecticHamiltonianVF(self._omega, f)

    def flat(self, X: Expr) -> Expr:
        """``ω♭(X) := ι_X ω`` (a 1-form)."""
        return Act(Interior(X), self._omega)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SymplecticStructure)
            and self._omega == other._omega
        )

    def __hash__(self) -> int:
        return hash(("symplectic-structure", self._omega))

    def __repr__(self) -> str:
        return f"SymplecticStructure({self._omega!r})"


def symplectic_structure(name: str = "ω") -> SymplecticStructure:
    """Create a symplectic structure with a fresh 2-form."""
    (omega,) = forms(name, degree=2)
    return SymplecticStructure(omega)


# --------------------------------------------------------------------- #
# Engine rules                                                           #
# --------------------------------------------------------------------- #


class SymplecticHamiltonianDefinition(Definition):
    """``ι_{X_f} ω → df`` — the canonical definition of the
    symplectic Hamiltonian vector field, in both firing shapes:

    * the operator node ``Act(Interior(X_f), ω)``,
    * the evaluated 2-form ``ω(X_f, Y) → ⟨df, Y⟩`` (and
      ``ω(Y, X_f) → −⟨df, Y⟩`` by antisymmetry); with two Hamiltonian
      slots the FIRST fires (deterministic)."""

    anchor = (Act, MultiEval)

    def __init__(self, structure: SymplecticStructure) -> None:
        self._S = structure
        self.name = (
            "symplectic Hamiltonian "
            f"({structure.omega._repr_inner()}): ι_{{X_f}}ω = df"
        )

    def _own_hamiltonian(self, expr: Expr) -> bool:
        return (
            isinstance(expr, SymplecticHamiltonianVF)
            and expr.omega == self._S.omega
        )

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Act):
            return (
                isinstance(expr.op, Interior)
                and self._own_hamiltonian(expr.op.vector)
                and expr.arg == self._S.omega
            )
        if isinstance(expr, MultiEval):
            return expr.head == self._S.omega and any(
                self._own_hamiltonian(a) for a in expr.args
            )
        return False

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Act):
            return d(expr.op.vector.f)
        args = expr.args
        i = next(
            idx
            for idx, a in enumerate(args)
            if self._own_hamiltonian(a)
        )
        rest = args[:i] + args[i + 1:]
        inner: Expr = (
            Pairing(d(args[i].f), rest[0])
            if len(rest) == 1
            else MultiEval(
                d(args[i].f),
                *rest,
                alternating=True,
                slot_kind="vector",
            )
        )
        return Neg(inner) if i % 2 else inner


class SymplecticClosedDeclaration(Definition):
    """``dω → 0`` — the DECLARED closedness of the symplectic form
    (node form and evaluated head form)."""

    anchor = (Act, MultiEval)

    def __init__(self, structure: SymplecticStructure) -> None:
        self._S = structure
        self.name = (
            "declared closed symplectic "
            f"({structure.omega._repr_inner()}): dω = 0"
        )

    def _is_d_omega(self, expr: Expr) -> bool:
        from jacopy.central.calculus.bracket_calculus import (
            ExteriorDerivative,
        )

        return (
            isinstance(expr, Act)
            and isinstance(expr.op, ExteriorDerivative)
            and expr.arg == self._S.omega
        )

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Act):
            return self._is_d_omega(expr)
        if isinstance(expr, MultiEval):
            return self._is_d_omega(expr.head)
        return False

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


class SharpFlatIdentityDeclaration(Definition):
    """``π♯(ι_X ω) → X`` — the primitive face of the DECLARED
    compatibility ``π = ω⁻¹`` (sharp ∘ flat = id)."""

    anchor = SharpVF

    def __init__(
        self, P: PoissonStructure, S: SymplecticStructure
    ) -> None:
        self._P = P
        self._S = S
        self.name = (
            "declared π = ω⁻¹ (primitive): π♯(ι_X ω) = X"
        )

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, SharpVF) and expr.pi == self._P.pi
        ):
            return False
        a = expr.alpha
        return (
            isinstance(a, Act)
            and isinstance(a.op, Interior)
            and a.arg == self._S.omega
        )

    def rewrite(self, expr: Expr) -> Expr:
        return expr.alpha.op.vector


class PoissonOmegaBridgeDeclaration(Definition):
    """``π(df, dg) → ω(X_g, X_f)`` — the exact-generator view of the
    DECLARED compatibility ``π = ω⁻¹`` (the sign is forced by the
    first-slot interior convention: ``ω(X_g, X_f) = (ι_{X_g}ω)(X_f)
    = ⟨dg, X_f⟩ = X_f(g) = {f, g}``)."""

    anchor = MultiEval

    def __init__(
        self, P: PoissonStructure, S: SymplecticStructure
    ) -> None:
        self._P = P
        self._S = S
        self.name = (
            "declared π = ω⁻¹ (exact view): π(df, dg) = ω(X_g, X_f)"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.packages.poisson.showcase import _is_exact

        return (
            isinstance(expr, MultiEval)
            and expr.head == self._P.pi
            and expr.arity == 2
            and all(_is_exact(a) for a in expr.args)
        )

    def rewrite(self, expr: Expr) -> Expr:
        f = expr.args[0].arg
        g = expr.args[1].arg
        return MultiEval(
            self._S.omega,
            self._S.hamiltonian(g),
            self._S.hamiltonian(f),
            alternating=True,
            slot_kind="vector",
        )


# --------------------------------------------------------------------- #
# Engine                                                                 #
# --------------------------------------------------------------------- #


def symplectic_engine(
    S: SymplecticStructure,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_closed: bool = True,
    poisson_bridge: Optional[PoissonStructure] = None,
) -> ExpansionEngine:
    """The tangent engine + the symplectic rules.

    ``declare_closed=False`` withholds the ``dω = 0`` declaration
    (honesty runs). ``poisson_bridge=P`` additionally registers the
    Poisson rules of ``P`` and the DECLARED ``π = ω⁻¹`` bridge."""
    if poisson_bridge is not None:
        from jacopy.packages.poisson.core import poisson_engine

        engine = poisson_engine(
            registry=registry, structures=(poisson_bridge,)
        )
        engine.register(
            SharpFlatIdentityDeclaration(poisson_bridge, S)
        )
        engine.register(
            PoissonOmegaBridgeDeclaration(poisson_bridge, S)
        )
    else:
        from jacopy.central.tangent.engine import tangent_engine

        engine = tangent_engine(registry=registry)
    engine.register(SymplecticHamiltonianDefinition(S))
    if declare_closed:
        engine.register(SymplecticClosedDeclaration(S))
    return engine


# --------------------------------------------------------------------- #
# Closedness instances (two-leg tactic) + theorems                       #
# --------------------------------------------------------------------- #


def _normalized_by(engine, seed: Expr, registry) -> Expr:
    from jacopy.algorithms.product_rule import product_rule
    from jacopy.algorithms.simplify import simplify

    cur = seed
    for _ in range(10):
        expanded, _steps = engine.expand(cur)
        after = product_rule(expanded, registry)
        reduced = simplify(after, registry)
        if reduced == cur:
            break
        cur = reduced
    return cur


def _cite_closedness_instances(
    engine, S: SymplecticStructure, triples, registry
) -> None:
    """Register ``±`` evaluated-closedness instance theorems for each
    vector-field triple ``(U, V, W)``: the Palais expansion of
    ``dω(U, V, W)`` vanishes. Two legs — the expansion with the
    declaration WITHHELD, the declared kill — combined explicitly
    (the poisson_jacobi tactic pattern)."""
    from jacopy.proof.theorems import Theorem, TheoremBook, cite

    book = TheoremBook()
    names = []
    for U, V, W in triples:
        eval_node = MultiEval(
            Act(
                __import__(
                    "jacopy.central.tangent.exterior",
                    fromlist=["CARTAN_TM"],
                ).CARTAN_TM.d,
                S.omega,
            ),
            U,
            V,
            W,
            alternating=True,
            slot_kind="vector",
        )
        bare = symplectic_engine(
            S, registry=registry, declare_closed=False
        )
        pal = _normalized_by(bare, eval_node, registry)
        leg1 = ExpandAndSimplify().prove(
            eval_node, pal, registry=registry, engine=bare
        )
        step1 = ProofStep(
            eval_node,
            pal,
            rule="Palais expansion of dω(U,V,W)",
            justification="declaration withheld",
        )
        for s in leg1:
            step1.add_child(s)
        declared = symplectic_engine(S, registry=registry)
        leg2 = ExpandAndSimplify().prove(
            eval_node,
            Integer(0),
            registry=registry,
            engine=declared,
        )
        step2 = ProofStep(
            eval_node,
            Integer(0),
            rule="declared closed symplectic structure",
            justification="dω = 0, so its evaluation vanishes",
        )
        for s in leg2:
            step2.add_child(s)
        step3 = ProofStep(
            pal,
            Integer(0),
            rule="combine the two legs",
            justification="the Palais form equals dω(U,V,W) = 0",
        )
        chain = ProofChain([step1, step2, step3])
        tag = "_".join(x._repr_inner() for x in (U, V, W))
        for sign, sfx in (((lambda x: x), "pos"), (Neg, "neg")):
            name = f"closedness_{S.omega._repr_inner()}_{tag}_{sfx}"
            if name in book:
                continue
            lhs = _normalized_by(engine, sign(pal), registry)
            if lhs == Integer(0):
                continue  # degenerate instance; nothing to cite
            book.add(
                Theorem(
                    name=name,
                    statement="dω(U,V,W)-Palais = 0 (closedness)",
                    lhs=lhs,
                    rhs=Integer(0),
                    proof=chain,
                    generality="instance",
                    from_axioms=(
                        f"closed symplectic "
                        f"({S.omega._repr_inner()}): dω = 0",
                    ),
                )
            )
            names.append(name)
    if names:
        cite(engine, book, *names)


def prove_hamiltonian_flow_invariance(
    S: SymplecticStructure,
    f: Expr,
    Y: Expr,
    Z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Q2(a): ``(L_{X_f} ω)(Y, Z) = 0`` — the Hamiltonian flow
    preserves the symplectic form. Mechanism: the intrinsic ``L``
    unroll + the defining relation ``ι_{X_f}ω = df`` + the cited
    evaluated-closedness instance ``dω(X_f, Y, Z) = 0``. Honest fail
    without the closedness declaration (tested)."""
    from jacopy.central.tangent.exterior import CARTAN_TM

    engine = symplectic_engine(S, registry=registry)
    Xf = S.hamiltonian(f)
    _cite_closedness_instances(engine, S, ((Xf, Y, Z),), registry)
    lhs = MultiEval(
        Act(CARTAN_TM.lie(Xf), S.omega),
        Y,
        Z,
        alternating=True,
        slot_kind="vector",
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


def prove_bracket_via_omega(
    P: PoissonStructure,
    S: SymplecticStructure,
    f: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Q2(b), leg 1: ``{f, g} = ω(X_g, X_f)`` (first-slot interior
    convention — see the module docstring; the opposite slot order is
    mechanically refused)."""
    rhs = MultiEval(
        S.omega,
        S.hamiltonian(g),
        S.hamiltonian(f),
        alternating=True,
        slot_kind="vector",
    )
    return ExpandAndSimplify().prove(
        P.bracket(f, g),
        rhs,
        registry=registry,
        engine=symplectic_engine(
            S, registry=registry, poisson_bridge=P
        ),
    )


def prove_bracket_via_iota(
    P: PoissonStructure,
    S: SymplecticStructure,
    f: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Q2(b), leg 2: ``{f, g} = ι_{X_f} dg``."""
    rhs = Act(Interior(S.hamiltonian(f)), d(g))
    return ExpandAndSimplify().prove(
        P.bracket(f, g),
        rhs,
        registry=registry,
        engine=symplectic_engine(
            S, registry=registry, poisson_bridge=P
        ),
    )


def prove_bracket_via_double_iota(
    P: PoissonStructure,
    S: SymplecticStructure,
    f: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Q2(b), leg 3: ``{f, g} = ι_{X_f} ι_{X_g} ω``."""
    rhs = Act(
        Interior(S.hamiltonian(f)),
        Act(Interior(S.hamiltonian(g)), S.omega),
    )
    return ExpandAndSimplify().prove(
        P.bracket(f, g),
        rhs,
        registry=registry,
        engine=symplectic_engine(
            S, registry=registry, poisson_bridge=P
        ),
    )


def prove_sharp_flat_identity(
    P: PoissonStructure,
    S: SymplecticStructure,
    X: Expr,
    beta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨β, π♯(ι_X ω)⟩ = ⟨β, X⟩`` — the primitive bridge face on a
    generic 1-form ``β``."""
    return ExpandAndSimplify().prove(
        Pairing(beta, SharpVF(P.pi, S.flat(X))),
        Pairing(beta, X),
        registry=registry,
        engine=symplectic_engine(
            S, registry=registry, poisson_bridge=P
        ),
    )
