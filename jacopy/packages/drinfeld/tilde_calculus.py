"""
The Nambu tilde calculus IS a calculus — App D instantiation of the
drinfeld paper [arXiv:2312.06584], eqs (D.5)-(D.13).

The 6.D/6.E layers built the tilde elements ``ℒ̃`` (:func:`~jacopy.
packages.drinfeld.double.lie_tilde_nambu`), ``ι̃_ω V = ι_V ω`` and
``d̃ = −Πd`` from a ``(p+1)``-vector Π. Here they are shown to
satisfy the CALCULUS conditions (C.1)-(C.3)/(4.34) in their tilde
instantiation:

    (D.5)  ℒ̃_ω ℒ̃_η W − ℒ̃_η ℒ̃_ω W − ℒ̃_{[ω,η]_Kos} W = 0,
    (D.6)  ℒ̃_ω 𝒦̃_η W − 𝒦̃_η ℒ̃_ω W − 𝒦̃_{[ω,η]_Kos} W = 0,
    (D.7)  ℒ̃_ω 𝒦̃_η W + 𝒦̃_η 𝒦̃_ω W − 𝒦̃_{[ω,η]_Kos} W = 0,

with ``𝒦̃_η = −ℒ̃_η + d̃ ι̃_η``. THE assumption is the fundamental
identity in its bracket-morphism face (the paper: "Π is a morphism
of brackets"), consumed as the opt-in declaration
:class:`NambuMorphismDeclaration`:

    [Πω, Πη]_Lie → Π [ω, η]_Kos

— exactly ``R′ = 0`` (the 6.E R-twist). Without the declaration the
conditions honestly FAIL with the R-twist as the residual; the Lie
side enters through cited vector-field Jacobi instances (the repair
loop). The simple Jacobi compatibility conditions (D.9), (D.12),
(D.13) — the ones the paper closes by ``d² = 0`` / Cartan relations
alone — are declaration-free.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import d
from jacopy.packages.drinfeld.double import lie_tilde_nambu
from jacopy.packages.poisson.nambu import (
    NambuSharpVF,
    nambu_koszul_bracket,
)


def iota_tilde(U: Expr, eta: Expr) -> Expr:
    """``ι̃_η U := ι_U η`` — the tilde interior [eq (6.7)]."""
    return Act(Interior(U), eta)


def d_tilde(N, x: Expr) -> Expr:
    """``d̃ x := −Π d x`` — the tilde differential [eq (6.7)]."""
    return Neg(N.sharp_vf(d(x)))


def kappa_tilde_nambu(N, eta: Expr, W: Expr) -> Expr:
    """``𝒦̃_η W := −ℒ̃_η W + d̃ ι̃_η W`` [the (4.14) pattern on the
    tilde side]."""
    return Sum(
        Neg(lie_tilde_nambu(N, eta, W)),
        d_tilde(N, iota_tilde(W, eta)),
    )


class NambuMorphismDeclaration(Definition):
    """DECLARED fundamental identity, bracket-morphism face
    [drinfeld eq (6.9) / the vanishing of the 6.E R-twist]:

        [Πω, Πη]_Lie → Π [ω, η]_Kos.

    Opt-in — Π must be Nambu-Poisson for this to hold; without the
    declaration the calculus conditions fail with the R-twist as the
    honest residual. Terminating direction: a bracket of two sharps
    becomes a single sharp."""

    anchor = LieBracketVF

    def __init__(self, structure) -> None:
        self._N = structure
        self.name = (
            "declared FI (bracket morphism): "
            "[Πω, Πη] = Π[ω,η]_Kos"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        X, Y = expr.X, expr.Y
        return (
            isinstance(X, NambuSharpVF)
            and isinstance(Y, NambuSharpVF)
            and X.pi == self._N.pi
            and Y.pi == self._N.pi
        )

    def rewrite(self, expr: Expr) -> Expr:
        return self._N.sharp_vf(
            nambu_koszul_bracket(self._N, expr.X.omega, expr.Y.omega)
        )


class LieIotaCommutatorDefinition(Definition):
    """THEOREM-classified Cartan relation ``ι_{[X,Y]} = ℒ_X ι_Y −
    ι_Y ℒ_X`` as an operator-level rewrite (proved in Phase 2 — the
    ``lie_iota`` relation; the always-on engines apply it at eval
    sites). Terminating: removes a bracket from an interior
    subscript; magic then converts the ``ℒ``'s to ``ι``/``d``."""

    anchor = Act

    def __init__(self) -> None:
        self.name = (
            "theorem lie-iota: ι_{[X,Y]} x = ℒ_X(ι_Y x) − ι_Y(ℒ_X x)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, Interior)
            and isinstance(expr.op.vector, LieBracketVF)
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.tangent.exterior import CARTAN_TM

        br = expr.op.vector
        x = expr.arg
        return Sum(
            Act(
                CARTAN_TM.lie(br.X),
                Act(Interior(br.Y), x),
            ),
            Neg(
                Act(
                    Interior(br.Y),
                    Act(CARTAN_TM.lie(br.X), x),
                )
            ),
        )

    def theorem_proof_builder(self):
        from jacopy.proof.step import ProofStep

        def _builder(matched: Expr) -> ProofChain:
            return ProofChain(
                [
                    ProofStep(
                        matched,
                        self.rewrite(matched),
                        rule="lie-iota Cartan relation (proven, Phase 2)",
                        justification=(
                            "the eval-level statement closes "
                            "mechanically in the Cartan suite"
                        ),
                    )
                ]
            )

        return _builder


class InteriorAnticommuteDefinition(Definition):
    """``ι_X ι_Y x → −ι_Y ι_X x`` when ``(X, Y)`` violates the
    canonical order — alternation of forms, applied as a sorting
    rewrite so composed interiors reach one normal form (strictly
    order-reducing, hence terminating)."""

    name = "interior anticommutation: ι_X ι_Y = −ι_Y ι_X (canonical order)"
    anchor = Act

    def _parts(self, expr: Expr):
        if not (
            isinstance(expr, Act)
            and isinstance(expr.op, Interior)
        ):
            return None
        inner = expr.arg
        sign = False
        if isinstance(inner, Neg):
            sign = True
            inner = inner.arg
        if not (
            isinstance(inner, Act)
            and isinstance(inner.op, Interior)
        ):
            return None
        X = expr.op.vector
        Y = inner.op.vector
        # Canonical order: SHARP-subscripted interiors outermost
        # (so ι_W ι_{Πω} → −ι_{Πω} ι_W and the FI pairing-swap rule
        # can see the sharp directly under d), then repr order.
        def key(v):
            return (
                0 if isinstance(v, NambuSharpVF) else 1,
                v._repr_inner(),
            )

        if key(X) <= key(Y):
            return None
        return sign, X, Y, inner.arg

    def matches(self, expr: Expr) -> bool:
        return self._parts(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        sign, X, Y, x = self._parts(expr)
        core = Act(Interior(Y), Act(Interior(X), x))
        return core if sign else Neg(core)


class FISharpPairingSwapDefinition(Definition):
    """THEOREM-tagged CONSEQUENCE of the declared FI (both
    orientations + Lie antisymmetry):

        Π( d( ι_{Πa} b ) ) → −Π( d( ι_{Πb} a ) )

    applied as a canonical-ordering rewrite (fires when ``repr(a) >
    repr(b)``; strictly order-reducing, hence terminating).

    Derivation: FI gives [Πa,Πb] = Π[a,b]_Kos for BOTH orderings;
    with [Πb,Πa] = −[Πa,Πb] and the Koszul symmetric part
    [a,b] + [b,a] = d g_Z(a,b) this forces Π(d g_Z(a,b)) = 0, i.e.
    Π(dι_{Πa}b) = −Π(dι_{Πb}a). THIS is the g_Z-family residual of
    the (D.5)-(D.7) diagnosis — not an extra assumption, a derivable
    face of the FI itself (found 2026-08-05 via the twisted-FI
    consistency residual)."""

    anchor = NambuSharpVF

    def __init__(self, structure) -> None:
        self._N = structure
        self.name = (
            "theorem (FI consequence): Π(dι_{Πa}b) = −Π(dι_{Πb}a)"
        )

    def _parts(self, expr: Expr):
        from jacopy.central.calculus.bracket_calculus import (
            ExteriorDerivative,
        )

        if not (
            isinstance(expr, NambuSharpVF)
            and expr.pi == self._N.pi
        ):
            return None
        slot = expr.omega
        if not (
            isinstance(slot, Act)
            and isinstance(slot.op, ExteriorDerivative)
        ):
            return None
        inner = slot.arg
        if not (
            isinstance(inner, Act)
            and isinstance(inner.op, Interior)
            and isinstance(inner.op.vector, NambuSharpVF)
            and inner.op.vector.pi == self._N.pi
        ):
            return None
        a = inner.op.vector.omega
        b = inner.arg
        if a._repr_inner() <= b._repr_inner():
            return None
        return a, b

    def matches(self, expr: Expr) -> bool:
        return self._parts(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        a, b = self._parts(expr)
        return Neg(
            self._N.sharp_vf(
                d(Act(Interior(self._N.sharp_vf(b)), a))
            )
        )

    def theorem_proof_builder(self):
        from jacopy.proof.step import ProofStep

        def _builder(matched: Expr) -> ProofChain:
            return ProofChain(
                [
                    ProofStep(
                        matched,
                        self.rewrite(matched),
                        rule=(
                            "FI (both orientations) + Lie "
                            "antisymmetry + Koszul symmetric part"
                        ),
                        justification=(
                            "Π(d g_Z(a,b)) = 0 follows from the "
                            "declared FI applied to [Πa,Πb] and "
                            "[Πb,Πa]"
                        ),
                    )
                ]
            )

        return _builder


class NambuSharpPairingEvalDefinition(Definition):
    """Evaluation view of the sharp: ``⟨β, Πω⟩ → Π(α₁,…,α_p, β)``
    when ``ω = α₁∧…∧α_p`` decomposes into 1-forms (a single 1-form
    at p = 1) — definitional, the Pairing face of the sharp's
    contraction (the action face is
    :class:`~jacopy.packages.poisson.nambu.NambuSharpActionDefinition`).
    At p = 1 it lets the canonical alternating sort merge ``⟨η,Πω⟩``
    with ``−⟨ω,Πη⟩`` (the g_Z degeneration); at p ≥ 2 on a
    decomposable coframe wedge it produces the multivector's frame
    components (the (4.13) ↔ ⊛ bridge, 2026-09-10). Opaque p-forms
    stay inert (honest)."""

    anchor = None  # Pairing (set in __init__)

    def __init__(self, structure, registry=None) -> None:
        from jacopy.core.pairing import Pairing

        self._N = structure
        self._registry = registry
        self.anchor = Pairing
        self.name = (
            "Nambu sharp pairing evaluation: ⟨β, Πω⟩ = Π(α₁,…,α_p, β) "
            "on decomposable ω"
        )

    def _legs(self, omega: Expr):
        from jacopy.algebra.derivation import degree_of
        from jacopy.core.symbolic_degree import Degree
        from jacopy.core.wedge import Wedge

        p = self._N.p
        if p == 1:
            return [omega]
        if not (
            isinstance(omega, Wedge) and len(omega.children) == p
        ):
            return None
        for leg in omega.children:
            try:
                if degree_of(leg, self._registry) != Degree.const(1):
                    return None
            except ValueError:
                return None
        return list(omega.children)

    def matches(self, expr: Expr) -> bool:
        from jacopy.core.pairing import Pairing

        return (
            isinstance(expr, Pairing)
            and isinstance(expr.X, NambuSharpVF)
            and expr.X.pi == self._N.pi
            and self._legs(expr.X.omega) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.multi_eval import MultiEval

        return MultiEval(
            self._N.pi,
            *self._legs(expr.X.omega),
            expr.alpha,
            alternating=True,
            slot_kind="covector",
        )


def _tilde_engine(N, registry, *, declare_fi: bool):
    from jacopy.packages.drinfeld.twist import _twist_engine

    eng = _twist_engine(N, registry)
    eng.register(LieIotaCommutatorDefinition())
    eng.register(InteriorAnticommuteDefinition())
    eng.register(NambuSharpPairingEvalDefinition(N, registry))
    if declare_fi:
        eng.register(NambuMorphismDeclaration(N))
        eng.register(FISharpPairingSwapDefinition(N))
    return eng


def _cite_fi_instances(engine, N, omega, eta, W, h, registry):
    """Register DECLARED-FI instances in their composed-action face
    (the 6.B two-normal-form technique): the defect

        Π(ω)(Π(η)(s)) − Π(η)(Π(ω)(s)) − Π([ω,η]_Kos)(s)

    for the probe scalars ``s ∈ {h, W(h)}``, each also W-lifted and
    in both signs, normalized BY THE CITING ENGINE and registered as
    Sum-subset citations with rhs 0. Provenance: axiom instance (the
    FI is a declaration — the instance chain records the congruence,
    it does not pretend a proof)."""
    from jacopy.proof.step import ProofStep
    from jacopy.proof.theorems import (
        Theorem,
        TheoremBook,
        TheoremDefinition,
    )
    from jacopy.packages.poisson.tilde import _normalized_by

    book = TheoremBook()
    names = []
    Pw = N.sharp_vf(omega)
    Pe = N.sharp_vf(eta)
    kos = N.sharp_vf(nambu_koszul_bracket(N, omega, eta))
    for s_expr, stag in ((h, "h"), (Act(W, h), "Wh")):
        S = Sum(
            Act(Pw, Act(Pe, s_expr)),
            Neg(Act(Pe, Act(Pw, s_expr))),
            Neg(Act(kos, s_expr)),
        )
        for lift, ltag in ((None, "flat"), (W, "Wlift")):
            for sgn, gtag in (((lambda x: x), "p"), (Neg, "n")):
                seed = S if lift is None else Act(lift, S)
                seed = sgn(seed)
                lifted = _normalized_by(engine, seed, registry)
                if lifted == Integer(0):
                    continue
                nm = f"fi_inst_{stag}_{ltag}_{gtag}"
                book.add(
                    Theorem(
                        name=nm,
                        statement=(
                            "declared FI instance "
                            "(composed-action face)"
                        ),
                        lhs=lifted,
                        rhs=Integer(0),
                        proof=ProofChain(
                            [
                                ProofStep(
                                    seed,
                                    Integer(0),
                                    rule=(
                                        "declared FI: [Πω,Πη] = "
                                        "Π[ω,η]_Kos (instance + "
                                        "congruence)"
                                    ),
                                    justification=(
                                        "axiom instance of the "
                                        "opt-in fundamental identity"
                                    ),
                                    provenance_tag="axiom",
                                )
                            ]
                        ),
                        generality="instance",
                    )
                )
                names.append(nm)
    for nm in names:
        engine.register(TheoremDefinition(book.get(nm)))


def _prove_condition(
    N,
    node: Expr,
    f: Expr,
    *,
    registry,
    declare_fi: bool,
    max_steps: int,
    instance_args=None,
) -> Tuple[ProofChain, List[Theorem]]:
    from jacopy.central.tangent.cartan import (
        prove_with_bracket_identities,
    )

    engine = _tilde_engine(N, registry, declare_fi=declare_fi)
    if declare_fi and instance_args is not None:
        omega, eta, W, h = instance_args
        _cite_fi_instances(engine, N, omega, eta, W, h, registry)
    try:
        return prove_with_bracket_identities(
            node,
            Integer(0),
            f,
            registry=registry,
            engine=engine,
            max_steps=max_steps,
        )
    except ProofFailure:
        # 6.J fallback (the p = 1 face): stall-time difference-test
        # citation with the deep-instance families. Honest: raises
        # again if the residual survives.
        if not (declare_fi and instance_args is not None and N.p == 1):
            raise
        omega, eta, W, h = instance_args
        deep = _p1_deep_instances(
            engine, N, omega, eta, W, h, registry
        )
        chain = _stall_difference_prove(
            engine, node, deep, registry
        )
        return chain, []


def prove_tilde_calculus_condition_one(
    N,
    omega: Expr,
    eta: Expr,
    W: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_fi: bool = True,
    max_steps: int = 30000,
) -> Tuple[ProofChain, List[Theorem]]:
    """(D.5): ``ℒ̃_ω ℒ̃_η W − ℒ̃_η ℒ̃_ω W = ℒ̃_{[ω,η]_Kos} W`` —
    the tilde Lie derivatives represent the Koszul bracket. Needs
    the declared FI (``declare_fi=False`` fails honestly with the
    R-twist residual); the Lie side enters as cited VF-Jacobi
    instances. Acting on the probe ``h``."""
    node = Act(
        Sum(
            lie_tilde_nambu(N, omega, lie_tilde_nambu(N, eta, W)),
            Neg(
                lie_tilde_nambu(
                    N, eta, lie_tilde_nambu(N, omega, W)
                )
            ),
            Neg(
                lie_tilde_nambu(
                    N, nambu_koszul_bracket(N, omega, eta), W
                )
            ),
        ),
        h,
    )
    return _prove_condition(
        N,
        node,
        f,
        registry=registry,
        declare_fi=declare_fi,
        max_steps=max_steps,
        instance_args=(omega, eta, W, h),
    )


def prove_tilde_calculus_condition_two(
    N,
    omega: Expr,
    eta: Expr,
    W: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_fi: bool = True,
    max_steps: int = 30000,
) -> Tuple[ProofChain, List[Theorem]]:
    """(D.6): ``ℒ̃_ω 𝒦̃_η W − 𝒦̃_η ℒ̃_ω W = 𝒦̃_{[ω,η]_Kos} W``.
    Same assumption structure as (D.5)."""
    node = Act(
        Sum(
            lie_tilde_nambu(
                N, omega, kappa_tilde_nambu(N, eta, W)
            ),
            Neg(
                kappa_tilde_nambu(
                    N, eta, lie_tilde_nambu(N, omega, W)
                )
            ),
            Neg(
                kappa_tilde_nambu(
                    N, nambu_koszul_bracket(N, omega, eta), W
                )
            ),
        ),
        h,
    )
    return _prove_condition(
        N,
        node,
        f,
        registry=registry,
        declare_fi=declare_fi,
        max_steps=max_steps,
        instance_args=(omega, eta, W, h),
    )


def prove_tilde_calculus_condition_three(
    N,
    omega: Expr,
    eta: Expr,
    W: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_fi: bool = True,
    max_steps: int = 30000,
) -> Tuple[ProofChain, List[Theorem]]:
    """(D.7): ``ℒ̃_ω 𝒦̃_η W + 𝒦̃_η 𝒦̃_ω W = 𝒦̃_{[ω,η]_Kos} W``.
    Same assumption structure as (D.5)."""
    node = Act(
        Sum(
            lie_tilde_nambu(
                N, omega, kappa_tilde_nambu(N, eta, W)
            ),
            kappa_tilde_nambu(
                N, eta, kappa_tilde_nambu(N, omega, W)
            ),
            Neg(
                kappa_tilde_nambu(
                    N, nambu_koszul_bracket(N, omega, eta), W
                )
            ),
        ),
        h,
    )
    return _prove_condition(
        N,
        node,
        f,
        registry=registry,
        declare_fi=declare_fi,
        max_steps=max_steps,
        instance_args=(omega, eta, W, h),
    )


# ------------------------------------------------------------------- #
# Jacobi compatibility — the declaration-free ones (D.9, D.11-D.13)    #
# ------------------------------------------------------------------- #


def prove_jacobi_compat_d9(
    N,
    U: Expr,
    eta: Expr,
    mu: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """(D.9) [the second Jacobi compatibility (4.37) in tilde
    instantiation]:

        ℒ_{d̃ι̃_η U} μ + [dι_U η, μ]_Kos + 𝒦_{Πμ}(dι_U η) = 0

    — follows from ``d² = 0`` (``𝒦_V = −ι_V d`` on the exact form
    ``dι_Uη``). DECLARATION-FREE. Evaluated on the given slots."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.tangent.exterior import CARTAN_TM
    from jacopy.packages.drinfeld.calculus_conditions import kappa
    from jacopy.packages.drinfeld.double import _ev

    theta = d(iota_tilde(U, eta))
    node = _ev(
        Sum(
            Act(CARTAN_TM.lie(d_tilde(N, iota_tilde(U, eta))), mu),
            nambu_koszul_bracket(N, theta, mu),
            kappa(N.sharp_vf(mu), theta),
        ),
        slots,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_jacobi_compat_d12(
    N,
    omega: Expr,
    V: Expr,
    W: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """(D.12) [the second dual condition in (4.40)]:

        ℒ̃_{dι_V ω} W + [d̃ι̃_ω V, W]_Lie = 0

    — ``ℒ̃`` on the exact form ``dι_Vω`` degenerates to the bracket
    with its sharp (``d² = 0`` kills the ``ι dω`` leg), which is
    minus the second term. DECLARATION-FREE; probe ``h``."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.tangent.lie_bracket import lie_bracket

    theta = d(iota_tilde(V, omega))
    node = Act(
        Sum(
            lie_tilde_nambu(N, theta, W),
            lie_bracket(d_tilde(N, iota_tilde(V, omega)), W),
        ),
        h,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_jacobi_compat_d13(
    N,
    mu: Expr,
    U: Expr,
    V: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """(D.13) [the third dual condition in (4.40), ``g_A = 0``]:

        d̃ι̃_{ℒ_U μ} V − d̃ι̃_{dι_V μ} U + d̃ι̃_μ [U,V]_Lie = 0

    — the combination inside ``−Πd(…)`` collapses to ``d(ι_Uι_Vμ)``
    by the lie-iota relation, then dies by ``d² = 0``.
    DECLARATION-FREE; probe ``h``."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.tangent.exterior import CARTAN_TM
    from jacopy.central.tangent.lie_bracket import lie_bracket

    node = Act(
        Sum(
            d_tilde(N, iota_tilde(V, Act(CARTAN_TM.lie(U), mu))),
            Neg(
                d_tilde(
                    N, iota_tilde(U, d(iota_tilde(V, mu)))
                )
            ),
            d_tilde(
                N,
                Act(Interior(lie_bracket(U, V)), mu),
            ),
        ),
        h,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_jacobi_compat_d11(
    N,
    omega: Expr,
    V: Expr,
    W: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, List[Theorem]]:
    """(D.11) [the first dual Jacobi compatibility in (4.40)]: the
    derivator of ``ℒ̃_ω`` over the Lie bracket is measured by the
    ``𝒦``-twisted tilde operators,

        ℒ̃_ω[V,W] − [ℒ̃_ωV, W] − [V, ℒ̃_ωW]
          − ℒ̃_{𝒦_V ω} W − 𝒦̃_{𝒦_W ω} V = 0

    — Lie Jacobi (cited instances) + Cartan relations.
    DECLARATION-FREE; probe ``h``."""
    from jacopy.central.tangent.lie_bracket import lie_bracket
    from jacopy.packages.drinfeld.calculus_conditions import kappa

    node = Act(
        Sum(
            lie_tilde_nambu(N, omega, lie_bracket(V, W)),
            Neg(
                lie_bracket(lie_tilde_nambu(N, omega, V), W)
            ),
            Neg(
                lie_bracket(V, lie_tilde_nambu(N, omega, W))
            ),
            Neg(lie_tilde_nambu(N, kappa(V, omega), W)),
            Neg(kappa_tilde_nambu(N, kappa(W, omega), V)),
        ),
        h,
    )
    from jacopy.central.tangent.cartan import (
        prove_with_bracket_identities,
    )

    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def g_z_pairing(N, a: Expr, b: Expr) -> Expr:
    """``g_Z(a, b) := ι_{Πa} b + ι_{Πb} a`` — the symmetric Z-metric
    [eq (6.17)]."""
    return Sum(
        Act(Interior(N.sharp_vf(a)), b),
        Act(Interior(N.sharp_vf(b)), a),
    )


def prove_jacobi_compat_d8(
    N,
    U: Expr,
    eta: Expr,
    mu: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 60000,
) -> ProofChain:
    """(D.8) [the first Jacobi compatibility (4.36) in tilde
    instantiation]: the Z-derivator of ``ℒ_U`` over the higher
    Koszul bracket is measured by the ``𝒦̃``-twisted operators,

        ℒ_U[η,μ]_Kos − [ℒ_Uη, μ]_Kos − [η, ℒ_Uμ]_Kos
          = ℒ_{𝒦̃_η U} μ + 𝒦_{𝒦̃_μ U} η.

    Paper route: two usual-Cartan commutator blocks — and indeed it
    closes DECLARATION-FREE at the raw expression level (no FI: the
    morphism property is never invoked)."""
    from jacopy.central.tangent.exterior import CARTAN_TM
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.calculus_conditions import kappa

    def L(X, x):
        return Act(CARTAN_TM.lie(X), x)

    node = Sum(
        L(U, nambu_koszul_bracket(N, eta, mu)),
        Neg(nambu_koszul_bracket(N, L(U, eta), mu)),
        Neg(nambu_koszul_bracket(N, eta, L(U, mu))),
        Neg(L(kappa_tilde_nambu(N, eta, U), mu)),
        Neg(kappa(kappa_tilde_nambu(N, mu, U), eta)),
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_jacobi_compat_d10(
    N,
    omega: Expr,
    eta: Expr,
    W: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 60000,
) -> ProofChain:
    """(D.10) [the third Jacobi compatibility (4.38) in tilde
    instantiation, with the ``g_Z`` metric and ``𝔻_Z = d``]:

        dι_{ℒ̃_ω W} η − dι_{d̃ι̃_η W} ω + dι_W [ω,η]_Kos
          + d g_Z(𝒦_W ω, η) − d g_Z(dι_W η, ω) = 0.

    Closes DECLARATION-FREE at the raw expression level via the
    usual Cartan relations."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.calculus_conditions import kappa
    from jacopy.packages.drinfeld.double import lie_tilde_nambu

    node = Sum(
        d(
            Act(
                Interior(lie_tilde_nambu(N, omega, W)), eta
            )
        ),
        Neg(
            d(
                Act(
                    Interior(
                        d_tilde(N, iota_tilde(W, eta))
                    ),
                    omega,
                )
            )
        ),
        d(
            Act(
                Interior(W),
                nambu_koszul_bracket(N, omega, eta),
            )
        ),
        d(g_z_pairing(N, kappa(W, omega), eta)),
        Neg(
            d(
                g_z_pairing(
                    N, d(iota_tilde(W, eta)), omega
                )
            )
        ),
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


# ------------------------------------------------------------------- #
# 6.J — stall-time difference-test citation (closes the p = 1 face)    #
# ------------------------------------------------------------------- #


def _node_size(x: Expr) -> int:
    if x.is_atom:
        n = 1
        slots = getattr(x, "rewritable_slots", None)
        if slots:
            for sl in slots:
                n += _node_size(sl)
        return n
    n = 1
    for c in x.children:
        n += _node_size(c)
    return n


def _p1_deep_instances(engine, N, omega, eta, W, h, registry):
    """The p = 1 deep-instance normal forms (6.J): the s-BRIDGE
    (⟨b,[W,Πa]⟩ = W(Π(a,b)) + Π-legs — the 5.E.2b family, derivable
    from the pairing Leibniz + magic, packaged as a declared-FI /
    congruence instance) and the FI-ON-EXACTS family (θ = d(Wh)),
    each in ±, d-, W- and Π-PAIRED lifts, normalized by the CITING
    engine. Returns ``[(name, lhs_nf, proof_chain)]``."""
    from jacopy.core.multi_eval import MultiEval
    from jacopy.core.pairing import Pairing
    from jacopy.central.tangent.lie_bracket import lie_bracket
    from jacopy.packages.poisson.tilde import _normalized_by
    from jacopy.proof.step import ProofStep

    def ME(*a):
        return MultiEval(
            N.pi, *a, alternating=True, slot_kind="covector"
        )

    book = []
    for (a, b) in ((omega, eta), (eta, omega)):
        book.append(("sbridge", Sum(
            ME(d(Pairing(b, W)), a),
            ME(Act(Interior(W), d(b)), a),
            Neg(Pairing(b, lie_bracket(W, N.sharp_vf(a)))),
            Act(W, ME(a, b)),
        )))
        theta = d(Act(W, h))
        book.append(("fiexact", Sum(
            ME(theta, nambu_koszul_bracket(N, a, b)),
            ME(d(ME(theta, b)), a),
            Neg(ME(d(ME(theta, a)), b)),
        )))
    out = []
    for tag, S in book:
        chain = ProofChain([ProofStep(
            S,
            Integer(0),
            rule=(
                "p=1 deep instance: declared FI / pairing-bridge "
                "(+ congruence lifts d, W, Π-pairing)"
            ),
            justification="axiom instance + congruence",
            provenance_tag="axiom",
        )])
        lifts = (
            ("", S),
            ("_d", d(S)),
            ("_W", Act(W, S)),
            ("_pdh", ME(d(S), d(h))),
            ("_pdWh", ME(d(S), d(Act(W, h)))),
        )
        for ltag, base_seed in lifts:
            for sgn, gtag in (
                ((lambda x: x), "p"),
                (Neg, "n"),
            ):
                nf = _normalized_by(
                    engine, sgn(base_seed), registry
                )
                if nf != Integer(0):
                    out.append(
                        (f"{tag}{ltag}_{gtag}", nf, chain)
                    )
    return out


def _stall_difference_prove(
    engine, node, instances, registry, *, max_rounds: int = 10
):
    """6.J stall-time citation: normalize ``node`` to its stall
    residual, then repeatedly subtract instance lhs's whenever the
    result is STRICTLY SMALLER (node-size metric ⟹ termination).
    Soundness: each accepted step rewrites
    ``R → NF(R − lhs)`` with ``lhs = 0`` a cited instance and NF a
    chain of registered-rule applications. Returns a
    :class:`ProofChain` or raises :class:`ProofFailure`."""
    from jacopy.packages.poisson.tilde import _normalized_by
    from jacopy.proof.step import ProofStep
    from jacopy.proof.strategies import ProofFailure

    steps = []
    residual = _normalized_by(engine, node, registry)
    if residual != node:
        steps.append(ProofStep(
            node, residual,
            rule="normalize (engine fixpoint + simplify)",
            justification="registered rules",
        ))
    rounds = 0
    while residual != Integer(0) and rounds < max_rounds:
        rounds += 1
        progressed = False
        for name, lhs, chain in instances:
            cand = _normalized_by(
                engine, Sum(residual, Neg(lhs)), registry
            )
            if _node_size(cand) < _node_size(residual):
                step = ProofStep(
                    residual, cand,
                    rule=f"cite instance {name}: subtract lhs = 0",
                    justification=(
                        "instance + normalization (6.J "
                        "stall-time difference test)"
                    ),
                    provenance_tag="axiom",
                )
                for sub in chain:
                    step.add_child(sub)
                steps.append(step)
                residual = cand
                progressed = True
                if residual == Integer(0):
                    break
        if not progressed:
            break
    if residual != Integer(0):
        raise ProofFailure(
            "stall-difference citation left residual "
            f"{residual._repr_inner()}"
        )
    return ProofChain(steps)
