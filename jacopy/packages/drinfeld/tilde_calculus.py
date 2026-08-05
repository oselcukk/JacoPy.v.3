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
        if X._repr_inner() <= Y._repr_inner():
            return None
        return sign, X, Y, inner.arg

    def matches(self, expr: Expr) -> bool:
        return self._parts(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        sign, X, Y, x = self._parts(expr)
        core = Act(Interior(Y), Act(Interior(X), x))
        return core if sign else Neg(core)


def _tilde_engine(N, registry, *, declare_fi: bool):
    from jacopy.packages.drinfeld.twist import _twist_engine

    eng = _twist_engine(N, registry)
    eng.register(LieIotaCommutatorDefinition())
    eng.register(InteriorAnticommuteDefinition())
    if declare_fi:
        eng.register(NambuMorphismDeclaration(N))
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
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=engine,
        max_steps=max_steps,
    )


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
