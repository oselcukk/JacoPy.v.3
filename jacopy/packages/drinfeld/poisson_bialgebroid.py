"""
The Poisson Lie-bialgebroid compatibility conditions (Phase 6.B) —
drinfeld paper [arXiv:2312.06584] eqs (4.36)-(4.38), instantiated on
``(A, Z) = (TM, T*M_π)`` with the Koszul bracket on ``Z`` and

    ℒ_U η          (usual Lie derivative),   ι_U η = ⟨η, U⟩,
    d              (usual exterior d),        𝒦_W = −ι_W d,
    ℒ̃_ω V = [π♯ω, V] + π♯ ι_V dω,           ι̃_ω V = ⟨ω, V⟩,
    d̃ = −π♯ d     [paper §6, Π = π].

* **(4.37)** ``ℒ_{d̃ι̃_η U} μ = −[dι_U η, μ]_π`` — closes BARE
  (27 steps): pure magic + Koszul mechanics.
* **(4.36)** ``𝒟^Z_{ℒ_U}(η, μ) = ℒ_{𝒦̃_η U} μ + 𝒦_{𝒦̃_μ U} η``
  (the derivator condition, PDF item 10f in action) — closes with
  cited VF-Jacobi instances (Sum-subset) + a Y-lifted scalar
  identity citation (mechanical core + labeled congruence).
* **(4.38)** at p = 1 the ``𝔻_Z g_Z``-side vanishes
  (``g_Z(ω,η) = 2(π(ω,η) + π(η,ω)) = 0`` by alternation) and the
  three-term left side closes.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.proof.theorems import Theorem, TheoremBook, cite
from jacopy.central.objects.interior import Interior

_AUX_COUNTER = 0
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.packages.poisson.core import PoissonStructure, SharpVF
from jacopy.packages.poisson.koszul import KoszulBracket


def _L(X: Expr, x: Expr) -> Expr:
    return Act(CARTAN_TM.lie(X), x)


def kappa_cartan(W: Expr, omega: Expr) -> Expr:
    """``𝒦_W ω = −ι_W dω`` (the Cartan-calculus 𝒦)."""
    return Neg(Act(Interior(W), d(omega)))


def kappa_tilde(P: PoissonStructure, eta: Expr, V: Expr) -> Expr:
    """``𝒦̃_η V = [V, π♯η] − π♯(ℒ_V η)`` [paper eq (4.41)]."""
    return Sum(
        LieBracketVF(V, SharpVF(P.pi, eta)),
        Neg(SharpVF(P.pi, _L(V, eta))),
    )


def lie_tilde_vf(P: PoissonStructure, omega: Expr, V: Expr) -> Expr:
    """``ℒ̃_ω V = [π♯ω, V] + π♯(ι_V dω)`` [paper eq (6.6)]."""
    return Sum(
        LieBracketVF(SharpVF(P.pi, omega), V),
        SharpVF(P.pi, Act(Interior(V), d(omega))),
    )


def prove_compat_condition_two(
    P: PoissonStructure,
    eta: Expr,
    mu: Expr,
    U: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[eq (4.37)]: ``ℒ_{d̃ι̃_η U} μ + [dι_U η, μ]_π = 0`` evaluated
    on ``Y`` — declaration-free."""
    from jacopy.packages.poisson.showcase import showcase_engine

    s = Pairing(eta, U)
    lhs = Sum(
        Neg(Act(CARTAN_TM.lie(SharpVF(P.pi, d(s))), mu)),
        KoszulBracket(P.pi, d(s), mu),
    )
    return ExpandAndSimplify().prove(
        Pairing(lhs, Y),
        Integer(0),
        registry=registry,
        engine=showcase_engine(
            P, registry=registry, declare_poisson=False
        ),
    )


def _cite_compat_instances(engine, P, U, Y, forms_, registry):
    """VF-Jacobi Sum-subset citations for ``(U, Y, π♯·)`` triples +
    the Y-lifted scalar identity
    ``⟨η,[U,π♯μ]⟩ − U(π(μ,η)) + π(μ, ℒ_U η)``-family (mechanical
    core, labeled congruence lift)."""
    from jacopy.core.multi_eval import MultiEval
    from jacopy.central.objects import functions
    from jacopy.central.tangent.lie_bracket import (
        jacobi_combination_theorems,
    )
    from jacopy.packages.poisson.showcase import showcase_engine
    from jacopy.packages.poisson.tilde import _normalized_by

    global _AUX_COUNTER
    _AUX_COUNTER += 1
    (f_aux,) = functions(
        f"f_aux_compat{_AUX_COUNTER}", registry=registry
    )
    book = TheoremBook()
    names = []
    for k, x in enumerate(forms_):
        thms = jacobi_combination_theorems(
            U, Y, SharpVF(P.pi, x), f_aux,
            registry=registry, engine=engine,
        )
        for j, thm in enumerate(thms):
            nm = f"compat_vfjac_{k}_{j}"
            book.add(
                Theorem(
                    name=nm,
                    statement=thm.statement,
                    lhs=thm.lhs,
                    rhs=thm.rhs,
                    proof=thm.proof,
                    generality="instance",
                )
            )
            names.append(nm)
    bare = showcase_engine(P, registry=registry, declare_poisson=False)
    from jacopy.central.tangent.engine import tangent_engine

    tm_only = tangent_engine(registry=registry)
    for k, (a, b) in enumerate(
        [(forms_[0], forms_[1]), (forms_[1], forms_[0])]
    ):
        # BRIDGE instance (the ⟨η,[U,V]⟩ ↔ Palais link of the
        # 5.E.2b plan, realized as a CITATION — no rewrite rule, no
        # ping-pong). Two mechanical legs on the node
        # ⟨ℒ_U a, π♯b⟩: the full engine gives the compact π-eval,
        # the tangent-only engine gives the Palais split; combining
        # yields the bracket-pairing identity. All steps mechanical.
        node = Pairing(_L(U, a), SharpVF(P.pi, b))
        n_split = _normalized_by(bare, node, registry)
        from jacopy.core.multi_eval import MultiEval as _ME
        from jacopy.proof.expansion import (
            ExpansionEngine as _EE,
        )
        from jacopy.packages.poisson.core import (
            SharpEvaluationDefinition,
        )

        sharp_only = _EE([SharpEvaluationDefinition(P)])
        n_compact = _normalized_by(sharp_only, node, registry)
        leg1 = ExpandAndSimplify().prove(
            node, n_split, registry=registry, engine=bare
        )
        leg2 = ExpandAndSimplify().prove(
            node,
            n_compact,
            registry=registry,
            engine=sharp_only,
        )
        S = Sum(n_compact, Neg(n_split))
        n_tm, n_full = n_compact, n_split  # adlar aşağıda
        step1 = ProofStep(
            node, n_full,
            rule="sharp evaluation (full engine)",
            justification="mechanical",
        )
        for st in leg1:
            step1.add_child(st)
        step2 = ProofStep(
            node, n_tm,
            rule="intrinsic-L Palais split (tangent engine)",
            justification="mechanical",
        )
        for st in leg2:
            step2.add_child(st)
        step3 = ProofStep(
            S, Integer(0),
            rule="combine the two legs",
            justification="both equal ⟨ℒ_U a, π♯b⟩",
        )
        chain = ProofChain([step1, step2, step3])
        for lift, tag in ((None, "flat"), (Y, "ylift")):
          for sgn, stag in (((lambda x: x), "p"), (Neg, "n")):
            base_seed = S if lift is None else Act(lift, S)
            seed = sgn(base_seed)
            lifted = _normalized_by(engine, seed, registry)
            if lifted == Integer(0):
                continue
            steps = list(chain)
            if lift is not None:
                steps = steps + [
                    ProofStep(
                        seed, Integer(0),
                        rule="congruence: apply Y to both sides",
                        justification="Y(0) = 0",
                    )
                ]
            nm = f"compat_bridge_{k}_{tag}_{stag}"
            book.add(
                Theorem(
                    name=nm,
                    statement="bracket-pairing bridge instance",
                    lhs=lifted,
                    rhs=Integer(0),
                    proof=ProofChain(steps),
                    generality="instance",
                )
            )
            names.append(nm)
    # MAGIC-difference family: the residual may carry the
    # magic-SPLIT of ℒ_U a (d⟨a,U⟩ + ι_U da) — the instance is the
    # π-paired, Y-lifted magic defect (core mechanical via the
    # Phase 2 magic theorem; pairing and Y-application are labeled
    # congruence steps).
    from jacopy.central.tangent.cartan import (
        prove_cartan_magic_on_one_forms,
    )

    for k, (a, b) in enumerate(
        [(forms_[0], forms_[1]), (forms_[1], forms_[0])]
    ):
        Mdiff = Sum(
            _L(U, a),
            Neg(d(Act(Interior(U), a))),
            Neg(Act(Interior(U), d(a))),
        )
        core = prove_cartan_magic_on_one_forms(
            a, U, Y, registry=registry
        )
        step1 = ProofStep(
            Mdiff, Integer(0),
            rule="Cartan magic (Phase 2 theorem)",
            justification="ℒ_U a = dι_U a + ι_U da (mechanical)",
        )
        for st in core:
            step1.add_child(st)
        from jacopy.core.multi_eval import MultiEval as _ME2

        paired = _ME2(
            P.pi, Mdiff, b, alternating=True, slot_kind="covector"
        )
        step2 = ProofStep(
            paired, Integer(0),
            rule="congruence: pair with b via π",
            justification="π(0, b) = 0",
        )
        for lift, tag in ((None, "flat"), (Y, "ylift")):
            for sgn, stag in (((lambda x: x), "p"), (Neg, "n")):
                base_seed = (
                    paired if lift is None else Act(lift, paired)
                )
                seed = sgn(base_seed)
                lifted = _normalized_by(engine, seed, registry)
                if lifted == Integer(0):
                    continue
                steps = [step1, step2]
                if lift is not None:
                    steps = steps + [
                        ProofStep(
                            seed, Integer(0),
                            rule="congruence: apply Y",
                            justification="Y(0) = 0",
                        )
                    ]
                nm = f"compat_magic_{k}_{tag}_{stag}"
                if nm in book:
                    continue
                book.add(
                    Theorem(
                        name=nm,
                        statement="π-paired magic defect",
                        lhs=lifted,
                        rhs=Integer(0),
                        proof=ProofChain(steps),
                        generality="instance",
                    )
                )
                names.append(nm)
    # COMBINED family: magic-defect + bridge in one seed (the 4.38
    # residual mixes both; sequential subset matching cannot thread
    # the intermediate).
    for k, (a, b) in enumerate(
        [(forms_[0], forms_[1]), (forms_[1], forms_[0])]
    ):
        from jacopy.core.multi_eval import MultiEval as _ME3

        Mdiff = Sum(
            _L(U, a),
            Neg(d(Act(Interior(U), a))),
            Neg(Act(Interior(U), d(a))),
        )
        paired = _ME3(
            P.pi, Mdiff, b, alternating=True, slot_kind="covector"
        )
        node = Pairing(_L(U, a), SharpVF(P.pi, b))
        n_split = _normalized_by(bare, node, registry)
        from jacopy.proof.expansion import ExpansionEngine as _EE3
        from jacopy.packages.poisson.core import (
            SharpEvaluationDefinition as _SED,
        )

        n_compact = _normalized_by(
            _EE3([_SED(P)]), node, registry
        )
        S_bridge = Sum(n_compact, Neg(n_split))
        combined = Sum(paired, S_bridge)
        step = ProofStep(
            combined, Integer(0),
            rule="magic defect + sharp/Palais bridge, combined",
            justification=(
                "both summands vanish (magic theorem; two mechanical "
                "normal forms of ⟨ℒ_U a, π♯b⟩)"
            ),
        )
        for lift, tag in ((None, "flat"), (Y, "ylift")):
            for sgn, stag in (((lambda x: x), "p"), (Neg, "n")):
                base_seed = (
                    combined if lift is None else Act(lift, combined)
                )
                seed = sgn(base_seed)
                lifted = _normalized_by(engine, seed, registry)
                if lifted == Integer(0):
                    continue
                nm = f"compat_comb_{k}_{tag}_{stag}"
                if nm in book:
                    continue
                book.add(
                    Theorem(
                        name=nm,
                        statement="combined magic+bridge instance",
                        lhs=lifted,
                        rhs=Integer(0),
                        proof=ProofChain([step]),
                        generality="instance",
                    )
                )
                names.append(nm)
    if names:
        cite(engine, book, *names)


def prove_compat_condition_one(
    P: PoissonStructure,
    eta: Expr,
    mu: Expr,
    U: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[eq (4.36), the DERIVATOR condition — PDF item 10f]:

        𝒟^Z_{ℒ_U}(η, μ) = ℒ_{𝒦̃_η U} μ + 𝒦_{𝒦̃_μ U} η

    with ``𝒟^Z`` the derivator of ``ℒ_U`` w.r.t. the Koszul
    bracket, evaluated on ``Y``. Cited VF-Jacobi instances +
    Y-lifted scalar identities."""
    from jacopy.packages.poisson.showcase import showcase_engine

    kb = lambda a, b: KoszulBracket(P.pi, a, b)
    lhs = Sum(
        _L(U, kb(eta, mu)),
        Neg(kb(_L(U, eta), mu)),
        Neg(kb(eta, _L(U, mu))),
        Neg(Act(CARTAN_TM.lie(kappa_tilde(P, eta, U)), mu)),
        Neg(kappa_cartan(kappa_tilde(P, mu, U), eta)),
    )
    engine = showcase_engine(
        P, registry=registry, declare_poisson=False
    )
    _cite_compat_instances(engine, P, U, Y, (eta, mu), registry)
    return ExpandAndSimplify().prove(
        Pairing(lhs, Y),
        Integer(0),
        registry=registry,
        engine=engine,
        max_steps=20000,
    )


def prove_compat_condition_three_p1(
    P: PoissonStructure,
    omega: Expr,
    eta: Expr,
    W: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """[eq (4.38) at p = 1]: the ``𝔻_Z g_Z`` side vanishes by
    alternation, leaving

        dι_{ℒ̃_ω W} η − dι_{d̃ι̃_η W} ω + dι_W [ω,η]_π = 0

    evaluated on ``Y``."""
    from jacopy.packages.poisson.showcase import showcase_engine

    s = Pairing(eta, W)
    lhs = Sum(
        d(Pairing(eta, lie_tilde_vf(P, omega, W))),
        Neg(d(Pairing(omega, Neg(SharpVF(P.pi, d(s)))))),
        d(Pairing(KoszulBracket(P.pi, omega, eta), W)),
    )
    engine = showcase_engine(
        P, registry=registry, declare_poisson=False
    )
    _cite_compat_instances(engine, P, W, Y, (omega, eta), registry)
    return ExpandAndSimplify().prove(
        Pairing(lhs, Y),
        Integer(0),
        registry=registry,
        engine=engine,
        max_steps=20000,
    )
