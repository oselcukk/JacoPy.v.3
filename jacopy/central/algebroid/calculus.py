"""
The algebroid Cartan calculus (Phase 3.F) — ``d_E``, ``L^E``, ``ι``
from the pair ``(ρ_E, [·,·]_E)``.

This module *instantiates* the generic
:class:`~jacopy.central.calculus.bracket_calculus.BracketCalculus`
with an algebroid's anchor and bracket — exactly how the TM case
instantiates it with ``(id, [·,·]_Lie)`` (PDF item 8a); no parallel
operator code. The canonical definitions are inherited verbatim:

* ``d_E`` — Palais intrinsic formula driven by ``ρ`` and ``[·,·]_E``:
  ``⟨d_E f, u⟩ = ρ(u)(f)`` (so ``d_E`` on functions COINCIDES with the
  coboundary ``D`` — provable, see
  :func:`prove_d_coincides_with_coboundary`), and
  ``(d_E α)(u,v) = ρ(u)(α(v)) − ρ(v)(α(u)) − α([u,v])`` on E-1-forms.
* ``L^E_u`` — core cases ``L^E_u f = ρ(u)(f)``, ``L^E_u v = [u,v]_E``
  plus the covariant-slot evaluation formula.
* ``ι_u`` — the generic slot-insertion interior product (bundle-blind
  by construction).

**Conditional Cartan structure** — the theorems hold exactly under
their hierarchy declarations, and fail honestly otherwise:

* the Cartan magic formula ``L^E_u = d_E ι_u + ι_u d_E`` needs NO
  hierarchy axiom (pure consequence of the definitions, any anchored
  bundle with an ℝ-bilinear bracket);
* ``d_E² = 0`` on functions ⟺ the anchor is a bracket morphism
  (closes under ``anchor-morphism``; under ``leibniz`` it closes by
  citing the Phase 3.D theorem instead);
* ``d_E² = 0`` on E-1-forms additionally needs the Leibniz-Jacobi
  identity, entering through Jacobiator-backed instance theorems
  (:func:`jacobi_combination_theorem`).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import Theorem, TheoremBook, cite
from jacopy.central.calculus import BracketCalculus, LieDerivative
from jacopy.central.objects.interior import Interior
from jacopy.central.algebroid.context import Algebroid


def algebroid_calculus(alg: Algebroid) -> BracketCalculus:
    """The bracket calculus of ``alg`` — ``(ρ_E, [·,·]_E)``.

    On the tangent algebroid this IS the TM Cartan calculus (the
    reduction is literal, PDF item 8a).
    """
    if not isinstance(alg, Algebroid):
        raise TypeError("algebroid_calculus expects an Algebroid")
    if alg.is_tangent:
        from jacopy.central.tangent.exterior import CARTAN_TM

        return CARTAN_TM
    return BracketCalculus(
        f"Cartan-{alg.name}",
        anchor=alg.anchor,
        bracket=alg.bracket,
        d_name=f"d_{alg.name}",
        lie_name=f"L^{alg.name}",
    )


def d_E(alg: Algebroid, omega: Expr) -> Act:
    """``d_E ω`` — an inert node; evaluation expands via Palais."""
    if not isinstance(omega, Expr):
        raise TypeError("d_E expects an Expr form")
    return Act(algebroid_calculus(alg).d, omega)


def lie_E(alg: Algebroid, u: Expr) -> LieDerivative:
    """``L^E_u`` — the algebroid Lie derivative operator."""
    return algebroid_calculus(alg).lie(u)


def _engine(alg: Algebroid, registry: Optional[PropertyRegistry]):
    from jacopy.central.algebroid.engine import algebroid_engine

    return algebroid_engine(alg, registry=registry)


# --------------------------------------------------------------------- #
# Definitional checks (no hierarchy axioms needed)                       #
# --------------------------------------------------------------------- #


def prove_d_on_functions(
    alg: Algebroid,
    u: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨d_E f, u⟩ = ρ(u)(f)`` — the 0-form case of the definition."""
    return ExpandAndSimplify().prove(
        Pairing(d_E(alg, f), u),
        Act(alg.anchor(u), f),
        registry=registry,
        engine=_engine(alg, registry),
    )


def prove_d_coincides_with_coboundary(
    alg: Algebroid,
    u: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨d_E f, u⟩ = ⟨Df, u⟩`` — the calculus' exterior derivative
    agrees on functions with the coboundary of Phase 3.A (both are
    defined by ``ρ(u)(f)``; the design-coherence check)."""
    return ExpandAndSimplify().prove(
        Pairing(d_E(alg, f), u),
        Pairing(alg.D(f), u),
        registry=registry,
        engine=_engine(alg, registry),
    )


def prove_palais_on_one_forms(
    alg: Algebroid,
    alpha: Expr,
    u: Expr,
    v: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``(d_E α)(u, v) = ρ(u)(α(v)) − ρ(v)(α(u)) − α([u,v]_E)``."""
    lhs = MultiEval(
        d_E(alg, alpha), u, v, alternating=True, slot_kind="vector"
    )
    rhs = Sum(
        Act(alg.anchor(u), Pairing(alpha, v)),
        Neg(Act(alg.anchor(v), Pairing(alpha, u))),
        Neg(Pairing(alpha, alg.bracket(u, v))),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(alg, registry)
    )


def prove_lie_core_cases(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, ProofChain]:
    """``L^E_u f = ρ(u)(f)`` and ``L^E_u v = [u, v]_E``."""
    L = lie_E(alg, u)
    eng = _engine(alg, registry)
    chain_f = ExpandAndSimplify().prove(
        Act(L, f),
        Act(alg.anchor(u), f),
        registry=registry,
        engine=eng,
    )
    chain_v = ExpandAndSimplify().prove(
        Act(L, v),
        alg.bracket(u, v),
        registry=registry,
        engine=_engine(alg, registry),
    )
    return chain_f, chain_v


def prove_cartan_magic_on_functions(
    alg: Algebroid,
    u: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``L^E_u f = (d_E ι_u + ι_u d_E) f`` — a THEOREM of the
    definitions; needs no hierarchy declaration."""
    iota = Interior(u)
    rhs = Sum(
        Act(algebroid_calculus(alg).d, Act(iota, f)),
        Act(iota, d_E(alg, f)),
    )
    return ExpandAndSimplify().prove(
        Act(lie_E(alg, u), f),
        rhs,
        registry=registry,
        engine=_engine(alg, registry),
    )


def prove_cartan_magic_on_one_forms(
    alg: Algebroid,
    u: Expr,
    alpha: Expr,
    v: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L^E_u α, v⟩ = ⟨(d_E ι_u + ι_u d_E) α, v⟩`` — again pure
    definitions: the magic formula predates the hierarchy."""
    iota = Interior(u)
    lhs = Pairing(Act(lie_E(alg, u), alpha), v)
    rhs = Sum(
        Pairing(Act(algebroid_calculus(alg).d, Act(iota, alpha)), v),
        Pairing(Act(iota, d_E(alg, alpha)), v),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(alg, registry)
    )


# --------------------------------------------------------------------- #
# Conditional Cartan: d² = 0 under the hierarchy                         #
# --------------------------------------------------------------------- #


def prove_d_squared_zero_on_functions(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    engine=None,
) -> ProofChain:
    """``(d_E d_E f)(u, v) = 0`` — holds exactly when the anchor is a
    bracket morphism: the obstruction reduces to
    ``[ρ(u),ρ(v)](f) − ρ([u,v])(f)``.

    Closes under a declared ``anchor-morphism``; on a Leibniz
    algebroid pass an ``engine`` with the Phase 3.D theorem cited.
    Fails honestly otherwise.
    """
    lhs = MultiEval(
        d_E(alg, d_E(alg, f)), u, v, alternating=True, slot_kind="vector"
    )
    return ExpandAndSimplify().prove(
        lhs,
        Integer(0),
        registry=registry,
        engine=engine if engine is not None else _engine(alg, registry),
    )


def jacobi_combination_theorem(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Theorem:
    """The instance theorem
    ``[u,[v,w]] − [[u,v],w] − [v,[u,w]] = 0`` on a Jacobi-declared
    algebroid, derived through the Jacobiator node:
    ``J(u,v,w) = 0`` (declared) and ``J(u,v,w) = the combination``
    (definitional expansion with the declaration withheld).

    The citable bracket-level Jacobi fact — the algebroid analogue of
    the TM repair loop's vector-level bracket identities (2.C/2.D);
    ``d_E² = 0`` beyond functions consumes it.
    """
    from jacopy.central.algebroid.engine import algebroid_engine
    from jacopy.central.algebroid.operators import jacobiator
    from jacopy.central.algebroid.theorems import (
        _normalize,
        _without_jacobi,
    )

    if not alg.declares("jacobi"):
        raise ProofFailure(
            "the bracket-level Jacobi combination needs the 'jacobi' "
            f"declaration which {alg!r} does not make"
        )
    full = algebroid_engine(alg, registry=registry)
    noJ = algebroid_engine(_without_jacobi(alg), registry=registry)
    j = jacobiator(alg, u, v, w)

    out, jac_steps = full.expand(j)
    if out != Integer(0) or not jac_steps:
        raise ProofFailure(  # pragma: no cover - guarded above
            "Jacobi-combination derivation: J(u,v,w) did not vanish"
        )
    combination, exp_steps = _normalize(j, noJ, registry)

    step_zero = jac_steps[0]
    step_expand = ProofStep(
        j,
        combination,
        rule="expand J(u,v,w) definitionally",
        justification="Jacobiator definition (declaration withheld)",
    )
    for s in exp_steps:
        step_expand.add_child(s)
    step_conclude = ProofStep(
        combination,
        Integer(0),
        rule="both sides of the Jacobiator",
        justification="J(u,v,w) = 0 and J(u,v,w) = the combination",
    )
    chain = ProofChain([step_zero, step_expand, step_conclude])
    names = tuple(
        s._repr_inner() for s in (u, v, w)
    )
    return Theorem(
        name=f"jacobi_combination_{alg.name}_" + "_".join(names),
        statement=(
            f"[{names[0]},[{names[1]},{names[2]}]] − "
            f"[[{names[0]},{names[1]}],{names[2]}] − "
            f"[{names[1]},[{names[0]},{names[2]}]] = 0 on {alg.name}"
        ),
        lhs=combination,
        rhs=Integer(0),
        proof=chain,
        generality="instance",
        from_axioms=(f"Leibniz-Jacobi ({alg.name})",),
        notes="bracket-level Jacobi via the Jacobiator carrier node",
    )


def prove_d_squared_zero_on_one_forms(
    alg: Algebroid,
    alpha: Expr,
    u: Expr,
    v: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """``(d_E d_E α)(u, v, w) = 0`` — needs the anchor morphism AND
    the Leibniz-Jacobi identity: the residual splits into anchor
    commutators (killed by the morphism) and a bracket-level Jacobi
    combination inside the ``α``-slot (killed by cited
    :func:`jacobi_combination_theorem` instances).

    Requires ``anchor-morphism``, ``jacobi`` AND ``antisymmetric``
    declared — ``d_E² = 0`` beyond functions is genuinely a LIE
    algebroid theorem: for a non-antisymmetric Leibniz bracket the
    residual double-bracket combination is NOT a Jacobi instance (the
    engine surfaced exactly this when the axiom was missing). Raises
    an honest :class:`ProofFailure` otherwise. Returns the chain and
    the cited instance theorems.
    """
    from jacopy.algorithms.simplify import simplify

    missing = [
        d
        for d in ("anchor-morphism", "jacobi", "antisymmetric")
        if not alg.declares(d)
    ]
    if missing:
        raise ProofFailure(
            "d_E² = 0 on E-1-forms is not derivable here: the proof "
            f"needs the declarations {missing} which {alg!r} does not "
            "make"
        )
    used: List[Theorem] = []
    book = TheoremBook()
    engine = _engine(alg, registry)
    for triple in ((u, v, w), (v, u, w), (u, w, v)):
        thm = jacobi_combination_theorem(alg, *triple, registry=registry)
        if thm.name in book:
            continue
        book.add(thm)
        used.append(thm)
        cite(engine, book, thm.name)
        # The combination may surface with the opposite overall sign;
        # register the negated instance alongside (same derivation
        # plus one sign step).
        neg_lhs = simplify(Neg(thm.lhs), registry)
        neg_chain = ProofChain(list(thm.proof.steps))
        neg_chain.append(
            ProofStep(
                neg_lhs,
                Integer(0),
                rule="negate both sides",
                justification="the combination vanishes, so does its negation",
            )
        )
        thm_neg = Theorem(
            name=thm.name + "_neg",
            statement=f"−({thm.statement})",
            lhs=neg_lhs,
            rhs=Integer(0),
            proof=neg_chain,
            generality="instance",
            from_axioms=thm.from_axioms,
        )
        book.add(thm_neg)
        used.append(thm_neg)
        cite(engine, book, thm_neg.name)
    lhs = MultiEval(
        d_E(alg, d_E(alg, alpha)),
        u,
        v,
        w,
        alternating=True,
        slot_kind="vector",
    )
    chain = ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )
    return chain, used
