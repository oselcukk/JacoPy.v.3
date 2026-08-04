"""
Algebroid theorems (Phases 3.D + 3.E.3) — relations BETWEEN declared
axioms.

Phase 3.E.3 adds the metric-family results [B 4.11]:

* :func:`prove_right_leibniz_from_metric` — metric invariance (C1) +
  the E-metric's definitional non-degeneracy ⇒ right-Leibniz
  (so every metric algebroid is an almost-Courant algebroid). The
  auxiliary-section argument: expand ``ρ(u)(g(f·v, w))`` two ways
  (C1 applied at the composite site vs bilinearity-first), subtract,
  and cancel the generic ``w`` by non-degeneracy.
* :func:`prove_left_leibniz_from_metric` — additionally with the
  symmetric part (C2): the left-Leibniz rule holds with the CONCRETE
  locality term ``L(Df, u, v) = g(u, v)·g⁻¹(Df)``:
  ``[f·u, v] = f·[u,v] − ρ(v)(f)·u + g(u,v)·g⁻¹(Df)``.

The flagship result of 3.D [B Thm 4.1]:

    **right-Leibniz + Leibniz-Jacobi  ⇒  anchor morphism**
    ("every Leibniz algebroid is a pre-Leibniz algebroid")

Mechanized proof (the classical auxiliary-section argument):

1. ``J(u, v, f·w) = 0`` for a generic function ``f`` and a generic
   auxiliary section ``w``            [declared Leibniz-Jacobi]
2. Expand ``J(u, v, f·w)`` definitionally, driving every bracket with
   the declared right-Leibniz rule; the normal form splits as
   ``A·w + f·(bracket form of J(u,v,w))`` where
   ``A = ρ(u)(ρ(v)(f)) − ρ(v)(ρ(u)(f)) − ρ([u,v])(f)``.
3. ``f·J(u, v, w) = 0``               [declared Leibniz-Jacobi again]
4. Hence ``A·w = 0`` with ``w`` generic ⟹ ``A = 0``
   (agreement on generators: a section-coefficient identity holding
   against a generic section forces the coefficient to vanish).
5. Solve ``A = 0`` for ``ρ([u,v])(f)`` and recognize the commutator:
   ``ρ([u,v])(f) = [ρ(u), ρ(v)](f)``.
6. ``f`` generic ⟹ ``ρ([u,v]) = [ρ(u), ρ(v)]`` as vector fields
   (agreement on generators once more).

Every mechanical leg (steps 1-3, 5's commutator recognition) is an
engine computation whose steps are attached as children of the
narrative step; steps 4 and 6 are the two generator-agreement
inferences, recorded as explicit synthetic steps. Nothing is assumed:
if the declarations are missing, the expansion stalls and the function
raises an honest :class:`ProofFailure`.

Per the definition policy the proven morphism enters later proofs only
through citation: the returned :class:`~jacopy.proof.theorems.Theorem`
(``lhs = ρ([u,v])`` node, ``rhs = [ρ(u), ρ(v)]``) is registered by the
caller and cited via :func:`jacopy.proof.theorems.cite`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.algorithms.product_rule import product_rule
from jacopy.algorithms.simplify import simplify
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import Theorem, TheoremBook, cite
from jacopy.central.algebroid.context import Algebroid, algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.operators import jacobiator


# --------------------------------------------------------------------- #
# Mechanical helpers                                                     #
# --------------------------------------------------------------------- #


def _normalize(
    expr: Expr,
    engine: ExpansionEngine,
    registry: Optional[PropertyRegistry],
) -> Tuple[Expr, List[ProofStep]]:
    """Drive ``expr`` to the engine's normal form, mirroring the
    :class:`ExpandAndSimplify` reduction loop (expand + product-rule
    fix-point, then canonical simplification, repeated)."""
    steps: List[ProofStep] = []
    current = expr
    for _outer in range(8):
        for _ in range(64):
            expanded, exp_steps = engine.expand(current)
            steps.extend(exp_steps)
            after = product_rule(expanded, registry)
            if after != expanded:
                steps.append(
                    ProofStep(
                        expanded,
                        after,
                        rule="product-rule",
                        justification="graded Leibniz + linearity",
                    )
                )
            if after == current:
                break
            current = after
        else:
            raise ProofFailure(
                "anchor-morphism derivation: expansion did not converge "
                f"on {expr._repr_inner()}"
            )
        reduced = simplify(current, registry)
        if reduced != current:
            steps.append(
                ProofStep(
                    current,
                    reduced,
                    rule="simplify",
                    justification="canonical-form pipeline",
                )
            )
        if reduced == current:
            break
        current = reduced
    return current, steps


def _terms(expr: Expr) -> Tuple[Expr, ...]:
    return expr.children if isinstance(expr, Sum) else (expr,)


def _strip_sign(term: Expr) -> Tuple[int, Expr]:
    sign = 1
    while isinstance(term, Neg):
        sign = -sign
        term = term.arg
    return sign, term


def _coefficient_of_generic_section(
    expr: Expr, w: Expr
) -> Tuple[Expr, ...]:
    """Split ``expr = Σ ±(cᵢ·w)`` into the signed coefficients ``±cᵢ``.

    Raises :class:`ProofFailure` when any term is not a product with
    exactly one factor equal to ``w`` — then the residual is NOT
    proportional to the generic section and the cancellation step
    would be unsound.
    """
    coeffs: List[Expr] = []
    for term in _terms(expr):
        sign, core = _strip_sign(term)
        if core == w:
            coeff: Expr = Integer(1)
        elif isinstance(core, Product):
            factors = list(core.children)
            hits = [i for i, c in enumerate(factors) if c == w]
            if len(hits) != 1:
                raise ProofFailure(
                    "anchor-morphism derivation: residual term "
                    f"{term._repr_inner()} is not linear in the generic "
                    f"section {w._repr_inner()}"
                )
            rest = factors[: hits[0]] + factors[hits[0] + 1 :]
            coeff = rest[0] if len(rest) == 1 else Product(*rest)
        else:
            raise ProofFailure(
                "anchor-morphism derivation: residual term "
                f"{term._repr_inner()} carries no factor of the generic "
                f"section {w._repr_inner()}"
            )
        coeffs.append(coeff if sign > 0 else Neg(coeff))
    return tuple(coeffs)


def _fresh_section(alg: Algebroid, *taken: Expr) -> Expr:
    """A generic auxiliary section whose name collides with none of
    ``taken``."""
    used = {t._repr_inner() for t in taken}
    for name in ("w", "z", "s", "r", "q", "t"):
        if name not in used:
            (sec,) = alg.sections(name)
            return sec
    (sec,) = alg.sections("w_aux")  # pragma: no cover - 6 collisions
    return sec


def _without_jacobi(alg: Algebroid) -> Algebroid:
    """The same algebroid context minus the ``jacobi`` declaration —
    used for the expansion legs, where the Jacobiator must unfold
    definitionally instead of vanishing outright."""
    return algebroid(
        alg.name,
        alg.bundle,
        anchor_name=alg.anchor_name,
        declare=tuple(sorted(alg.declarations - {"jacobi"})),
    )


# --------------------------------------------------------------------- #
# The flagship theorem                                                   #
# --------------------------------------------------------------------- #


def prove_anchor_morphism(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    w: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Prove ``ρ([u, v]) = [ρ(u), ρ(v)]`` from right-Leibniz + Jacobi.

    ``f`` must be a generic scalar function (registered degree 0);
    ``w`` an optional generic auxiliary section (a fresh one is created
    when omitted). Returns the proof chain and the citable
    :class:`Theorem` (generality ``"generic-function"``, scoped to the
    given sections ``u``, ``v``).

    Raises :class:`ValueError` when the statement is not a theorem in
    the given context (tangent algebroid: the morphism is literal;
    declared ``anchor-morphism``: it is an axiom there), and an honest
    :class:`ProofFailure` when a required declaration is missing or
    the mechanical expansion does not produce the expected shape.
    """
    if not isinstance(alg, Algebroid):
        raise TypeError("prove_anchor_morphism expects an Algebroid")
    if alg.is_tangent:
        raise ValueError(
            "on the tangent algebroid the anchor is the identity and the "
            "morphism property is definitional, not a theorem"
        )
    if alg.declares("anchor-morphism"):
        raise ValueError(
            "anchor-morphism is DECLARED on this algebroid — there it is "
            "an axiom, not a theorem; derive it on a context declaring "
            "only right-leibniz + jacobi"
        )
    missing = [
        d for d in ("right-leibniz", "jacobi") if not alg.declares(d)
    ]
    if missing:
        raise ProofFailure(
            "anchor morphism is not derivable here: the proof needs the "
            f"declarations {missing} which {alg!r} does not make"
        )

    if w is None:
        w = _fresh_section(alg, u, v)

    full = algebroid_engine(alg, registry=registry)
    noJ = algebroid_engine(_without_jacobi(alg), registry=registry)

    j_fw = jacobiator(alg, u, v, Product(f, w))
    j_w = jacobiator(alg, u, v, w)

    # Leg 1: J(u, v, f·w) = 0 by the declared Leibniz-Jacobi axiom.
    out, jac_steps = full.expand(j_fw)
    if out != Integer(0) or not jac_steps:
        raise ProofFailure(  # pragma: no cover - guarded by `missing`
            "anchor-morphism derivation: J(u,v,f·w) did not vanish under "
            "the declared Leibniz-Jacobi rule"
        )
    step_jacobi = jac_steps[0]

    # Leg 2: expand J(u, v, f·w) definitionally (jacobi withheld).
    n1, n1_steps = _normalize(j_fw, noJ, registry)
    step_expand_fw = ProofStep(
        j_fw,
        n1,
        rule="expand J(u,v,f·w) by definitions + declared right-Leibniz",
        justification=(
            "Jacobiator definition, bracket bilinearity, right-Leibniz, "
            "canonical simplification"
        ),
    )
    for s in n1_steps:
        step_expand_fw.add_child(s)

    # Leg 3: f·J(u, v, w) = 0 by the same axiom.
    fj_zero, fj_steps = _normalize(Product(f, j_w), full, registry)
    if fj_zero != Integer(0):
        raise ProofFailure(  # pragma: no cover - guarded by `missing`
            "anchor-morphism derivation: f·J(u,v,w) did not vanish under "
            "the declared Leibniz-Jacobi rule"
        )
    step_fj = ProofStep(
        Product(f, j_w),
        Integer(0),
        rule="f·J(u,v,w) = 0 (declared Leibniz-Jacobi)",
        justification="scalar multiple of the vanishing Jacobiator",
    )
    for s in fj_steps:
        step_fj.add_child(s)

    # Leg 4: expand f·J(u, v, w) definitionally and subtract:
    # N1 − N2 must be proportional to the generic section w.
    n2, n2_steps = _normalize(Product(f, j_w), noJ, registry)
    diff, diff_steps = _normalize(Sum(n1, Neg(n2)), noJ, registry)
    step_collect = ProofStep(
        Sum(n1, Neg(n2)),
        diff,
        rule="collect: J(u,v,f·w) − f·J(u,v,w) = A·w",
        justification=(
            "the bracket-valued terms cancel; only coefficients of the "
            "auxiliary section w survive"
        ),
    )
    for s in n2_steps + diff_steps:
        step_collect.add_child(s)

    coeffs = _coefficient_of_generic_section(diff, w)
    A = simplify(
        coeffs[0] if len(coeffs) == 1 else Sum(*coeffs), registry
    )

    # Both legs equal zero, so A·w = 0 with w generic ⟹ A = 0.
    step_cancel_w = ProofStep(
        diff,
        Integer(0),
        rule="agreement on generators (w)",
        justification=(
            "A·w = J(u,v,f·w) − f·J(u,v,w) = 0 − 0 and w is a generic "
            "section, hence A = 0"
        ),
    )

    # Solve A = 0 for the ρ([u,v])(f) term.
    target = Act(alg.anchor(alg.bracket(u, v)), f)
    solved: Optional[Expr] = None
    a_terms = list(_terms(A))
    for i, term in enumerate(a_terms):
        sign, core = _strip_sign(term)
        if core == target:
            rest = a_terms[:i] + a_terms[i + 1 :]
            rest_sum: Expr = (
                Integer(0)
                if not rest
                else (rest[0] if len(rest) == 1 else Sum(*rest))
            )
            solved = simplify(
                Neg(rest_sum) if sign > 0 else rest_sum, registry
            )
            break
    if solved is None:
        raise ProofFailure(
            "anchor-morphism derivation: the coefficient identity "
            f"{A._repr_inner()} = 0 does not contain the expected term "
            f"{target._repr_inner()}"
        )
    step_solve = ProofStep(
        target,
        solved,
        rule="solve A = 0 for ρ([u,v])(f)",
        justification="rearrange the vanishing coefficient identity",
    )

    # Recognize the commutator: [ρ(u), ρ(v)](f) equals the solved form.
    commutator = LieBracketVF(alg.anchor(u), alg.anchor(v))
    from jacopy.central.tangent.lie_bracket import (
        LieBracketActionDefinition,
    )

    comm_engine = algebroid_engine(_without_jacobi(alg), registry=registry)
    comm_engine.register(LieBracketActionDefinition())
    comm_chain = ExpandAndSimplify().prove(
        Act(commutator, f), solved, registry=registry, engine=comm_engine
    )
    step_commutator = ProofStep(
        Act(commutator, f),
        solved,
        rule="Lie bracket definition: [ρ(u), ρ(v)](f) = ρ(u)(ρ(v)(f)) − ρ(v)(ρ(u)(f))",
        justification="commutator of derivations (canonical definition)",
    )
    for s in comm_chain:
        step_commutator.add_child(s)

    # f generic ⟹ equality of the vector fields themselves.
    lhs_node = alg.anchor(alg.bracket(u, v))
    step_cancel_f = ProofStep(
        lhs_node,
        commutator,
        rule="agreement on generators (f)",
        justification=(
            "ρ([u,v])(f) = [ρ(u), ρ(v)](f) for a generic function f, "
            "and vector fields agreeing on all functions are equal"
        ),
    )

    chain = ProofChain(
        [
            step_jacobi,
            step_expand_fw,
            step_fj,
            step_collect,
            step_cancel_w,
            step_solve,
            step_commutator,
            step_cancel_f,
        ]
    )

    u_name, v_name = u._repr_inner(), v._repr_inner()
    theorem = Theorem(
        name=f"anchor_morphism_{alg.name}_{u_name}_{v_name}",
        statement=(
            f"ρ([{u_name},{v_name}]) = [ρ({u_name}), ρ({v_name})] on "
            f"{alg.name} (every Leibniz algebroid is pre-Leibniz)"
        ),
        lhs=lhs_node,
        rhs=commutator,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"right-Leibniz ({alg.name})",
            f"Leibniz-Jacobi ({alg.name})",
        ),
        notes=(
            "Bourbaki Thm 4.1 route: J(u,v,f·w) = 0 expanded against "
            "generic f and auxiliary section w"
        ),
    )
    return chain, theorem


def _metric_coefficient_of(
    expr: Expr, w: Expr, alg: Algebroid
) -> Tuple[Expr, ...]:
    """Split ``expr = Σ ±(cᵢ·g(xᵢ, w))`` into the signed section
    contributions ``±(cᵢ·xᵢ)`` (bilinearity read backwards:
    ``Σ ±cᵢ·g(xᵢ,w) = g(Σ ±cᵢ·xᵢ, w)``).

    Raises :class:`ProofFailure` when a term carries no metric factor
    evaluated against the generic section ``w`` — then the
    non-degeneracy step would be unsound.
    """
    from jacopy.central.algebroid.context import EMetric

    def _split_term(core: Expr):
        factors = (
            list(core.children) if isinstance(core, Product) else [core]
        )
        hits = [
            (k, c)
            for k, c in enumerate(factors)
            if isinstance(c, EMetric)
            and c.algebroid_name == alg.name
            and (c.u == w or c.v == w)
        ]
        if len(hits) != 1:
            return None
        k, gnode = hits[0]
        other = gnode.u if gnode.v == w else gnode.v
        cofactors = [c for m, c in enumerate(factors) if m != k]
        if not cofactors:
            return other
        return Product(*cofactors, other)

    contributions: List[Expr] = []
    for term in _terms(expr):
        sign, core = _strip_sign(term)
        piece = _split_term(core)
        if piece is None:
            raise ProofFailure(
                "metric-family derivation: residual term "
                f"{term._repr_inner()} is not a metric evaluation "
                f"against the generic section {w._repr_inner()}"
            )
        contributions.append(piece if sign > 0 else Neg(piece))
    return tuple(contributions)


def prove_right_leibniz_from_metric(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    w: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[B 4.11, first identity] Prove
    ``[u, f·v] = ρ(u)(f)·v + f·[u,v]`` from metric invariance (C1)
    and the E-metric's definitional non-degeneracy.

    The argument, mechanized: for a generic auxiliary section ``w``,

    1. apply C1 at the composite site:
       ``ρ(u)(g(f·v, w)) = g([u, f·v], w) + g(f·v, [u, w])``;
       normalize the right-hand side (leg A),
    2. normalize the same start bilinearity-first (leg B): the scalar
       pulls out, the product rule splits, and C1 fires on
       ``ρ(u)(g(v, w))``,
    3. subtract: the difference is ``g(X, w)`` with
       ``X = [u, f·v] − ρ(u)(f)·v − f·[u,v]``,
    4. ``g(X, w) = 0`` with ``w`` generic ⟹ ``X = 0``
       (non-degeneracy — definitional for an E-metric),
    5. solve for ``[u, f·v]`` and recognize the right-Leibniz form.

    Returns the chain and the citable :class:`Theorem`.
    """
    from jacopy.central.algebroid.declarations import (
        MetricInvarianceDeclaration,
    )

    if not isinstance(alg, Algebroid):
        raise TypeError(
            "prove_right_leibniz_from_metric expects an Algebroid"
        )
    if alg.is_tangent:
        raise ValueError(
            "on the tangent algebroid the right-Leibniz rule is the "
            "Lie bracket's definitional property, not a metric theorem"
        )
    if alg.declares("right-leibniz"):
        raise ValueError(
            "right-leibniz is DECLARED on this algebroid — there it is "
            "an axiom, not a theorem; derive it on a metric context "
            "that does not declare it"
        )
    if not alg.declares("metric-invariance"):
        raise ProofFailure(
            "the right-Leibniz rule is not derivable here: the proof "
            f"needs the 'metric-invariance' declaration which {alg!r} "
            "does not make"
        )

    if w is None:
        w = _fresh_section(alg, u, v, f)
    engine = algebroid_engine(alg, registry=registry)

    start = Act(alg.anchor(u), alg.metric(Product(f, v), w))

    # Leg A: C1 applied manually at the composite site, then normalize.
    c1 = MetricInvarianceDeclaration(alg)
    if not c1.matches(start):
        raise ProofFailure(  # pragma: no cover - shape is fixed above
            "metric-family derivation: C1 does not apply to "
            f"{start._repr_inner()}"
        )
    after_c1 = c1.rewrite(start)
    step_c1 = ProofStep(
        start,
        after_c1,
        rule=c1.name,
        justification="declared axiom applied at the composite site",
        provenance_tag="axiom",
    )
    n_a, steps_a = _normalize(after_c1, engine, registry)
    step_leg_a = ProofStep(
        after_c1,
        n_a,
        rule="normalize leg A (bilinearity, canonical form)",
        justification="engine normal form of the C1 expansion",
    )
    for s in steps_a:
        step_leg_a.add_child(s)

    # Leg B: same start, bilinearity-first (the engine's own order).
    n_b, steps_b = _normalize(start, engine, registry)
    step_leg_b = ProofStep(
        start,
        n_b,
        rule="normalize leg B (scalar pull-out, product rule, C1)",
        justification=(
            "engine normal form of the same expression; C1 fires on "
            "ρ(u)(g(v,w))"
        ),
    )
    for s in steps_b:
        step_leg_b.add_child(s)

    # Legs A and B normalize the SAME start: their difference is zero.
    diff, diff_steps = _normalize(Sum(n_a, Neg(n_b)), engine, registry)
    step_diff = ProofStep(
        Sum(n_a, Neg(n_b)),
        diff,
        rule="collect: leg A − leg B = g(X, w)",
        justification=(
            "common terms cancel; bilinearity groups the rest against "
            "the auxiliary section w"
        ),
    )
    for s in diff_steps:
        step_diff.add_child(s)

    contributions = _metric_coefficient_of(diff, w, alg)
    X = simplify(
        contributions[0]
        if len(contributions) == 1
        else Sum(*contributions),
        registry,
    )

    step_nondeg = ProofStep(
        diff,
        Integer(0),
        rule="non-degeneracy of g (agreement on generic section w)",
        justification=(
            "g(X, w) = 0 for the generic section w and the E-metric "
            "is non-degenerate by definition, hence X = 0"
        ),
    )

    # Solve X = 0 for the [u, f·v] term.
    target = alg.bracket(u, Product(f, v))
    solved: Optional[Expr] = None
    x_terms = list(_terms(X))
    for i, term in enumerate(x_terms):
        sign, core = _strip_sign(term)
        if core == target:
            rest = x_terms[:i] + x_terms[i + 1 :]
            rest_sum: Expr = (
                Integer(0)
                if not rest
                else (rest[0] if len(rest) == 1 else Sum(*rest))
            )
            solved = simplify(
                Neg(rest_sum) if sign > 0 else rest_sum, registry
            )
            break
    if solved is None:
        raise ProofFailure(
            "metric-family derivation: the section identity "
            f"{X._repr_inner()} = 0 does not contain the expected "
            f"term {target._repr_inner()}"
        )
    step_solve = ProofStep(
        target,
        solved,
        rule="solve X = 0 for [u, f·v]",
        justification="rearrange the vanishing section identity",
    )

    # Recognize the right-Leibniz form mechanically.
    rhs = Sum(Product(Act(alg.anchor(u), f), v), Product(f, alg.bracket(u, v)))
    recog = ExpandAndSimplify().prove(
        solved, rhs, registry=registry, engine=algebroid_engine(alg, registry=registry)
    )
    step_recog = ProofStep(
        target,
        rhs,
        rule="right-Leibniz form recognized",
        justification="canonical-form comparison of the solved side",
    )
    for s in recog:
        step_recog.add_child(s)

    chain = ProofChain(
        [
            step_c1,
            step_leg_a,
            step_leg_b,
            step_diff,
            step_nondeg,
            step_solve,
            step_recog,
        ]
    )
    u_name, v_name, f_name = (
        u._repr_inner(),
        v._repr_inner(),
        f._repr_inner(),
    )
    theorem = Theorem(
        name=f"right_leibniz_{alg.name}_{u_name}_{v_name}_{f_name}",
        statement=(
            f"[{u_name}, {f_name}·{v_name}] = "
            f"ρ({u_name})({f_name})·{v_name} + "
            f"{f_name}·[{u_name},{v_name}] on {alg.name} "
            "(every metric algebroid is almost-Courant)"
        ),
        lhs=target,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"metric invariance ({alg.name})",
            "non-degeneracy of the E-metric (definitional)",
        ),
        notes=(
            "B 4.11 first identity: ρ(u)(g(f·v, w)) expanded two ways "
            "against a generic auxiliary section w"
        ),
    )
    return chain, theorem


def prove_left_leibniz_from_metric(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    w: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[B 4.11, second identity] Prove
    ``[f·u, v] = f·[u,v] − ρ(v)(f)·u + g(u,v)·g⁻¹(Df)`` on a metric
    algebroid — the left-Leibniz rule with the CONCRETE locality term
    ``L(Df, u, v) = g(u,v)·g⁻¹(Df)``.

    Route: (1) the symmetric part (C2) in equation form,
    ``[f·u, v] = −[v, f·u] + g⁻¹D g(f·u, v)`` (closes mechanically);
    (2) the first [B 4.11] identity for ``[v, f·u]`` (derived and
    cited); (3) the final identity closes with both cited — the
    ``f``-cofactor bracket pair collects by C2 and the coboundary
    Leibniz splits ``g⁻¹D(f·g(u,v))``.
    """
    if not isinstance(alg, Algebroid):
        raise TypeError(
            "prove_left_leibniz_from_metric expects an Algebroid"
        )
    if alg.is_tangent:
        raise ValueError(
            "on the tangent algebroid the left-Leibniz rule is "
            "definitional (antisymmetry + right-Leibniz), not a "
            "metric theorem"
        )
    if alg.declares("left-leibniz"):
        raise ValueError(
            "left-leibniz is DECLARED on this algebroid — there it is "
            "an axiom, not a theorem; derive it on a metric context "
            "that does not declare it"
        )
    if not alg.declares("symmetric-part"):
        raise ProofFailure(
            "the left-Leibniz identity is not derivable here: the "
            "proof needs the 'symmetric-part' declaration which "
            f"{alg!r} does not make"
        )

    fu = Product(f, u)

    # (1) C2 in equation (swap) form — mechanical.
    c2_rhs = Sum(
        Neg(alg.bracket(v, fu)),
        alg.sharp(alg.D(alg.metric(fu, v))),
    )
    chain_c2 = ExpandAndSimplify().prove(
        alg.bracket(fu, v),
        c2_rhs,
        registry=registry,
        engine=algebroid_engine(alg, registry=registry),
    )
    t_c2 = Theorem(
        name=f"symmetric_part_swap_{alg.name}_{f._repr_inner()}"
        f"_{u._repr_inner()}_{v._repr_inner()}",
        statement="[f·u, v] = −[v, f·u] + g⁻¹D g(f·u, v)",
        lhs=alg.bracket(fu, v),
        rhs=c2_rhs,
        proof=chain_c2,
        generality="generic-function",
        from_axioms=(f"symmetric part ({alg.name})",),
    )
    step_c2 = ProofStep(
        t_c2.lhs,
        t_c2.rhs,
        rule="symmetric part (C2), equation form",
        justification="collection of the transposed bracket pair",
    )
    for s in chain_c2:
        step_c2.add_child(s)

    # (2) the first identity for [v, f·u] — derived, then cited.
    chain_rl, t_rl = prove_right_leibniz_from_metric(
        alg, v, u, f, w=w, registry=registry
    )
    step_rl = ProofStep(
        t_rl.lhs,
        t_rl.rhs,
        rule=f"cite theorem: {t_rl.statement}",
        justification="B 4.11 first identity, derived above",
        provenance_tag="theorem",
    )
    for s in chain_rl:
        step_rl.add_child(s)

    # (3) close the final identity with both results cited.
    book = TheoremBook()
    book.add(t_c2)
    book.add(t_rl)
    engine = algebroid_engine(alg, registry=registry)
    cite(engine, book, t_c2.name, t_rl.name)
    lhs = alg.bracket(fu, v)
    rhs = Sum(
        Product(f, alg.bracket(u, v)),
        Neg(Product(Act(alg.anchor(v), f), u)),
        Product(alg.metric(u, v), alg.sharp(alg.D(f))),
    )
    chain_final = ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=engine
    )
    step_final = ProofStep(
        lhs,
        rhs,
        rule="left-Leibniz with concrete locality term",
        justification=(
            "C2 + cited right-Leibniz; the f-cofactor pair collects "
            "and g⁻¹D(f·g(u,v)) splits by the coboundary Leibniz — "
            "exhibiting L(Df,u,v) = g(u,v)·g⁻¹(Df)"
        ),
    )
    for s in chain_final:
        step_final.add_child(s)

    chain = ProofChain([step_c2, step_rl, step_final])
    u_name, v_name, f_name = (
        u._repr_inner(),
        v._repr_inner(),
        f._repr_inner(),
    )
    theorem = Theorem(
        name=f"left_leibniz_{alg.name}_{f_name}_{u_name}_{v_name}",
        statement=(
            f"[{f_name}·{u_name}, {v_name}] = "
            f"{f_name}·[{u_name},{v_name}] − "
            f"ρ({v_name})({f_name})·{u_name} + "
            f"g({u_name},{v_name})·g⁻¹(D{f_name}) on {alg.name} "
            "(metric locality term is concrete: "
            "L(Df,u,v) = g(u,v)·g⁻¹(Df))"
        ),
        lhs=lhs,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"symmetric part ({alg.name})",
            f"metric invariance ({alg.name})",
            "non-degeneracy of the E-metric (definitional)",
        ),
        notes="B 4.11 second identity",
    )
    return chain, theorem


def prove_locality_anchor_annihilation(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    h: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[MC §3] Prove ``ρ(L(Df, u, v)) = 0`` on a local pre-Leibniz
    algebroid — the locality values at coboundary forms lie in the
    anchor's kernel (the consistency fact behind the locality
    projector's ``Df`` case).

    Mechanization: expand ``ρ([f·u, v])`` acting on a generic function
    ``h`` two ways —

    1. anchor morphism applied at the composite site:
       ``ρ([f·u, v]) = [ρ(f·u), ρ(v)]``, then the Lie-bracket
       commutator on ``h`` (leg A),
    2. the declared left-Leibniz first (the engine's own order), then
       anchor linearity — leaving the ``ρ(L(Df,u,v))(h)`` term
       (leg B),

    subtract, and cancel the generic ``h`` (agreement on generators).
    """
    from jacopy.central.algebroid.declarations import (
        AnchorMorphismDeclaration,
    )
    from jacopy.central.algebroid.context import LocalityOperator
    from jacopy.central.tangent.lie_bracket import (
        LieBracketActionDefinition,
    )

    if not isinstance(alg, Algebroid):
        raise TypeError(
            "prove_locality_anchor_annihilation expects an Algebroid"
        )
    if alg.is_tangent:
        raise ValueError(
            "the tangent algebroid has no locality operator"
        )
    missing = [
        d
        for d in ("left-leibniz", "anchor-morphism")
        if not alg.declares(d)
    ]
    if missing:
        raise ProofFailure(
            "ρ(L(Df,u,v)) = 0 is not derivable here: the proof needs "
            f"the declarations {missing} which {alg!r} does not make"
        )

    if h is None:
        from jacopy.central.objects import functions

        (h,) = functions("h", registry=registry)

    def _engine():
        eng = algebroid_engine(alg, registry=registry)
        eng.register(LieBracketActionDefinition())
        return eng

    start = alg.anchor(alg.bracket(Product(f, u), v))

    # Leg A: anchor morphism applied manually at the composite site.
    am = AnchorMorphismDeclaration(alg)
    if not am.matches(start):
        raise ProofFailure(  # pragma: no cover - guarded above
            "locality-annihilation derivation: the anchor-morphism "
            f"rule does not apply to {start._repr_inner()}"
        )
    after_am = am.rewrite(start)
    step_am = ProofStep(
        start,
        after_am,
        rule=am.name,
        justification="declared axiom applied at the composite site",
        provenance_tag="axiom",
    )
    n_a, steps_a = _normalize(Act(after_am, h), _engine(), registry)
    step_leg_a = ProofStep(
        Act(after_am, h),
        n_a,
        rule="normalize leg A (anchor linearity, Lie commutator)",
        justification=(
            "engine normal form of [ρ(f·u), ρ(v)] acting on h"
        ),
    )
    for s in steps_a:
        step_leg_a.add_child(s)

    # Leg B: left-Leibniz first (the engine's own order).
    n_b, steps_b = _normalize(Act(start, h), _engine(), registry)
    step_leg_b = ProofStep(
        Act(start, h),
        n_b,
        rule="normalize leg B (left-Leibniz, anchor linearity)",
        justification=(
            "engine normal form of ρ([f·u, v]) acting on h; the "
            "locality term survives as ρ(L(Df,u,v))(h)"
        ),
    )
    for s in steps_b:
        step_leg_b.add_child(s)

    diff, diff_steps = _normalize(
        Sum(n_a, Neg(n_b)), _engine(), registry
    )
    step_diff = ProofStep(
        Sum(n_a, Neg(n_b)),
        diff,
        rule="collect: leg A − leg B",
        justification="common derivative terms cancel",
    )
    for s in diff_steps:
        step_diff.add_child(s)

    # The difference must be exactly ∓ρ(L(Df,u,v))(h).
    lhs_node = alg.anchor(
        LocalityOperator(alg.name, alg.D(f), u, v)
    )
    expected = Act(lhs_node, h)
    sign, core = _strip_sign(diff)
    if core != expected:
        raise ProofFailure(
            "locality-annihilation derivation: the two legs differ by "
            f"{diff._repr_inner()}, not by the locality term "
            f"{expected._repr_inner()}"
        )
    step_isolate = ProofStep(
        expected,
        Integer(0),
        rule="both legs expand the same expression",
        justification=(
            "leg A − leg B = 0, and the difference is exactly "
            "±ρ(L(Df,u,v))(h)"
        ),
    )
    step_cancel = ProofStep(
        lhs_node,
        Integer(0),
        rule="agreement on generators (h)",
        justification=(
            "ρ(L(Df,u,v))(h) = 0 for a generic function h, and a "
            "vector field vanishing on all functions is zero"
        ),
    )

    chain = ProofChain(
        [
            step_am,
            step_leg_a,
            step_leg_b,
            step_diff,
            step_isolate,
            step_cancel,
        ]
    )
    u_name, v_name, f_name = (
        u._repr_inner(),
        v._repr_inner(),
        f._repr_inner(),
    )
    theorem = Theorem(
        name=(
            f"locality_anchor_annihilation_{alg.name}_{f_name}"
            f"_{u_name}_{v_name}"
        ),
        statement=(
            f"ρ(L(D{f_name},{u_name},{v_name})) = 0 on {alg.name} "
            "(locality values at coboundary forms lie in ker ρ)"
        ),
        lhs=lhs_node,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"left-Leibniz ({alg.name})",
            f"anchor morphism ({alg.name})",
        ),
        notes="MC §3: (ρ ∘ L)(Df, u, v) = 0 on local pre-Leibniz",
    )
    return chain, theorem


def prove_anchor_predator_vanishes(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    w: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``P_ρ(u, v) = 0`` on a Leibniz algebroid — the defect-operator
    face of the flagship theorem.

    Derives the anchor-morphism theorem for ``(u, v)`` and cites it;
    the predator's definitional expansion then cancels. The chain's
    theorem-tagged step carries the full derivation in foundational
    mode.
    """
    from jacopy.core.expr import Integer as _Int

    from jacopy.central.algebroid.operators import anchor_predator

    _, theorem = prove_anchor_morphism(
        alg, u, v, f, w=w, registry=registry
    )
    book = TheoremBook()
    book.add(theorem)
    engine = algebroid_engine(alg, registry=registry)
    cite(engine, book, theorem.name)
    return ExpandAndSimplify().prove(
        anchor_predator(alg, u, v),
        _Int(0),
        registry=registry,
        engine=engine,
    )
