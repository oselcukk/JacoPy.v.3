"""
The ALTERNATIVE (Dorfman/Roytenberg) Courant axiomatics and its
redundancies (Phase 7.A.2) [uchino.pdf Rem 2.3].

The non-skew form: a bilinear ``∘`` on ``ΓE`` with [C'1] (left
Leibniz-Jacobi), [C'2] (anchor morphism), [C'3] (right Leibniz with
NO defect), [C'4] (``x∘x = D(x,x)``), [C'5] (invariance with NO
D-terms). Uchino's closing remark — *"[C'2], [C'3] and the D-formula
are also derived from the rest of the axioms and condition (L) in a
similar way"* — is stated WITHOUT proof in the paper. This module
writes those proofs mechanically:

* **[C'3]** — since [C'5] carries no D-terms it IS the Phase 3
  metric-invariance axiom C1 [B 4.10], and the derivation is
  literally the [B 4.11] tactic
  (:func:`~jacopy.central.algebroid.theorems.prove_right_leibniz_from_metric`):
  every metric algebroid is almost-Courant. The wrapper below only
  adds the Uchino framing.
* **[C'2]** — from [C'1] + the proven [C'3]: normalize the cited
  Jacobi instance ``x∘(y∘(f·z)) − (x∘y)∘(f·z) − y∘(x∘(f·z))`` with
  the [C'3] theorems as rewrites; the bracket-shaped remainder is
  ``f`` times a plain [C'1] instance (cited), and the rest is
  proportional to the generic ``z`` — agreement on generators gives
  ``ρ(x∘y)(f) = ρ(x)ρ(y)(f) − ρ(y)ρ(x)(f) = [ρ(x),ρ(y)](f)``.

Honest boundary: the Dorfman-form D-FORMULA needs [C'4] polarized
over sums, i.e. the ℝ-additivity of ``D`` — an assumption (L) alone
does not provide. It is left as the documented open slice rather
than smuggled in.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem, TheoremDefinition
from jacopy.central.algebroid.context import Algebroid, algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.theorems import (
    _coefficient_of_generic_section,
    _normalize,
    _strip_sign,
    _terms,
    prove_right_leibniz_from_metric,
)
from jacopy.algorithms.simplify import simplify


def dorfman_axiom_context(name: str = "E") -> Algebroid:
    """The alternative-definition DATA: non-skew opaque ``∘``
    (ℝ-bilinear by definition), anchor, definitional pairing, and
    the declared [C'5] — which, having no D-terms, is EXACTLY the
    Phase 3 ``metric-invariance`` axiom. [C'2]/[C'3] are absent:
    they are the theorems."""
    return algebroid(name, declare=("metric-invariance",))


def prove_dorfman_c3_redundant(
    alg: Algebroid,
    x: Expr,
    y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'3] is redundant: ``x∘(f·y) = f·(x∘y) + ρ(x)(f)·y`` from
    [C'5] + non-degeneracy — the defect-free Leibniz rule, because
    [C'5] carries no D-terms. Mechanically this IS the [B 4.11]
    two-leg tactic; Uchino's "in a similar way", already in the
    Phase 3 library."""
    chain, thm = prove_right_leibniz_from_metric(
        alg, x, y, f, registry=registry
    )
    theorem = Theorem(
        name=f"uchino_cprime3_{alg.name}_"
        f"{x._repr_inner()}_{y._repr_inner()}_{f._repr_inner()}",
        statement=(
            f"x∘(f·y) = f·(x∘y) + ρ(x)(f)·y on {alg.name} "
            "(the alternative-definition axiom [C'3] is redundant "
            "— Uchino Rem 2.3, mechanized)"
        ),
        lhs=thm.lhs,
        rhs=thm.rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"[C'5] invariance ({alg.name}) — no D-terms",
            "non-degeneracy of the pairing (definitional)",
        ),
        notes=(
            "uchino.pdf Rem 2.3 closing remark; the derivation is "
            "the B 4.11 tactic verbatim (every metric algebroid is "
            "almost-Courant)"
        ),
    )
    return chain, theorem


def _c3_engine(
    alg: Algebroid,
    theorems,
    registry: Optional[PropertyRegistry],
) -> ExpansionEngine:
    eng = ExpansionEngine(
        [TheoremDefinition(t) for t in theorems]
    )
    for d in algebroid_engine(alg, registry=registry).definitions:
        eng.register(d)
    return eng


def prove_dorfman_c2_redundant(
    alg: Algebroid,
    x: Expr,
    y: Expr,
    z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'2] is redundant: ``ρ(x∘y)(f) = [ρ(x), ρ(y)](f)`` from
    [C'1] + the proven [C'3] + agreement on generators (the
    Grabowski-Marmo pattern for the non-skew form; Uchino Rem 2.2
    cites their Lie-algebroid result — here the engine writes the
    Courant-side derivation):

    1. cite the [C'1] instance
       ``x∘(y∘(f·z)) = (x∘y)∘(f·z) + y∘(x∘(f·z))`` (declared
       axiom of the alternative definition),
    2. normalize its defect with the [C'3] THEOREMS as rewrites —
       the scalar peels off every slot,
    3. the bracket-shaped remainder is ``f`` times the PLAIN [C'1]
       instance — cite it,
    4. what survives is proportional to the generic ``z``:
       agreement on generators leaves the scalar identity, and it
       is ``ρ(x∘y)(f) − ρ(x)ρ(y)(f) + ρ(y)ρ(x)(f) = 0``.
    """
    br = alg.bracket
    if alg.declares("anchor-morphism"):
        raise ValueError(
            "anchor-morphism is DECLARED on this context — [C'2] "
            "would be an axiom, not a theorem; use a bare "
            "dorfman_axiom_context()"
        )

    fz = Product(f, z)
    jac_f = Sum(
        br(x, br(y, fz)),
        Neg(br(br(x, y), fz)),
        Neg(br(y, br(x, fz))),
    )

    # SELF-EXTENDING [C'3] instance set: normalize, find the still
    # unopened brackets-with-scalar-slot, prove the [C'3] instance
    # each needs, register, repeat to fixpoint. Every added rewrite
    # is itself a proven theorem — the loop is a proof search, not
    # an assumption.
    from jacopy.central.calculus.scalars import (
        is_scalar_function,
    )
    from jacopy.central.algebroid.context import (
        AlgebroidBracket,
    )

    def _unopened(expr: Expr, found: List) -> None:
        if (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == alg.name
            and isinstance(expr.v, Product)
            and len(expr.v.children) >= 2
        ):
            for k, c in enumerate(expr.v.children):
                if is_scalar_function(c, registry):
                    rest = [
                        r
                        for m, r in enumerate(expr.v.children)
                        if m != k
                    ]
                    found.append(
                        (
                            expr.u,
                            rest[0]
                            if len(rest) == 1
                            else Product(*rest),
                            c,
                        )
                    )
                    break
        slots = getattr(expr, "rewritable_slots", None)
        if expr.is_atom:
            for sl in slots or ():
                _unopened(sl, found)
            return
        for c in expr.children:
            _unopened(c, found)

    c3_thms: List[Theorem] = []
    seen_instances = set()
    n1 = jac_f
    for _round in range(6):
        engine = _c3_engine(alg, c3_thms, registry)
        n1, steps1 = _normalize(jac_f, engine, registry)
        found: List = []
        _unopened(n1, found)
        fresh = [
            (u, v, h)
            for (u, v, h) in found
            if (
                u._repr_inner(),
                v._repr_inner(),
                h._repr_inner(),
            )
            not in seen_instances
        ]
        if not fresh:
            break
        for (u, v, h) in fresh:
            seen_instances.add(
                (
                    u._repr_inner(),
                    v._repr_inner(),
                    h._repr_inner(),
                )
            )
            _, t = prove_dorfman_c3_redundant(
                alg, u, v, h, registry=registry
            )
            c3_thms.append(t)
    else:
        raise ProofFailure(
            "Uchino [C'2]: the [C'3] instance search did not reach "
            "a fixpoint"
        )
    engine = _c3_engine(alg, c3_thms, registry)
    step_cite = ProofStep(
        jac_f,
        jac_f,
        rule="[C'1] instance cited: x∘(y∘(f·z)) − (x∘y)∘(f·z) "
        "− y∘(x∘(f·z)) = 0",
        justification=(
            "declared axiom of the alternative definition, at the "
            "composite argument f·z"
        ),
        provenance_tag="axiom",
    )

    n1, steps1 = _normalize(jac_f, engine, registry)
    step_norm = ProofStep(
        jac_f,
        n1,
        rule="normalize with the proven [C'3] theorems "
        "(scalar peels off every slot)",
        justification="engine normal form; each peel cites [C'3]",
    )
    for s in steps1:
        step_norm.add_child(s)

    jac_plain = Sum(
        br(x, br(y, z)),
        Neg(br(br(x, y), z)),
        Neg(br(y, br(x, z))),
    )
    after_plain, steps2 = _normalize(
        Sum(n1, Neg(Product(f, jac_plain))), engine, registry
    )
    step_plain = ProofStep(
        n1,
        after_plain,
        rule="cite the PLAIN [C'1] instance: subtract "
        "f·(x∘(y∘z) − (x∘y)∘z − y∘(x∘z)) = 0",
        justification=(
            "declared axiom on plain sections; subtracting a "
            "declared zero"
        ),
        provenance_tag="axiom",
    )
    for s in steps2:
        step_plain.add_child(s)
    if after_plain == Integer(0):
        raise ProofFailure(
            "Uchino [C'2]: the residual vanished before the "
            "generator step — nothing to extract (degenerate)"
        )

    coeffs = _coefficient_of_generic_section(after_plain, z)
    identity = simplify(
        coeffs[0] if len(coeffs) == 1 else Sum(*coeffs),
        registry,
    )
    step_generators = ProofStep(
        after_plain,
        Integer(0),
        rule="agreement on generators (generic section z)",
        justification=(
            "the residual is c·z = 0 with z generic, hence the "
            "scalar coefficient c = 0"
        ),
    )

    lhs = Act(alg.anchor(br(x, y)), f)
    rhs = Sum(
        Act(alg.anchor(x), Act(alg.anchor(y), f)),
        Neg(Act(alg.anchor(y), Act(alg.anchor(x), f))),
    )
    check_pos, _ = _normalize(
        Sum(identity, Neg(Sum(lhs, Neg(rhs)))), engine, registry
    )
    check_neg, _ = _normalize(
        Sum(identity, Sum(lhs, Neg(rhs))), engine, registry
    )
    if check_pos != Integer(0) and check_neg != Integer(0):
        raise ProofFailure(
            "Uchino [C'2]: the generator identity is not the "
            "anchor-morphism defect: "
            + identity._repr_inner()[:160]
        )
    step_solve = ProofStep(
        lhs,
        rhs,
        rule="the vanishing coefficient IS the anchor-morphism "
        "defect: ρ(x∘y)(f) = ρ(x)ρ(y)(f) − ρ(y)ρ(x)(f)",
        justification=(
            "engine-checked equality of the extracted identity "
            "with the [C'2] defect; the right side is the Lie "
            "bracket action [ρ(x),ρ(y)](f) by definition"
        ),
    )

    chain = ProofChain(
        [
            step_cite,
            step_norm,
            step_plain,
            step_generators,
            step_solve,
        ]
    )
    xn, yn, fn = (
        x._repr_inner(),
        y._repr_inner(),
        f._repr_inner(),
    )
    theorem = Theorem(
        name=f"uchino_cprime2_{alg.name}_{xn}_{yn}_{fn}",
        statement=(
            f"ρ({xn}∘{yn})({fn}) = [ρ({xn}), ρ({yn})]({fn}) on "
            f"{alg.name} (the alternative-definition axiom [C'2] "
            "is redundant — Uchino Rem 2.3, mechanized via the "
            "Grabowski-Marmo pattern)"
        ),
        lhs=lhs,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "[C'1] left Leibniz-Jacobi (two cited instances)",
            "[C'3] (itself proven from [C'5] + non-degeneracy)",
            "agreement on generators (generic section z)",
        ),
        notes=(
            "uchino.pdf Rem 2.3 closing remark, written out: the "
            "paper says 'in a similar way' without proof"
        ),
    )
    return chain, theorem
