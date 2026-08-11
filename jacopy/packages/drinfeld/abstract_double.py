"""
The ABSTRACT main theorem of the double (Phase 6.I.4) — the 6.C
program item "property subset + compatibility ⟹ the double satisfies
the same subset", realized as a FINITE DECLARATION SWEEP on the
Phase 3 abstract-algebroid layer (the engine proves instances, not
schemas; the sweep over declared subsets is the mechanical face of
the schema).

Construction: for an abstract algebroid ``A`` (Phase 3 declaration
context) with its OWN Cartan calculus ``(d_E, ℒ^E, ι)`` (Phase 3.F),
the abstract STANDARD double on ``A ⊕ Λᵖ A*`` is

    [u+α, v+β] := [u,v]_A ⊕ ( ℒ^E_u β − ℒ^E_v α + d_E ι_v α ).

Sweep results (each an ⟹ with an honest-fail partner):

* **right-Leibniz**: the double is right-Leibniz with anchor
  ``ρ(u+α) = u``  ⟺  ``A`` declares ``right-leibniz`` +
  ``left-leibniz`` + ``local`` — the paper's "LOCAL almost-Leibniz"
  hypothesis rediscovered mechanically, term by term: right-Leibniz
  alone leaves the left-Leibniz defect; adding left-Leibniz leaves
  EXACTLY ``⟨α, L(df,v,w)⟩`` — the locality operator — as the
  one-term residual, so ``local`` (L = 0) is forced.
* **symmetric part**: form component ``= d_E⟨e₁,e₂⟩₊`` needs NO
  declaration (the 3.F magic theorem); the vector component
  ``[u,v]+[v,u] = 0`` is exactly the ``antisymmetric`` declaration.
* **Leibniz-Jacobi (form)**: closes under ``jacobi`` +
  ``anchor-morphism`` (+ right-leibniz), with bracket-level Jacobi
  entering as cited :func:`jacobi_combination_theorem` instances —
  and honest-fails when ``jacobi`` is withheld.

Together with the three CONCRETE instantiations already in the suite
(standard TM double, Poisson/LWX, Nambu) this closes the abstract
reading of the main theorem: the double inherits exactly the
declared subset, mechanically.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.proof.theorems import Theorem, TheoremDefinition
from jacopy.central.objects.interior import Interior
from jacopy.central.algebroid.calculus import (
    _engine,
    d_E,
    jacobi_combination_theorem,
    lie_E,
)
from jacopy.central.algebroid.context import Algebroid


def _iota(u: Expr, x: Expr) -> Expr:
    return Act(Interior(u), x)


def _double_engine(alg: Algebroid, registry):
    """The 3.F calculus engine LAYERED with the Phase 3 declaration
    engine — the calculus rules alone never see the declared axioms
    (anchor-morphism, Leibniz, locality), which is exactly what the
    sweep is about."""
    from jacopy.central.algebroid.engine import algebroid_engine

    eng = _engine(alg, registry)
    for d in algebroid_engine(alg, registry=registry).definitions:
        eng.register(d)
    return eng


def abstract_double(
    alg: Algebroid, u: Expr, alpha: Expr, v: Expr, beta: Expr
) -> Tuple[Expr, Expr]:
    """The abstract standard double bracket on ``A ⊕ Λᵖ A*`` as a
    ``(section, form)`` pair."""
    return (
        alg.bracket(u, v),
        Sum(
            Act(lie_E(alg, u), beta),
            Neg(Act(lie_E(alg, v), alpha)),
            d_E(alg, _iota(v, alpha)),
        ),
    )


def abstract_pairing(
    u: Expr, alpha: Expr, v: Expr, beta: Expr
) -> Expr:
    """``⟨e₁,e₂⟩₊ = ι_u β + ι_v α``."""
    return Sum(_iota(u, beta), _iota(v, alpha))


def prove_abstract_double_right_leibniz(
    alg: Algebroid,
    u: Expr,
    alpha: Expr,
    v: Expr,
    beta: Expr,
    f: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, ProofChain]:
    """The double's right-Leibniz law with anchor ``ρ(u+α) = u``,
    as an EXACT DEFECT FORMULA (the 6.G pattern): under the "local"
    level (right- + left-Leibniz) the form-component defect is
    precisely the LOCALITY term,

        form([e₁, f·e₂]) − f·form([e₁,e₂]) − (ρ(e₁)f)·β
            ⟨·, w⟩ = ⟨α, L(df, v, w)⟩,

    (surfaced mechanically: the bare run leaves the left-Leibniz
    defect; the local run leaves exactly this one term). The vector
    component is plain right-Leibniz. Corollary: the double is
    right-Leibniz ⟺ the locality operator vanishes — the paper's
    (5.8)/(3.17) decomposition in instance form."""
    vec_l, form_l = abstract_double(
        alg, u, alpha, Product(f, v), Product(f, beta)
    )
    vec_b, form_b = abstract_double(alg, u, alpha, v, beta)
    anchor_f = Act(alg.anchor(u), f)
    eng = _double_engine(alg, registry)
    c_vec = ExpandAndSimplify().prove(
        Act(
            Sum(
                vec_l,
                Neg(
                    Sum(
                        Product(f, vec_b),
                        Product(anchor_f, v),
                    )
                ),
            ),
            f,
        ),
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )
    from jacopy.central.algebroid.context import locality_term

    c_form = ExpandAndSimplify().prove(
        Pairing(
            Sum(
                form_l,
                Neg(
                    Sum(
                        Product(f, form_b),
                        Product(anchor_f, beta),
                    )
                ),
            ),
            w,
        ),
        Pairing(alpha, Act(locality_term(alg, f, v, w), f))
        if False
        else Pairing(
            alpha, locality_term(alg, f, v, w)
        ),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )
    return c_vec, c_form


def prove_abstract_double_symmetric_part(
    alg: Algebroid,
    u: Expr,
    alpha: Expr,
    v: Expr,
    beta: Expr,
    f: Expr,
    w: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, ProofChain]:
    """Symmetric part ``= (0, d_E⟨e₁,e₂⟩₊)``: the form component is
    DECLARATION-FREE (the 3.F magic theorem); the vector component is
    exactly the ``antisymmetric`` declaration (honest-fail
    without)."""
    v12, f12 = abstract_double(alg, u, alpha, v, beta)
    v21, f21 = abstract_double(alg, v, beta, u, alpha)
    eng = _double_engine(alg, registry)
    c_vec = ExpandAndSimplify().prove(
        Act(Sum(v12, v21), f),
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )
    c_form = ExpandAndSimplify().prove(
        Pairing(
            Sum(
                f12,
                f21,
                Neg(
                    d_E(
                        alg,
                        abstract_pairing(u, alpha, v, beta),
                    )
                ),
            ),
            w,
        ),
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )
    return c_vec, c_form


def prove_abstract_double_jacobi_form(
    alg: Algebroid,
    u: Expr,
    alpha: Expr,
    v: Expr,
    beta: Expr,
    w: Expr,
    gamma: Expr,
    probe: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 60000,
) -> Tuple[ProofChain, List[Theorem]]:
    """Form component of the double's Leibniz-Jacobi — the abstract
    6.C theorem: closes under ``jacobi`` + ``anchor-morphism`` (+
    ``right-leibniz``), with the bracket-level Jacobi entering as
    cited :func:`jacobi_combination_theorem` instances. Honest-fail
    when ``jacobi`` is withheld. Evaluated against the probe section
    ``probe``."""

    def nest(u1, a1, u2, a2, u3, a3):
        iv, if_ = abstract_double(alg, u2, a2, u3, a3)
        return abstract_double(alg, u1, a1, iv, if_)[1]

    v12, f12 = abstract_double(alg, u, alpha, v, beta)
    lhs = Sum(
        nest(u, alpha, v, beta, w, gamma),
        Neg(abstract_double(alg, v12, f12, w, gamma)[1]),
        Neg(nest(v, beta, u, alpha, w, gamma)),
    )
    eng = _double_engine(alg, registry)
    used: List[Theorem] = []
    if alg.declares("jacobi"):
        from itertools import combinations

        from jacopy.packages.poisson.tilde import (
            _normalized_by,
        )

        # Stage 1: normalize ALL ± lhs's in the citing engine BEFORE
        # registering any (6.I.4 lesson: registering the + variant
        # first makes the − seed normalize to 0 and get skipped,
        # while the slot residual carries exactly the − shape).
        staged = []
        for trip in combinations((u, v, w, probe), 3):
            thm = jacobi_combination_theorem(
                alg, *trip, registry=registry
            )
            for sgn, tag in (
                ((lambda x: x), ""),
                (Neg, "_neg"),
            ):
                lhs_nf = _normalized_by(
                    eng, sgn(thm.lhs), registry
                )
                if lhs_nf == Integer(0):
                    continue
                staged.append((thm, tag, lhs_nf))
        # Stage 2: register.
        for thm, tag, lhs_nf in staged:
            variant = Theorem(
                name=thm.name + tag,
                statement=thm.statement,
                lhs=lhs_nf,
                rhs=Integer(0),
                proof=thm.proof,
                generality="instance",
            )
            eng.register(TheoremDefinition(variant))
            used.append(variant)
    chain = ExpandAndSimplify().prove(
        Pairing(lhs, probe),
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )
    return chain, used
