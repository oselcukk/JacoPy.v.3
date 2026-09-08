"""
Roytenberg → Courant (Phase 7.B.2d; PDF item 14g): the general
8-component Roytenberg bracket of the Poisson case
(:func:`~jacopy.packages.drinfeld.roytenberg.roytenberg_bracket`,
eq (5.2) at p = 1) IDENTIFIED, switch by switch, with the Courant
structures whose axiom suites Phase 7.B already verified:

* all switches off      → the STANDARD Dorfman bracket
  (:mod:`standard_courant` — Courant axioms verified in 7.B.1);
* tilde + R on, under the DECLARED Poisson condition
                        → the Poisson generalized double
  (:mod:`poisson_generalized` — the derived R-twist dies, 7.B.2c,
  and what remains is exactly Watamura's ``(TM)₀ ⊕ (T*M)_θ``);
* H on (tilde/R off)    → the H-TWISTED Dorfman bracket
  (:mod:`twisted_courant` — Courant ⟺ dH = 0, 7.B.2a).

This is the PDF 14g statement "the Roytenberg bracket from the
Poisson geometry case yields a Courant algebroid", mechanized as
identification theorems: each Roytenberg configuration IS one of the
verified Courant structures, so its axioms are inherited from the
corresponding suite (the Jacobi conditions themselves are the 6.F
theorems — H-closure ⟺ dH, proven degree-generally). The symmetric
part of the FULL bracket (everything on) is also proven: the H- and
R-terms are antisymmetric, so ``D_θ`` survives every twist.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Product,
    Sum,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.packages.drinfeld.double import (
    canonical_pairing,
    dorfman_double,
)
from jacopy.packages.drinfeld.roytenberg import (
    roytenberg_bracket,
)
from jacopy.packages.drinfeld.twist import dorfman_double_h
from jacopy.packages.generalized.poisson_generalized import (
    _normalize,
    _require_poisson,
    d_operator_theta,
    theta_dorfman,
)
from jacopy.packages.generalized.r_twisted import (
    _engine as _r_engine,
)
from jacopy.packages.poisson.nambu import NambuPoissonStructure


def _zero_theorem(
    name: str,
    statement: str,
    diffs,
    engine,
    registry,
    *,
    from_axioms,
    notes: str,
    labels,
) -> Tuple[ProofChain, Theorem]:
    steps: List[ProofStep] = []
    for label, diff in zip(labels, diffs):
        nf = _normalize(engine, diff, registry)
        if nf != Integer(0):
            raise ProofFailure(
                f"{name}: {label} FAILS — residual "
                + nf._repr_inner()[:160]
            )
        steps.append(
            ProofStep(
                diff,
                Integer(0),
                rule=label,
                justification=(
                    "engine normal form of the difference"
                ),
            )
        )
    chain = ProofChain(steps)
    theorem = Theorem(
        name=name,
        statement=statement,
        lhs=diffs[0],
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=from_axioms,
        notes=notes,
    )
    return chain, theorem


def prove_roytenberg_reduces_to_dorfman(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """All switches OFF: the Roytenberg bracket is the STANDARD
    Dorfman bracket of 7.B.1 — component-wise identification (the
    𝒦 combination ``−ℒ_V ω + dι_V ω`` matches Dorfman's form
    side)."""
    _require_poisson(N)
    engine = _r_engine(N, registry, declare_fi=False)
    r_vec, r_form = roytenberg_bracket(
        N, U, omega, V, eta, with_tilde=False, with_r=False
    )
    d_vec, d_form = dorfman_double(U, omega, V, eta)
    return _zero_theorem(
        "roytenberg_reduces_to_dorfman",
        "Roytenberg(all off) = standard Dorfman on TM ⊕ T*M — "
        "the 7.B.1 Courant suite applies verbatim (PDF 14g)",
        [Sum(r_vec, Neg(d_vec)), Sum(r_form, Neg(d_form))],
        engine,
        registry,
        from_axioms=(
            "Roytenberg (5.2) + Dorfman definitions (𝒦 = −ℒ + dι)",
        ),
        notes="decomposition (5.4) at the trivial switch point",
        labels=(
            "vector component normalizes to 0",
            "form component normalizes to 0",
        ),
    )


def prove_roytenberg_is_theta_double_under_poisson(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """THE 14g statement: with tilde + R on and the DECLARED Poisson
    condition, the Roytenberg bracket IS the Poisson generalized
    double — the natural R-twist ``R′ = [θ♯·,θ♯·] − θ♯[·,·]_θ``
    dies under the fundamental identity (7.B.2c), and the remainder
    is exactly Watamura's ``(TM)₀ ⊕ (T*M)_θ`` whose Courant axioms
    Phase 7.B.2b verified. Honest-fail without the declaration: the
    difference is the surviving R′."""
    _require_poisson(N)
    engine = _r_engine(
        N, registry, declare_fi=declare_poisson
    )
    r_vec, r_form = roytenberg_bracket(
        N, U, omega, V, eta, with_tilde=True, with_r=True
    )
    t_vec, t_form = theta_dorfman(N, U, omega, V, eta)
    diffs = [
        Sum(r_vec, Neg(t_vec)),
        Sum(r_form, Neg(t_form)),
    ]
    if not declare_poisson:
        nf = _normalize(engine, diffs[0], registry)
        raise ProofFailure(
            "Roytenberg = θ-double needs the declared Poisson "
            "condition (the difference IS the derived R-twist); "
            "residual: " + nf._repr_inner()[:140]
        )
    return _zero_theorem(
        "roytenberg_is_theta_double_under_poisson",
        "Roytenberg(tilde, R) = the Poisson generalized double "
        "(TM)₀ ⊕ (T*M)_θ under the declared Poisson condition — "
        "the Poisson-case Roytenberg bracket yields the verified "
        "Courant structure (PDF 14g)",
        diffs,
        engine,
        registry,
        from_axioms=(
            "Roytenberg (5.2) + θ-double definitions",
            "declared Poisson condition (kills the derived "
            "R-twist — 7.B.2c)",
        ),
        notes="decomposition (5.5) at p = 1 / Watamura §2-3",
        labels=(
            "vector component normalizes to 0",
            "form component normalizes to 0",
        ),
    )


def prove_roytenberg_h_is_twisted_dorfman(
    N: NambuPoissonStructure,
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """H on (tilde/R off): the Roytenberg bracket is the H-TWISTED
    Dorfman bracket of 7.B.2a — Courant ⟺ dH = 0 applies."""
    _require_poisson(N)
    engine = _r_engine(N, registry, declare_fi=False)
    r_vec, r_form = roytenberg_bracket(
        N,
        U,
        omega,
        V,
        eta,
        H=H,
        with_tilde=False,
        with_r=False,
    )
    d_vec, d_form = dorfman_double_h(H, U, omega, V, eta)
    return _zero_theorem(
        "roytenberg_h_is_twisted_dorfman",
        "Roytenberg(H) = the H-twisted Dorfman bracket — the "
        "7.B.2a suite (Courant ⟺ dH = 0, Ševera) applies "
        "verbatim (PDF 14g / 14e)",
        [Sum(r_vec, Neg(d_vec)), Sum(r_form, Neg(d_form))],
        engine,
        registry,
        from_axioms=(
            "Roytenberg (5.2) + H-twisted Dorfman definitions",
        ),
        notes="decomposition (5.4) with the H-switch",
        labels=(
            "vector component normalizes to 0",
            "form component normalizes to 0",
        ),
    )


def prove_full_roytenberg_symmetric_part(
    N: NambuPoissonStructure,
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'4] of the FULL Roytenberg bracket (everything on): the H-
    and R-terms are antisymmetric, so

    ``[x,y]_Roy + [y,x]_Roy = 2·D_θ⟨x,y⟩₊``

    — the coboundary ``D_θ`` of the Poisson generalized double
    survives every twist."""
    _require_poisson(N)
    engine = _r_engine(N, registry, declare_fi=False)
    ab_vec, ab_form = roytenberg_bracket(
        N, U, omega, V, eta, H=H
    )
    ba_vec, ba_form = roytenberg_bracket(
        N, V, eta, U, omega, H=H
    )
    D_vec, D_form = d_operator_theta(
        N, canonical_pairing(U, omega, V, eta)
    )
    return _zero_theorem(
        "full_roytenberg_symmetric_part",
        "[x,y]_Roy + [y,x]_Roy = 2·D_θ⟨x,y⟩₊ for the FULL "
        "Roytenberg bracket — D_θ survives the H- and R-twists "
        "([C'4]; PDF 14g)",
        [
            Sum(
                ab_vec,
                ba_vec,
                Neg(Product(Integer(2), D_vec)),
            ),
            Sum(
                ab_form,
                ba_form,
                Neg(Product(Integer(2), D_form)),
            ),
        ],
        engine,
        registry,
        from_axioms=(
            "Roytenberg (5.2) + pairing + D_θ definitions",
            "interior/tilde-interior anticommutation "
            "(H- and R-terms antisymmetric)",
        ),
        notes="the 6.D symmetric part, twist-invariant",
        labels=(
            "vector component normalizes to 0",
            "form component normalizes to 0",
        ),
    )
