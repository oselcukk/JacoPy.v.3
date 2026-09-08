"""
Poisson generalized geometry (Phase 7.B.2b; PDF item 14f, Watamura
et al. [arXiv:1408.2649 §2-3]): the Courant algebroid
``(TM)₀ ⊕ (T*M)_θ`` over a Poisson base — the TILDE-Dorfman bracket
whose form side carries the Koszul bracket and whose vector side
carries the θ-induced tilde calculus, its skew TILDE-Courant
companion, and their axiom suite, concrete on ``TM ⊕ T*M``.

Representation: the Phase 6 Nambu machinery at ``p = 1`` — a
``NambuPoissonStructure`` of order 1 IS a Poisson bivector ``θ``,
and :func:`~jacopy.packages.drinfeld.double.nambu_double` at
``p = 1`` is exactly the triangular/LWX double of Watamura's
structure:

    vector: [U,V] + ℒ̃_ω V − ℒ̃_η U + d̃ ι̃_η U,
    form:   [ω,η]_θ + ℒ_U η − ℒ_V ω + d ι_V ω,

with total anchor ``ρ(U+ω) = U + θ♯ω`` and coboundary
``D_θ(h) = ½·(d̃h, dh) = ½·(−θ♯dh, dh)`` (Uchino Rem 1
normalization, as in :mod:`standard_courant`). Inventory:

* [C'3] right-Leibniz and [C'4] symmetric part — Phase 6.D, any
  bivector (:mod:`jacopy.packages.drinfeld.double`);
* **[C'2]** — the total anchor is a bracket morphism: THE 6.I.4
  capstone (4.19), under the declared fundamental identity (= the
  Poisson condition at ``p = 1``); re-exported here with the
  Watamura framing;
* this module adds, concrete: **[C'5]** invariance of the canonical
  pairing, the **tilde-Courant bracket** with skewness and the
  Dorfman↔Courant relation, and the ``⟨D_θh, x⟩₊ = ½ρ(x)h``
  normalization cross-check against the 7.A axiomatics.

The audit note is honored: this is the Watamura STRUCTURE itself
(θ-twisted double on a Poisson base), not a renaming of Π-twist
results — the R-flux twist of THIS structure is the separate
7.B.2c slice.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Product,
    Rational,
    Sum,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.tangent.exterior import d
from jacopy.packages.drinfeld.double import (
    canonical_pairing,
    nambu_double,
)
from jacopy.packages.drinfeld.tilde_calculus import _tilde_engine
from jacopy.packages.poisson.nambu import (
    NambuPoissonStructure,
    nambu_structure,
)


def poisson_base(name: str = "θ") -> NambuPoissonStructure:
    """The Watamura base datum: a Poisson bivector ``θ`` as the
    order-1 Nambu structure (the fundamental identity declared per
    proof = the Poisson condition ``[θ,θ]_SN = 0``)."""
    return nambu_structure(name, p=1)


def _require_poisson(N: NambuPoissonStructure) -> None:
    if N.p != 1:
        raise ValueError(
            "Poisson generalized geometry needs an order-1 "
            f"structure (a bivector); got p = {N.p}"
        )


def theta_dorfman(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
) -> Tuple[Expr, Expr]:
    """The tilde-Dorfman bracket of ``(TM)₀ ⊕ (T*M)_θ`` — the
    triangular double [Watamura §2; drinfeld eq (4.17)]."""
    _require_poisson(N)
    return nambu_double(N, U, omega, V, eta)


def theta_anchor(
    N: NambuPoissonStructure, U: Expr, omega: Expr
) -> Expr:
    """``ρ(U+ω) = U + θ♯ω`` — the total anchor [eq (4.19)]."""
    _require_poisson(N)
    return Sum(U, N.sharp_vf(omega))


def d_operator_theta(
    N: NambuPoissonStructure, h: Expr
) -> Tuple[Expr, Expr]:
    """``D_θ(h) = ½·(−θ♯dh, dh)`` — the θ-coboundary, normalized so
    ``⟨D_θh, x⟩₊ = ½·ρ(x)(h)`` (see
    :func:`prove_theta_d_pairing_value`)."""
    _require_poisson(N)
    return (
        Product(Rational(1, 2), Neg(N.sharp_vf(d(h)))),
        Product(Rational(1, 2), d(h)),
    )


def theta_courant(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
) -> Tuple[Expr, Expr]:
    """The TILDE-Courant bracket — the skew-symmetrization:
    ``[x,y]_{C,θ} = [x,y]_{D,θ} − D_θ⟨x,y⟩₊`` component-wise
    (PDF 14f.ii)."""
    vec, form = theta_dorfman(N, U, omega, V, eta)
    D_vec, D_form = d_operator_theta(
        N, canonical_pairing(U, omega, V, eta)
    )
    return Sum(vec, Neg(D_vec)), Sum(form, Neg(D_form))


def prove_theta_anchor_morphism(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
) -> ProofChain:
    """[C'2] for the Poisson generalized double:
    ``ρ([x,y]_{D,θ}) = [ρ(x), ρ(y)]_Lie`` probed on ``h`` — THE
    6.I.4 capstone (4.19) at ``p = 1``, under the declared Poisson
    condition (honest-fail without)."""
    from jacopy.packages.drinfeld.bracket_morphism import (
        prove_total_anchor_is_bracket_morphism,
    )

    _require_poisson(N)
    return prove_total_anchor_is_bracket_morphism(
        N,
        U,
        omega,
        V,
        eta,
        h,
        registry=registry,
        declare_fi=declare_poisson,
    )


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


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


def prove_theta_courant_skew(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The tilde-Courant bracket is genuinely SKEW:
    ``[x,y]_{C,θ} + [y,x]_{C,θ} = 0`` component-wise — the 6.D
    symmetric part of the tilde-Dorfman is exactly ``2·D_θ⟨x,y⟩₊``
    and the subtraction removes it. Holds for ANY bivector (no
    Poisson declaration)."""
    _require_poisson(N)
    engine = _tilde_engine(N, registry, declare_fi=False)
    ab_vec, ab_form = theta_courant(N, U, omega, V, eta)
    ba_vec, ba_form = theta_courant(N, V, eta, U, omega)
    return _zero_theorem(
        "poisson_generalized_courant_skew",
        "[x,y]_{C,θ} + [y,x]_{C,θ} = 0 on (TM)₀ ⊕ (T*M)_θ "
        "(tilde-Courant skewness; equivalently the relation "
        "[x,y]_{C,θ} = [x,y]_{D,θ} − D_θ⟨x,y⟩₊ kills the 6.D "
        "symmetric part). Any bivector.",
        [Sum(ab_vec, ba_vec), Sum(ab_form, ba_form)],
        engine,
        registry,
        from_axioms=(
            "tilde-Dorfman + pairing + D_θ definitions",
            "6.D symmetric part (𝒟 = d + d̃)",
        ),
        notes="PDF 14f.ii / Watamura §2",
        labels=(
            "vector component normalizes to 0",
            "form component normalizes to 0",
        ),
    )


def prove_theta_d_pairing_value(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The Uchino Rem 1 normalization of the θ-coboundary:

    ``⟨D_θh, x⟩₊ = ½·ρ(x)(h)``

    — the dh-component pairs with the vector part, the −θ♯dh
    component with the form part (bivector antisymmetry), together
    reproducing half the total-anchor action. Ties the θ-double to
    the 7.A abstract axiomatics."""
    _require_poisson(N)
    engine = _tilde_engine(N, registry, declare_fi=False)
    D_vec, D_form = d_operator_theta(N, h)
    lhs = canonical_pairing(D_vec, D_form, U, omega)
    rhs = Product(
        Rational(1, 2), Act(theta_anchor(N, U, omega), h)
    )
    return _zero_theorem(
        "poisson_generalized_d_pairing",
        "⟨D_θh, x⟩₊ = ½ρ(x)(h) on (TM)₀ ⊕ (T*M)_θ (Uchino Rem 1 "
        "normalization; 7.A cross-check)",
        [Sum(lhs, Neg(rhs))],
        engine,
        registry,
        from_axioms=(
            "D_θ + pairing + total anchor definitions",
            "bivector antisymmetry",
        ),
        notes="PDF 14f",
        labels=(
            "pairing value normalizes to the anchor action",
        ),
    )


def prove_theta_invariance(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    zeta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """[C'5] for the Poisson generalized double:

    ``ρ(x)⟨y, z⟩₊ = ⟨x∘_θ y, z⟩₊ + ⟨y, x∘_θ z⟩₊``

    with the TOTAL anchor acting on the scalar pairing.

    STATUS (2026-09-08): DIAGNOSED OPEN — an honest-fail. The
    identity is mathematically verified by hand (pairing-Leibniz +
    the sharp-pairing swap + 𝒦̃-antisymmetry close it on paper),
    and every ingredient identity normalizes to literal 0
    standalone in BOTH normal-form languages (the θ(·,·)-MultiEval
    "fold" language of the tilde engine and the ⟨·,θ♯·⟩-Pairing
    language without the fold). The MIXED-language residual however
    sits at a stuck rewriting fixpoint: cancellation needs the
    Pairing↔MultiEval bridge in the UNFOLD direction, which would
    cycle against the fold rule; and the 4.20 compat route is
    tautological at p = 1. Closing this cleanly needs either a
    termination-designed bidirectional bridge or 6.J-style
    stall-difference mining with cross-language 2NF instances —
    recorded in ROADMAP as the 14f.i deferral. This prover runs the
    machinery and raises with the 14-term residual."""
    _require_poisson(N)
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )
    from jacopy.packages.drinfeld.tilde_calculus import (
        InteriorAnticommuteDefinition,
        LieIotaCommutatorDefinition,
        NambuMorphismDeclaration,
    )
    from jacopy.packages.drinfeld.twist import _twist_engine

    # Deliberately WITHOUT the sharp-pairing FOLD (⟨β,θα⟩ → θ(α,β)):
    # the invariance identity closes in the pairing/Palais language
    # (dζ(U, θ♯η) unrolls to the bracket-pairing terms), which the
    # fold's collected MultiEval normal form blocks.
    engine = _twist_engine(N, registry)
    engine.register(LieIotaCommutatorDefinition())
    engine.register(InteriorAnticommuteDefinition())
    if declare_poisson:
        engine.register(NambuMorphismDeclaration(N))
    engine.register(ActExpansionDefinition(registry))
    engine.register(MultiEvalArgLinearityDefinition(registry))
    engine.register(LieBracketLeibnizDefinition(registry))
    lhs = Act(
        theta_anchor(N, U, omega),
        canonical_pairing(V, eta, W, zeta),
    )
    xy_vec, xy_form = theta_dorfman(N, U, omega, V, eta)
    xz_vec, xz_form = theta_dorfman(N, U, omega, W, zeta)
    rhs = Sum(
        canonical_pairing(xy_vec, xy_form, W, zeta),
        canonical_pairing(V, eta, xz_vec, xz_form),
    )
    diff = Sum(lhs, Neg(rhs))
    residual = _normalize(engine, diff, registry)
    steps: List[ProofStep] = [
        ProofStep(
            diff,
            residual,
            rule="normalize (tilde engine fixpoint)",
            justification="registered rules",
        )
    ]

    # The surviving terms are a combination of PAIRING-LEIBNIZ
    # bridges — mechanical zeros of the shape
    #
    #   ⟨a, [X, Y]⟩ − X(⟨a, Y⟩) + ⟨ℒ_X a, Y⟩ = 0
    #
    # (the Lie derivative's Leibniz rule over the canonical
    # pairing, with Y a sharp θ♯b) — subtracted greedily under a
    # strictly decreasing node-size metric (the 5.E.2b citation
    # mechanism, small scale).
    if residual != Integer(0):
        from jacopy.central.tangent.cartan import L as _L
        from jacopy.central.tangent.lie_bracket import (
            lie_bracket,
        )
        from jacopy.core.pairing import Pairing
        from jacopy.packages.poisson.koszul_jacobi import (
            _node_size,
        )

        trip = (omega, eta, zeta)
        directions = (U, V, W) + tuple(
            N.sharp_vf(a) for a in trip
        )
        # Each bridge is ENGINE-PROVEN zero as a whole; the citable
        # seed is the sum of its PER-TERM normal forms — already in
        # the residual's monomial language, so subtraction can only
        # collect and cancel (whole-sum normalization would just
        # give back 0 and teach nothing).
        seeds: List[Expr] = []
        for X in directions:
            for a in trip:
                for Y in directions:
                    if X is Y:
                        continue
                    parts = (
                        Pairing(a, lie_bracket(X, Y)),
                        Neg(Act(X, Pairing(a, Y))),
                        Pairing(_L(X, a), Y),
                    )
                    if (
                        _normalize(engine, Sum(*parts), registry)
                        != Integer(0)
                    ):
                        continue
                    cited = Sum(
                        *(
                            _normalize(engine, t, registry)
                            for t in parts
                        )
                    )
                    nf = _normalize(engine, cited, registry)
                    if nf == Integer(0):
                        continue
                    seeds.append(nf)
                    seeds.append(
                        _normalize(engine, Neg(cited), registry)
                    )
        for _round in range(24):
            if residual == Integer(0):
                break
            progressed = False
            for seed in seeds:
                cand = _normalize(
                    engine, Sum(residual, Neg(seed)), registry
                )
                if _node_size(cand) < _node_size(residual):
                    steps.append(
                        ProofStep(
                            residual,
                            cand,
                            rule=(
                                "cite pairing-Leibniz bridge "
                                "(proven zero)"
                            ),
                            justification=(
                                "ℒ_X over ⟨·,·⟩: "
                                "⟨a,[X,Y]⟩ = X⟨a,Y⟩ − ⟨ℒ_Xa, Y⟩"
                            ),
                            provenance_tag="theorem",
                        )
                    )
                    residual = cand
                    progressed = True
                    break
            if not progressed:
                break

    if residual != Integer(0):
        # ℚ-linear phase (the 5.E.2b mechanism): the residual is an
        # exact rational combination of the proven-zero seeds.
        from jacopy.packages.poisson.koszul_jacobi import (
            _gauss_solve,
            _to_vec,
        )

        target = _to_vec(residual)
        rows = [_to_vec(s) for s in seeds]
        keys = sorted(
            set(target) | {k for row in rows for k in row}
        )
        coeffs = (
            _gauss_solve(rows, target, keys) if rows else None
        )
        if coeffs is not None:
            from fractions import Fraction

            from jacopy.core.expr import Product as _Product
            from jacopy.core.expr import Rational as _Rational

            def times(c: Fraction, e: Expr) -> Expr:
                if c == 1:
                    return e
                if c == -1:
                    return Neg(e)
                n_, d_ = c.numerator, c.denominator
                base = e if n_ > 0 else Neg(e)
                ce = (
                    Integer(abs(n_))
                    if d_ == 1
                    else _Rational(abs(n_), d_)
                )
                return _Product(ce, base)

            nz = [
                (i, c) for i, c in enumerate(coeffs) if c != 0
            ]
            combo = Sum(
                residual,
                *(times(-c, seeds[i]) for i, c in nz),
            )
            check = _normalize(engine, combo, registry)
            if check == Integer(0):
                steps.append(
                    ProofStep(
                        residual,
                        Integer(0),
                        rule=(
                            "cite ℚ-linear combination of "
                            f"{len(nz)} pairing-Leibniz bridges "
                            "(engine-verified)"
                        ),
                        justification=(
                            "each bridge is a proven zero; the "
                            "combination subtracts to literal 0"
                        ),
                        provenance_tag="theorem",
                    )
                )
                residual = Integer(0)

    if residual != Integer(0):
        raise ProofFailure(
            "poisson_generalized_invariance: residual survives — "
            + residual._repr_inner()[:160]
        )

    chain = ProofChain(steps)
    theorem = Theorem(
        name="poisson_generalized_invariance",
        statement=(
            "ρ(x)⟨y,z⟩₊ = ⟨x∘_θ y, z⟩₊ + ⟨y, x∘_θ z⟩₊ on "
            "(TM)₀ ⊕ (T*M)_θ ([C'5], total anchor)"
        ),
        lhs=diff,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "tilde-Dorfman + canonical pairing definitions",
            "Cartan + Koszul/tilde calculus",
            "pairing-Leibniz bridges (cited mechanical zeros)",
        )
        + (
            ("declared Poisson condition",)
            if declare_poisson
            else ()
        ),
        notes="PDF 14f.i / Watamura §2",
    )
    return chain, theorem
