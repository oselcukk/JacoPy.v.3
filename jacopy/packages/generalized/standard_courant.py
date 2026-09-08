"""
The STANDARD Courant algebroid on ``TM ⊕ T*M`` (Phase 7.B; PDF item
14b-d): the Dorfman bracket's remaining Courant axioms, the Courant
(skew) bracket, and the Dorfman↔Courant relation — all CONCRETE
(component-wise on ``(vector, form)`` pairs), closing the loop with
the ABSTRACT axiomatics of Phase 7.A.

Inventory: Phase 6.C already proved, for the standard double,
right-Leibniz [C'3], the exact symmetric part ``x∘x = D(x,x)``
[C'4], and Leibniz-Jacobi [C'1]. This module adds

* **[C'2]** anchor morphism (``ρ(U+ω) = U`` — the vector component
  of the Dorfman bracket IS the Lie bracket),
* **[C'5]** invariance of the canonical pairing,
* the **Courant bracket** ``[x,y]_C = [x,y]_D − D⟨x,y⟩₊`` with
  ``D(h) = (0, ½·dh)`` (the ``½`` is exactly Uchino's Remark 1 /
  Prop 2.2 value: ``⟨Dh, V+η⟩₊ = ½ι_V dh = ½ρ(V+η)(h)``),
* its skew-symmetry and the relation theorem,
* the LWX axiom **[C3]** for the Courant bracket — the Leibniz rule
  with the pairing defect ``−⟨x,y⟩·Df``, concrete here after being
  proven redundant in the abstract setting (7.A).

Everything is engine-normalized on components; honest-fail
throughout (no identity is asserted without the difference
normalizing to literal 0).
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
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.drinfeld.double import (
    _double_engine,
    canonical_pairing,
    dorfman_double,
)


def _L(X: Expr, x: Expr) -> Expr:
    return Act(CARTAN_TM.lie(X), x)


def _iota(X: Expr, x: Expr) -> Expr:
    return Act(Interior(X), x)


def d_operator_std(h: Expr) -> Tuple[Expr, Expr]:
    """``D(h) = (0, ½·dh)`` — the standard double's coboundary,
    normalized so that ``⟨Dh, x⟩₊ = ½ρ(x)(h)`` (Uchino Rem 1 /
    Prop 2.2)."""
    return (Integer(0), Product(Rational(1, 2), d(h)))


def courant_bracket_std(
    U: Expr, omega: Expr, V: Expr, eta: Expr
) -> Tuple[Expr, Expr]:
    """The STANDARD Courant bracket — the skew-symmetrization of
    Dorfman:

        ( [U,V]_Lie , ℒ_U η − ℒ_V ω + ½·d(ι_V ω − ι_U η) )."""
    return (
        lie_bracket(U, V),
        Sum(
            _L(U, eta),
            Neg(_L(V, omega)),
            Product(Rational(1, 2), d(_iota(V, omega))),
            Neg(
                Product(Rational(1, 2), d(_iota(U, eta)))
            ),
        ),
    )


def _norm(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


def _component_theorem(
    name: str,
    statement: str,
    pairs,
    engine,
    registry,
    *,
    from_axioms: Tuple[str, ...],
    notes: str,
    rule_labels: Tuple[str, str] = (
        "vector component normalizes to 0",
        "form component normalizes to 0",
    ),
) -> Tuple[ProofChain, Theorem]:
    """Normalize each component difference to literal 0 and package
    the result; honest-fail with the surviving residual otherwise."""
    steps: List[ProofStep] = []
    for (label, diff) in zip(rule_labels, pairs):
        nf = _norm(engine, diff, registry)
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
                    "engine normal form of the component "
                    "difference"
                ),
            )
        )
    chain = ProofChain(steps)
    theorem = Theorem(
        name=name,
        statement=statement,
        lhs=pairs[0],
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=from_axioms,
        notes=notes,
    )
    return chain, theorem


def prove_anchor_morphism_std(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'2] for the standard double, concretely: with
    ``ρ(U+ω) = U``, the anchor of the Dorfman bracket is the Lie
    bracket of the anchors — the vector component of the bracket IS
    ``[U,V]_Lie``, so the defect normalizes to 0 in one step."""
    engine = _double_engine(registry)
    vec, _form = dorfman_double(U, omega, V, eta)
    diff = Sum(vec, Neg(lie_bracket(U, V)))
    return _component_theorem(
        "standard_dorfman_anchor_morphism",
        "ρ([x,y]_D) = [ρ(x), ρ(y)]_Lie on TM ⊕ T*M "
        "(the axiom [C'2], concrete — and redundant in the "
        "abstract setting by Phase 7.A)",
        [diff],
        engine,
        registry,
        from_axioms=(
            "Dorfman bracket definition (vector component)",
            "ρ(U+ω) = U (anchor definition)",
        ),
        notes=(
            "the vector component of the standard double is the "
            "Lie bracket by construction"
        ),
        rule_labels=(
            "anchor defect normalizes to 0",
        ),
    )


def prove_dorfman_invariance_std(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    zeta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'5] for the standard double, concretely:

    ``ρ(x)⟨y, z⟩₊ = ⟨x∘y, z⟩₊ + ⟨y, x∘z⟩₊``

    with ``x = U+ω``, ``y = V+η``, ``z = W+ζ``. A genuine Cartan
    computation: the left side is ``U(ι_W η + ι_V ζ)`` and the
    right side opens through the magic formula and the ``[ℒ,ι]``
    commutator."""
    engine = _double_engine(registry)
    lhs = Act(U, canonical_pairing(V, eta, W, zeta))
    xy_vec, xy_form = dorfman_double(U, omega, V, eta)
    xz_vec, xz_form = dorfman_double(U, omega, W, zeta)
    rhs = Sum(
        canonical_pairing(xy_vec, xy_form, W, zeta),
        canonical_pairing(V, eta, xz_vec, xz_form),
    )
    diff = Sum(lhs, Neg(rhs))
    return _component_theorem(
        "standard_dorfman_invariance",
        "ρ(x)⟨y,z⟩₊ = ⟨x∘y, z⟩₊ + ⟨y, x∘z⟩₊ on TM ⊕ T*M "
        "(the axiom [C'5], concrete)",
        [diff],
        engine,
        registry,
        from_axioms=(
            "Dorfman bracket + canonical pairing definitions",
            "Cartan calculus (magic formula, [ℒ,ι] commutator)",
        ),
        notes=(
            "uchino.pdf [C'5]; the invariance that makes the "
            "standard double a Courant algebroid"
        ),
        rule_labels=(
            "invariance defect normalizes to 0",
        ),
    )


def prove_courant_dorfman_relation(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """PDF 14d — the relation:

    ``[x, y]_C = [x, y]_D − D⟨x, y⟩₊``

    (equivalently: Courant is the skew-symmetrization of Dorfman,
    since ``[x,y]_D + [y,x]_D = 2·D⟨x,y⟩₊``). Component-wise
    normalization to 0."""
    engine = _double_engine(registry)
    c_vec, c_form = courant_bracket_std(U, omega, V, eta)
    d_vec, d_form = dorfman_double(U, omega, V, eta)
    D_vec, D_form = d_operator_std(
        canonical_pairing(U, omega, V, eta)
    )
    return _component_theorem(
        "standard_courant_dorfman_relation",
        "[x,y]_C = [x,y]_D − D⟨x,y⟩₊ on TM ⊕ T*M (PDF 14d)",
        [
            Sum(c_vec, Neg(d_vec), D_vec),
            Sum(c_form, Neg(d_form), D_form),
        ],
        engine,
        registry,
        from_axioms=(
            "Courant/Dorfman bracket + pairing + D definitions",
        ),
        notes=(
            "with D(h) = (0, ½dh); the ½ is the Uchino Rem 1 "
            "normalization"
        ),
    )


def prove_courant_skew_std(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The Courant bracket is genuinely SKEW:
    ``[x,y]_C + [y,x]_C = 0`` component-wise (the D-exact symmetric
    part of Dorfman is subtracted off exactly)."""
    engine = _double_engine(registry)
    ab_vec, ab_form = courant_bracket_std(U, omega, V, eta)
    ba_vec, ba_form = courant_bracket_std(V, eta, U, omega)
    return _component_theorem(
        "standard_courant_skew",
        "[x,y]_C + [y,x]_C = 0 on TM ⊕ T*M",
        [Sum(ab_vec, ba_vec), Sum(ab_form, ba_form)],
        engine,
        registry,
        from_axioms=(
            "Courant bracket definition",
            "Cartan calculus",
        ),
        notes="the LWX [C1]-side skew-symmetry, concrete",
    )


def prove_courant_c3_lwx_std(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The LWX axiom [C3] for the standard Courant bracket,
    concretely:

    ``[x, f·y]_C = f·[x,y]_C + (ρ(x)f)·y − ⟨x,y⟩₊·D(f)``

    — the pairing-defect Leibniz rule whose REDUNDANCY was the
    Phase 7.A abstract theorem; here it is verified on the nose.
    The scaled-direction Lie derivatives (``ℒ_{fV}``) open through
    the MAGIC FORMULA + interior C∞-linearity — no ``ℒ_{fX}`` law
    is assumed."""
    from jacopy.central.calculus import (
        InteriorVectorLinearityDefinition,
    )
    from jacopy.packages.drinfeld.twist import (
        MagicFormulaDefinition,
    )

    engine = _double_engine(registry)
    engine.register(MagicFormulaDefinition(registry))
    engine.register(
        InteriorVectorLinearityDefinition(registry)
    )
    lhs_vec, lhs_form = courant_bracket_std(
        U, omega, Product(f, V), Product(f, eta)
    )
    c_vec, c_form = courant_bracket_std(U, omega, V, eta)
    pair = canonical_pairing(U, omega, V, eta)
    D_vec, D_form = d_operator_std(f)
    rhs_vec = Sum(
        Product(f, c_vec),
        Product(Act(U, f), V),
        Neg(Product(pair, D_vec)),
    )
    rhs_form = Sum(
        Product(f, c_form),
        Product(Act(U, f), eta),
        Neg(Product(pair, D_form)),
    )
    return _component_theorem(
        "standard_courant_c3_lwx",
        "[x, f·y]_C = f·[x,y]_C + (ρ(x)f)·y − ⟨x,y⟩₊·Df on "
        "TM ⊕ T*M (LWX [C3], concrete; redundant abstractly by "
        "Phase 7.A)",
        [
            Sum(lhs_vec, Neg(rhs_vec)),
            Sum(lhs_form, Neg(rhs_form)),
        ],
        engine,
        registry,
        from_axioms=(
            "Courant bracket + pairing + D definitions",
            "Cartan calculus (Leibniz rules)",
        ),
        notes=(
            "matches Uchino Prop 2.1(i) on the standard double"
        ),
    )


def courant_jacobiator_std(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    mu: Expr,
) -> Tuple[Expr, Expr]:
    """``Σcyc [[x,y]_C, z]_C`` — the cyclic Jacobiator of the
    Courant bracket, as a component pair."""

    def nest(a, b, c, d_, e, g):
        iv, if_ = courant_bracket_std(a, b, c, d_)
        return courant_bracket_std(iv, if_, e, g)

    parts = [
        nest(U, omega, V, eta, W, mu),
        nest(V, eta, W, mu, U, omega),
        nest(W, mu, U, omega, V, eta),
    ]
    return (
        Sum(*(p[0] for p in parts)),
        Sum(*(p[1] for p in parts)),
    )


def courant_t_scalar_std(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    mu: Expr,
) -> Expr:
    """``T(x,y,z) = ⅓·Σcyc ⟨[x,y]_C, z⟩₊`` — the LWX Jacobi
    defect scalar."""
    terms = []
    for (a, b, c, d_, e, g) in (
        (U, omega, V, eta, W, mu),
        (V, eta, W, mu, U, omega),
        (W, mu, U, omega, V, eta),
    ):
        bv, bf = courant_bracket_std(a, b, c, d_)
        terms.append(canonical_pairing(bv, bf, e, g))
    return Product(Rational(1, 3), Sum(*terms))


def prove_courant_jacobi_dt_std(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    mu: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, List[Theorem]]:
    """The LWX axiom [C1] for the standard Courant bracket,
    concretely:

    ``[[x,y]_C, z]_C + [[y,z]_C, x]_C + [[z,x]_C, y]_C =
    D T(x,y,z)``

    with ``T = ⅓Σcyc⟨[x,y]_C, z⟩₊`` and ``D h = (0, ½dh)``. The
    vector component is the VF Jacobi (cyclic form — closes on the
    action probe); the FORM component is evaluated on the given
    probe slots and closes through the 6.C bracket-identity repair
    loop. Returns the form chain + the cited VF identities."""
    from jacopy.central.tangent.cartan import (
        prove_with_bracket_identities,
    )
    from jacopy.central.calculus import (
        InteriorVectorLinearityDefinition,
    )
    from jacopy.packages.drinfeld.twist import (
        MagicFormulaDefinition,
    )
    from jacopy.packages.drinfeld.double import _ev

    engine = _double_engine(registry)
    engine.register(MagicFormulaDefinition(registry))
    engine.register(
        InteriorVectorLinearityDefinition(registry)
    )

    jac_vec, jac_form = courant_jacobiator_std(
        U, omega, V, eta, W, mu
    )
    _dt_vec, dt_form = d_operator_std(
        courant_t_scalar_std(U, omega, V, eta, W, mu)
    )

    # Vector component: Σcyc[[U,V],W] = 0 on the action probe f
    # (agreement on generators — a VF vanishing on every function
    # is zero); D T has no vector component.
    from jacopy.packages.poisson.tilde import _normalized_by

    probe = Act(jac_vec, f)
    nf = _normalized_by(engine, probe, registry)
    if nf != Integer(0):
        raise ProofFailure(
            "Courant [C1]: the vector-component Jacobi probe "
            "does not vanish — residual "
            + nf._repr_inner()[:140]
        )

    node = Sum(
        _ev(jac_form, slots), Neg(_ev(dt_form, slots))
    )
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=engine,
        max_steps=max_steps,
    )
