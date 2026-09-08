"""
The H-TWISTED standard Courant algebroid on ``TM ⊕ T*M`` (Phase
7.B.2; PDF item 14e): the H-twisted Dorfman bracket's Courant-axiom
suite, the H-twisted Courant (skew) bracket, and the dH-defect
Jacobi, concrete — parallel to :mod:`standard_courant` (the H = 0
case) and built on the Phase 6.E/6.F machinery
(:func:`~jacopy.packages.drinfeld.twist.dorfman_double_h`, the 6.F
"twisted Courant ⟺ dH = 0" theorem).

The H-term ``ι_V ι_U H`` (a closed 3-form's two-leg contraction)
is C∞-TENSORIAL and ANTISYMMETRIC in ``U, V``; consequently:

* [C'2] anchor morphism — unchanged (the H-term is form-valued),
* [C'3] right-Leibniz — the H-term is tensorial, no new defect,
* [C'4] symmetric part — the H-term drops out (antisymmetry), the
  twisted bracket has the SAME ``D(h) = (0, ½dh)``,
* [C'5] invariance — the two H-contributions cancel
  (``ι_W ι_V ι_U H + ι_V ι_W ι_U H = 0``),
* [C'1] Leibniz-Jacobi — the ONLY axiom that sees ``H``: the defect
  is exactly ``−ι_W ι_V ι_U dH`` (proved in 6.F degree-generally;
  here instantiated concretely at p = 1), so the twisted bracket is
  Courant ⟺ ``dH = 0``, and ``H = dB`` (a Ševera B-transform)
  always restores Jacobi.

Everything engine-normalized on components; honest-fail throughout.
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
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.drinfeld.double import (
    _double_engine,
    canonical_pairing,
)
from jacopy.packages.drinfeld.twist import (
    dorfman_double_h,
    h_term,
)
from jacopy.packages.generalized.standard_courant import (
    _component_theorem,
    courant_bracket_std,
    d_operator_std,
)


def _iota(X: Expr, x: Expr) -> Expr:
    return Act(Interior(X), x)


def courant_bracket_h(
    H: Expr, U: Expr, omega: Expr, V: Expr, eta: Expr
) -> Tuple[Expr, Expr]:
    """The H-twisted Courant bracket — the skew form:
    standard Courant plus the (already antisymmetric) H-term."""
    vec, form = courant_bracket_std(U, omega, V, eta)
    return vec, Sum(form, h_term(H, U, V))


def prove_twisted_anchor_morphism(
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'2] for the H-twisted Dorfman bracket: the H-term is
    form-valued, so the vector component — and with it the anchor
    morphism — is untouched by the twist."""
    engine = _double_engine(registry)
    vec, _form = dorfman_double_h(H, U, omega, V, eta)
    diff = Sum(vec, Neg(lie_bracket(U, V)))
    return _component_theorem(
        "twisted_dorfman_anchor_morphism",
        "ρ([x,y]_{D,H}) = [ρ(x), ρ(y)]_Lie on TM ⊕ T*M — the "
        "H-twist never touches the vector component ([C'2])",
        [diff],
        engine,
        registry,
        from_axioms=(
            "H-twisted Dorfman definition (vector component)",
            "ρ(U+ω) = U (anchor definition)",
        ),
        notes="PDF 14e; the H = 0 case is standard_courant [C'2]",
        rule_labels=("anchor defect normalizes to 0",),
    )


def prove_twisted_right_leibniz(
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'3] for the H-twisted Dorfman bracket:

    ``[x, f·y]_{D,H} = f·[x,y]_{D,H} + (ρ(x)f)·y``

    — the H-term ``ι_{fV} ι_U H = f·ι_V ι_U H`` is C∞-tensorial, so
    the twist adds NO new Leibniz defect. As in the untwisted [C3]
    (standard_courant), the scaled-direction Lie derivative opens
    through the magic formula + interior C∞-linearity — no
    ``ℒ_{fX}`` law is assumed."""
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
    lhs_vec, lhs_form = dorfman_double_h(
        H, U, omega, Product(f, V), Product(f, eta)
    )
    b_vec, b_form = dorfman_double_h(H, U, omega, V, eta)
    rhs_vec = Sum(Product(f, b_vec), Product(Act(U, f), V))
    rhs_form = Sum(
        Product(f, b_form), Product(Act(U, f), eta)
    )
    return _component_theorem(
        "twisted_dorfman_right_leibniz",
        "[x, f·y]_{D,H} = f·[x,y]_{D,H} + (ρ(x)f)·y on TM ⊕ T*M "
        "([C'3]; the H-term is tensorial)",
        [
            Sum(lhs_vec, Neg(rhs_vec)),
            Sum(lhs_form, Neg(rhs_form)),
        ],
        engine,
        registry,
        from_axioms=(
            "H-twisted Dorfman + pairing definitions",
            "Cartan calculus (interior C∞-linearity)",
        ),
        notes="PDF 14e",
    )


def prove_twisted_symmetric_part(
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'4] for the H-twisted Dorfman bracket:

    ``[x,y]_{D,H} + [y,x]_{D,H} = 2·D⟨x,y⟩₊``

    — the H-term is ANTISYMMETRIC (``ι_V ι_U H + ι_U ι_V H = 0``,
    the theorem-classified interior anticommutation), so the
    symmetric part — and with it the operator ``D`` — is the
    UNTWISTED one."""
    from jacopy.packages.drinfeld.tilde_calculus import (
        InteriorAnticommuteDefinition,
    )

    engine = _double_engine(registry)
    engine.register(InteriorAnticommuteDefinition())
    ab_vec, ab_form = dorfman_double_h(H, U, omega, V, eta)
    ba_vec, ba_form = dorfman_double_h(H, V, eta, U, omega)
    D_vec, D_form = d_operator_std(
        canonical_pairing(U, omega, V, eta)
    )
    return _component_theorem(
        "twisted_dorfman_symmetric_part",
        "[x,y]_{D,H} + [y,x]_{D,H} = 2·D⟨x,y⟩₊ on TM ⊕ T*M "
        "([C'4]; the antisymmetric H-term drops out — the twist "
        "does not change D)",
        [
            Sum(ab_vec, ba_vec, Neg(Product(Integer(2), D_vec))),
            Sum(
                ab_form,
                ba_form,
                Neg(Product(Integer(2), D_form)),
            ),
        ],
        engine,
        registry,
        from_axioms=(
            "H-twisted Dorfman + pairing + D definitions",
            "antisymmetry of the interior contraction",
        ),
        notes="PDF 14e",
    )


def prove_twisted_invariance(
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    zeta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'5] for the H-twisted Dorfman bracket:

    ``ρ(x)⟨y,z⟩₊ = ⟨x∘_H y, z⟩₊ + ⟨y, x∘_H z⟩₊``

    — the two H-contributions ``ι_W ι_V ι_U H + ι_V ι_W ι_U H``
    cancel by antisymmetry, so the twisted bracket preserves the
    SAME canonical pairing."""
    engine = _double_engine(registry)
    lhs = Act(U, canonical_pairing(V, eta, W, zeta))
    xy_vec, xy_form = dorfman_double_h(H, U, omega, V, eta)
    xz_vec, xz_form = dorfman_double_h(H, U, omega, W, zeta)
    rhs = Sum(
        canonical_pairing(xy_vec, xy_form, W, zeta),
        canonical_pairing(V, eta, xz_vec, xz_form),
    )
    return _component_theorem(
        "twisted_dorfman_invariance",
        "ρ(x)⟨y,z⟩₊ = ⟨x∘_H y, z⟩₊ + ⟨y, x∘_H z⟩₊ on TM ⊕ T*M "
        "([C'5]; the H-contributions cancel pairwise)",
        [Sum(lhs, Neg(rhs))],
        engine,
        registry,
        from_axioms=(
            "H-twisted Dorfman + pairing definitions",
            "Cartan calculus (magic formula, [ℒ,ι] commutator, "
            "interior antisymmetry)",
        ),
        notes="PDF 14e",
        rule_labels=("invariance defect normalizes to 0",),
    )


def prove_twisted_courant_dorfman_relation(
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The H-twisted Dorfman↔Courant relation:
    ``[x,y]_{C,H} = [x,y]_{D,H} − D⟨x,y⟩₊`` — the twist commutes
    with skew-symmetrization (the H-term is already skew)."""
    engine = _double_engine(registry)
    c_vec, c_form = courant_bracket_h(H, U, omega, V, eta)
    d_vec, d_form = dorfman_double_h(H, U, omega, V, eta)
    D_vec, D_form = d_operator_std(
        canonical_pairing(U, omega, V, eta)
    )
    return _component_theorem(
        "twisted_courant_dorfman_relation",
        "[x,y]_{C,H} = [x,y]_{D,H} − D⟨x,y⟩₊ on TM ⊕ T*M "
        "(PDF 14d-e; the twist commutes with the "
        "skew-symmetrization)",
        [
            Sum(c_vec, Neg(d_vec), D_vec),
            Sum(c_form, Neg(d_form), D_form),
        ],
        engine,
        registry,
        from_axioms=(
            "H-twisted Courant/Dorfman + pairing + D definitions",
        ),
        notes="PDF 14e",
    )


def prove_twisted_jacobi_dh_defect(
    H: Expr,
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
    max_steps: int = 60000,
) -> Tuple[ProofChain, List[Theorem]]:
    """[C'1] for the H-twisted Dorfman bracket, concretely at p = 1:
    the form-component Leibniz-Jacobi defect is EXACTLY the ``dH``
    evaluation,

        form([x,[y,z]_H]_H − [[x,y]_H,z]_H − [y,[x,z]_H]_H)
            = ι_W ι_V ι_U dH,

    evaluated on the probe slots (the vector components never see
    ``H`` — the VF Jacobi closes them as in the untwisted case).
    Corollary (Ševera): ``H = dB`` gives ``dH = d²B = 0``, so a
    B-transform of a Courant structure is again Courant; general
    ``H`` is Courant ⟺ ``dH = 0`` (PDF 14e / the concrete p = 1
    face of the 6.F degree-general theorem)."""
    from jacopy.central.tangent.cartan import (
        prove_with_bracket_identities,
    )
    from jacopy.packages.drinfeld.double import _ev

    def nest(U1, o1, U2, o2, U3, o3):
        iv, if_ = dorfman_double_h(H, U2, o2, U3, o3)
        return dorfman_double_h(H, U1, o1, iv, if_)[1]

    v12, f12 = dorfman_double_h(H, U, omega, V, eta)
    defect = Sum(
        nest(U, omega, V, eta, W, mu),
        Neg(dorfman_double_h(H, v12, f12, W, mu)[1]),
        Neg(nest(V, eta, U, omega, W, mu)),
        Neg(_iota(W, _iota(V, _iota(U, d(H))))),
    )
    node = _ev(defect, slots)
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=_double_engine(registry),
        max_steps=max_steps,
    )


def prove_severa_twist_is_courant(
    B: Expr,
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
    max_steps: int = 60000,
) -> Tuple[ProofChain, List[Theorem]]:
    """The Ševera corollary, concrete: for EXACT twist ``H = dB``
    (a B-transform) the Leibniz-Jacobi of the twisted Dorfman
    bracket closes EXACTLY — ``dH = d²B = 0`` kills the defect, so
    the B-transform of the standard Courant structure is again a
    Courant algebroid (PDF 14e)."""
    from jacopy.central.tangent.cartan import (
        prove_with_bracket_identities,
    )
    from jacopy.packages.drinfeld.double import _ev

    H = d(B)

    def nest(U1, o1, U2, o2, U3, o3):
        iv, if_ = dorfman_double_h(H, U2, o2, U3, o3)
        return dorfman_double_h(H, U1, o1, iv, if_)[1]

    v12, f12 = dorfman_double_h(H, U, omega, V, eta)
    defect = Sum(
        nest(U, omega, V, eta, W, mu),
        Neg(dorfman_double_h(H, v12, f12, W, mu)[1]),
        Neg(nest(V, eta, U, omega, W, mu)),
    )
    node = _ev(defect, slots)
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=_double_engine(registry),
        max_steps=max_steps,
    )
