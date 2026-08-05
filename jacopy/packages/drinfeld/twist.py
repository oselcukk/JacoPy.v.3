"""
Twist automorphisms and the brackets they induce (Phase 6.E) —
[2409.11973 §7]: a bundle automorphism ``Ψ ∈ Aut(A ⊕ Z)`` turns an
initial bracket into

    [u, v]_Ψ := Ψ⁻¹ [Ψu, Ψv]                                  (7.3)

and the twisted bracket's components read off a whole twisted
calculus family (7.25). Two of the three special shapes (7.11) land
here (the frame-change ``Ψ_m`` arrives with 6.E-devam):

* **Π-transform** ``Ψ_Π = [[1, Π], [0, 1]]`` with ``Π`` a
  ``(p+1)``-vector acting as ``Λᵖ T*M → TM``. THE structural
  theorem: twisting the STANDARD Dorfman bracket by ``Ψ_Π``
  reproduces the Nambu-Poisson tilde calculus —

      form([e₁,e₂]_Ψ) = form of the Nambu double
                        (the higher Koszul bracket is BORN from the
                        twist),
      vec([e₁,e₂]_Ψ)  = vec of the Nambu double + R′(ω, η),

  with the R-twist obstruction ``R′(ω,η) = [Πω,Πη] − Π[ω,η]_Kos``
  — all DECLARATION-FREE (any ``(p+1)``-vector; the twist does not
  know Π is Nambu-Poisson). ``R′ = 0`` is exactly the
  fundamental-identity declaration, consumed separately.

* **B-transform** ``Ψ_B = [[1, 0], [B, 1]]`` with ``B`` a
  ``(p+1)``-form acting as ``TM → Λᵖ T*M``, ``BU = ι_U B``. The
  Ševera remnant (7.20): the twisted Dorfman differs from the
  original by an H-twist with ``H = dB`` —

      vec([e₁,e₂]_Ψ)  = vec of the Dorfman double (unchanged),
      form([e₁,e₂]_Ψ) = form of the Dorfman double + ι_V ι_U dB.

Twisted calculus elements (7.25, H = 0 initial bracket) are proved
against the 6.D tilde calculus componentwise; the mediating fact is
the Cartan magic formula (``𝒦_V = −ι_V d``), applied by the engine.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition
from jacopy.proof.chain import ProofChain
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.packages.drinfeld.double import (
    _ev,
    _iota,
    _L,
    dorfman_double,
    lie_tilde_nambu,
    nambu_double,
)


# ------------------------------------------------------------------- #
# Twist maps on component pairs                                        #
# ------------------------------------------------------------------- #


def pi_twist(N, vec: Expr, form: Expr) -> Tuple[Expr, Expr]:
    """``Ψ_Π (vec ⊕ form) = (vec + Π form) ⊕ form`` [eq (7.11)]."""
    return Sum(vec, N.sharp_vf(form)), form


def pi_twist_inverse(N, vec: Expr, form: Expr) -> Tuple[Expr, Expr]:
    """``Ψ_Π⁻¹ (vec ⊕ form) = (vec − Π form) ⊕ form`` [eq (7.12)]."""
    return Sum(vec, Neg(N.sharp_vf(form))), form


def b_twist(B: Expr, vec: Expr, form: Expr) -> Tuple[Expr, Expr]:
    """``Ψ_B (vec ⊕ form) = vec ⊕ (ι_vec B + form)`` [eq (7.11)] —
    ``B`` is a ``(p+1)``-form acting as ``TM → Λᵖ`` by interior
    product."""
    return vec, Sum(Act(Interior(vec), B), form)


def b_twist_inverse(B: Expr, vec: Expr, form: Expr) -> Tuple[Expr, Expr]:
    """``Ψ_B⁻¹ (vec ⊕ form) = vec ⊕ (form − ι_vec B)``."""
    return vec, Sum(form, Neg(Act(Interior(vec), B)))


def twisted_dorfman(
    apply_twist, invert_twist, U: Expr, omega: Expr, V: Expr, eta: Expr
) -> Tuple[Expr, Expr]:
    """``[e₁, e₂]_Ψ = Ψ⁻¹ [Ψ e₁, Ψ e₂]_Dorfman`` [eq (7.3)] on
    component pairs; ``apply_twist(vec, form)`` and
    ``invert_twist(vec, form)`` return component pairs."""
    u1, o1 = apply_twist(U, omega)
    v1, e1 = apply_twist(V, eta)
    return invert_twist(*dorfman_double(u1, o1, v1, e1))


def r_twist(N, omega: Expr, eta: Expr) -> Expr:
    """``R′(ω, η) = [Πω, Πη]_Lie − Π[ω,η]_Kos`` [eq (7.25) with the
    Dorfman initial bracket] — the anchor defect of the twisted
    Z-bracket; vanishes exactly under the fundamental identity."""
    from jacopy.central.tangent.lie_bracket import lie_bracket
    from jacopy.packages.poisson.nambu import nambu_koszul_bracket

    return Sum(
        lie_bracket(N.sharp_vf(omega), N.sharp_vf(eta)),
        Neg(N.sharp_vf(nambu_koszul_bracket(N, omega, eta))),
    )




# ------------------------------------------------------------------- #
# Scoped engine rules (registered by _twist_engine only)               #
# ------------------------------------------------------------------- #


class MagicFormulaDefinition(Definition):
    """THEOREM-classified Cartan magic as a form-level rewrite:
    ``L_X ω → ι_X(dω) + d(ι_X ω)`` for a TM Cartan Lie derivative on
    a determinable form of degree ≥ 1.

    The magic formula is a proven theorem (Phase 2 seed + the
    higher-degree audit to p = 5); the always-on engines apply it at
    EVAL sites via the Palais route. The twist proofs need it at the
    OPERATOR level: (7.25) writes the twisted family through
    ``𝒦_V = −ℒ_V + dι_V`` and the equivalence with the tilde
    calculus is mediated by magic on forms sitting INSIDE sharp
    slots, where no evaluation ever happens. Guarded away from
    vectors/multivectors (``wedge_degree`` protocol) — ``L_X Y`` is a
    bracket, not a magic candidate."""

    anchor = Act

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry
        self.name = (
            "theorem Cartan magic: L_X ω = ι_X(dω) + d(ι_X ω)"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.algebra.derivation import degree_of
        from jacopy.central.calculus.bracket_calculus import (
            LieDerivative,
        )

        if not (
            isinstance(expr, Act)
            and isinstance(expr.op, LieDerivative)
            and expr.op.calculus_name == CARTAN_TM.name
        ):
            return False
        arg = expr.arg
        if hasattr(arg, "wedge_degree"):
            return False
        try:
            k = degree_of(arg, self._registry).as_int()
        except ValueError:
            return False
        return k is not None and k >= 1

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.op.vector
        arg = expr.arg
        return Sum(
            Act(Interior(X), d(arg)),
            d(Act(Interior(X), arg)),
        )

    def theorem_proof_builder(self):
        from jacopy.proof.chain import ProofChain
        from jacopy.proof.step import ProofStep

        def _builder(matched: Expr) -> ProofChain:
            return ProofChain(
                [
                    ProofStep(
                        matched,
                        self.rewrite(matched),
                        rule="Cartan magic (proven, Phase 2 + audit p ≤ 5)",
                        justification=(
                            "prove_cartan_magic_* close the eval-level "
                            "statement mechanically"
                        ),
                    )
                ]
            )

        return _builder


class PairingArgSplitDefinition(Definition):
    """``⟨α, X + Y⟩ → ⟨α, X⟩ + ⟨α, Y⟩`` — pairing-argument
    additivity, SCOPED to the twist engine.

    The always-on engines deliberately leave pairing-argument sums to
    the COLLECTING pass (``collect_pairings``) because the
    citation-driven flows need the collected shape. Twist proofs are
    pure cancellation: both sides must reach one normal form, and the
    split direction is the terminating one here. Do not register this
    next to theorem-citation tactics."""

    name = "pairing argument additivity: ⟨α, X+Y⟩ = ⟨α,X⟩ + ⟨α,Y⟩"
    anchor = Pairing

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Pairing) and isinstance(expr.X, Sum)

    def rewrite(self, expr: Expr) -> Expr:
        return Sum(
            *(Pairing(expr.alpha, t) for t in expr.X.children)
        )



def _twist_engine(N, registry):
    """6.D double engine + operator slot additivity (the twist puts
    composite directions like ``U + Πω`` inside Lie subscripts that
    sit in other operators' slots — the 6.E gap)."""
    from jacopy.central.calculus import (
        OperatorSlotAdditivityDefinition,
    )
    from jacopy.packages.drinfeld.double import _nambu_double_engine

    eng = _nambu_double_engine(N, registry)
    eng.register(OperatorSlotAdditivityDefinition())
    eng.register(MagicFormulaDefinition(registry))
    eng.register(PairingArgSplitDefinition())
    eng.register(FormProductToWedgeDefinition(registry))
    eng.register(WedgeSumScalarDefinition(registry))
    return eng


# ------------------------------------------------------------------- #
# Π-transform theorems                                                 #
# ------------------------------------------------------------------- #


def prove_pi_twist_form_is_nambu(
    N,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Form component of the Ψ_Π-twisted Dorfman bracket equals the
    form component of the Nambu double — the higher Koszul bracket
    emerges mechanically from the twist. Declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify

    _, tw_form = twisted_dorfman(
        lambda v, f: pi_twist(N, v, f),
        lambda v, f: pi_twist_inverse(N, v, f),
        U, omega, V, eta,
    )
    _, nb_form = nambu_double(N, U, omega, V, eta)
    node = _ev(Sum(tw_form, Neg(nb_form)), slots)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


def prove_pi_twist_vec_is_nambu_plus_r(
    N,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Vector component of the Ψ_Π-twisted Dorfman bracket equals
    the Nambu double's vector component PLUS the R-twist
    ``R′(ω,η) = [Πω,Πη] − Π[ω,η]_Kos`` [the (7.25) obstruction] —
    declaration-free for ANY ``(p+1)``-vector. Under the fundamental
    identity ``R′ = 0`` and the twist reproduces the tilde calculus
    exactly. Acting on the probe function ``h``."""
    from jacopy.proof.strategies import ExpandAndSimplify

    tw_vec, _ = twisted_dorfman(
        lambda v, f: pi_twist(N, v, f),
        lambda v, f: pi_twist_inverse(N, v, f),
        U, omega, V, eta,
    )
    nb_vec, _ = nambu_double(N, U, omega, V, eta)
    node = Act(
        Sum(tw_vec, Neg(nb_vec), Neg(r_twist(N, omega, eta))), h
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


def prove_twisted_lie_tilde(
    N,
    omega: Expr,
    V: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """``ℒ̃′_ω V = [Πω, V] − Π(𝒦_V ω)`` [eq (7.25)] equals the 6.D
    tilde Lie derivative ``ℒ̃_ω V = [Πω, V] + Π(ι_V dω)`` — the
    mediating fact is the magic formula ``𝒦_V = −ι_V d``. Acting on
    ``h``; declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.calculus_conditions import kappa

    lhs = Sum(
        _twisted_lie_bracket_part(N, omega, V),
        Neg(N.sharp_vf(kappa(V, omega))),
    )
    node = Act(Sum(lhs, Neg(lie_tilde_nambu(N, omega, V))), h)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


def _twisted_lie_bracket_part(N, omega: Expr, V: Expr) -> Expr:
    from jacopy.central.tangent.lie_bracket import lie_bracket

    return lie_bracket(N.sharp_vf(omega), V)


def prove_twisted_kappa_tilde(
    N,
    eta: Expr,
    U: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """``𝒦̃′_η U = [U, Πη] − Π(ℒ_U η)`` [eq (7.25)] equals the 6.D
    combination ``−ℒ̃_η U + d̃(ι_U η)`` with ``d̃ = −Πd`` (again via
    the magic formula). Acting on ``h``; declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.tangent.lie_bracket import lie_bracket

    lhs = Sum(
        lie_bracket(U, N.sharp_vf(eta)),
        Neg(N.sharp_vf(_L(U, eta))),
    )
    rhs = Sum(
        Neg(lie_tilde_nambu(N, eta, U)),
        Neg(N.sharp_vf(d(_iota(U, eta)))),
    )
    node = Act(Sum(lhs, Neg(rhs)), h)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


def prove_twisted_z_bracket_is_koszul(
    N,
    omega: Expr,
    eta: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """``[ω,η]′_Z = ℒ_{Πω} η + 𝒦_{Πη} ω`` [eq (7.25)] equals the
    higher Koszul bracket ``ℒ_{Πω} η − ι_{Πη} dω`` (magic formula).
    Evaluated on the given slots; declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.calculus_conditions import kappa
    from jacopy.packages.poisson.nambu import nambu_koszul_bracket

    lhs = Sum(
        _L(N.sharp_vf(omega), eta),
        kappa(N.sharp_vf(eta), omega),
    )
    node = _ev(
        Sum(lhs, Neg(nambu_koszul_bracket(N, omega, eta))), slots
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


# ------------------------------------------------------------------- #
# B-transform theorems (Ševera remnant)                                #
# ------------------------------------------------------------------- #


def prove_b_twist_vec_unchanged(
    B: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Vector component of the Ψ_B-twisted Dorfman bracket is the
    untouched ``[U,V]_Lie`` [eq (7.18) with zero tilde side]. Acting
    on ``h``; declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.tangent.engine import tangent_engine

    tw_vec, _ = twisted_dorfman(
        lambda v, f: b_twist(B, v, f),
        lambda v, f: b_twist_inverse(B, v, f),
        U, omega, V, eta,
    )
    base_vec, _ = dorfman_double(U, omega, V, eta)
    node = Act(Sum(tw_vec, Neg(base_vec)), h)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=tangent_engine(registry=registry),
        max_steps=max_steps,
    )


def prove_b_twist_severa(
    B: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """The Ševera remnant [eq (7.20)]: the Ψ_B-twisted Dorfman
    bracket differs from the original by the H-twist ``H = dB``,

        form([e₁,e₂]_Ψ) − form([e₁,e₂]) = ι_V ι_U dB,

    evaluated on the given slots. Declaration-free (any ``(p+1)``-form
    ``B``); when ``dB = 0`` the Dorfman bracket is preserved."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        HeadFormProductLiftDefinition,
        InteriorVectorLinearityDefinition,
        OperatorSlotAdditivityDefinition,
    )
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )
    from jacopy.central.calculus import (
        MultiEvalArgLinearityDefinition,
    )

    _, tw_form = twisted_dorfman(
        lambda v, f: b_twist(B, v, f),
        lambda v, f: b_twist_inverse(B, v, f),
        U, omega, V, eta,
    )
    _, base_form = dorfman_double(U, omega, V, eta)
    h_term = Act(Interior(V), Act(Interior(U), d(B)))
    node = _ev(
        Sum(tw_form, Neg(base_form), Neg(h_term)), slots
    )
    eng = tangent_engine(registry=registry)
    eng.register(LieBracketLeibnizDefinition(registry))
    eng.register(MultiEvalArgLinearityDefinition(registry))
    eng.register(InteriorVectorLinearityDefinition(registry))
    eng.register(ActExpansionDefinition(registry))
    eng.register(HeadFormProductLiftDefinition(registry))
    eng.register(OperatorSlotAdditivityDefinition())
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )


# ------------------------------------------------------------------- #
# R′-linearity obstruction (7.26)                                      #
# ------------------------------------------------------------------- #


def prove_r_twist_second_entry_linear(
    N,
    omega: Expr,
    eta: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """``R′(ω, f·η) = f·R′(ω, η)`` — the R-twist IS C∞-linear in its
    SECOND entry [eq (7.26) context]. Acting on the probe ``h``;
    declaration-free."""
    from jacopy.core.expr import Product
    from jacopy.proof.strategies import ExpandAndSimplify

    node = Act(
        Sum(
            r_twist(N, omega, Product(f, eta)),
            Neg(Product(f, r_twist(N, omega, eta))),
        ),
        h,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


def prove_r_twist_first_entry_obstruction(
    N,
    omega: Expr,
    eta: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """The FIRST-entry C∞-linearity obstruction of the R-twist
    [eq (7.26), TM Lie algebroid case ``L_A = 0``,
    ``σ_d(f, x) = df ∧ x``]:

        R′(f·ω, η) = f·R′(ω, η) − Π( df ∧ (ι_{Πω}η + ι_{Πη}ω) ).

    The anomaly is ``−Π(df ∧ g_Z(ω,η))`` — nonzero in general, so a
    C∞-linear R′ forces ``Π`` to have nontrivial kernel content.
    Acting on ``h``; declaration-free."""
    from jacopy.core.expr import Product
    from jacopy.core.wedge import Wedge
    from jacopy.proof.strategies import ExpandAndSimplify

    g_z = Sum(
        _iota(N.sharp_vf(omega), eta),
        _iota(N.sharp_vf(eta), omega),
    )
    anomaly = Neg(N.sharp_vf(Wedge(d(f), g_z)))
    node = Act(
        Sum(
            r_twist(N, Product(f, omega), eta),
            Neg(Product(f, r_twist(N, omega, eta))),
            Neg(anomaly),
        ),
        h,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


# ------------------------------------------------------------------- #
# H-twisted initial bracket                                            #
# ------------------------------------------------------------------- #


def h_term(H: Expr, U: Expr, V: Expr) -> Expr:
    """``H(U, V) := ι_V ι_U H`` — the ``Λᵖ``-valued H-twist term of a
    ``(p+2)``-form ``H``."""
    return Act(Interior(V), Act(Interior(U), H))


def dorfman_double_h(
    H: Expr, U: Expr, omega: Expr, V: Expr, eta: Expr
) -> Tuple[Expr, Expr]:
    """The H-TWISTED standard Dorfman bracket on ``TM ⊕ Λᵖ``:
    the form component gains ``ι_V ι_U H``."""
    vec, form = dorfman_double(U, omega, V, eta)
    return vec, Sum(form, h_term(H, U, V))


def twisted_dorfman_h(
    H: Expr,
    apply_twist,
    invert_twist,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
) -> Tuple[Expr, Expr]:
    """``Ψ⁻¹ [Ψe₁, Ψe₂]_{H-Dorfman}`` [eq (7.3) with an H-twisted
    initial bracket]."""
    u1, o1 = apply_twist(U, omega)
    v1, e1 = apply_twist(V, eta)
    return invert_twist(*dorfman_double_h(H, u1, o1, v1, e1))


def prove_b_twist_severa_h(
    H: Expr,
    B: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """The FULL Ševera statement [eq (7.19)-(7.20)]: Ψ_B applied to
    the H-TWISTED Dorfman bracket gives the ``(H + dB)``-twisted
    Dorfman bracket,

        form([e₁,e₂]_{Ψ_B, H}) = form([e₁,e₂]_{H+dB}),

    evaluated on the given slots. Declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        HeadFormProductLiftDefinition,
        InteriorVectorLinearityDefinition,
        MultiEvalArgLinearityDefinition,
        OperatorSlotAdditivityDefinition,
    )
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )

    _, tw_form = twisted_dorfman_h(
        H,
        lambda v, f: b_twist(B, v, f),
        lambda v, f: b_twist_inverse(B, v, f),
        U, omega, V, eta,
    )
    _, target_form = dorfman_double_h(
        Sum(H, d(B)), U, omega, V, eta
    )
    node = _ev(Sum(tw_form, Neg(target_form)), slots)
    eng = tangent_engine(registry=registry)
    eng.register(LieBracketLeibnizDefinition(registry))
    eng.register(MultiEvalArgLinearityDefinition(registry))
    eng.register(InteriorVectorLinearityDefinition(registry))
    eng.register(ActExpansionDefinition(registry))
    eng.register(HeadFormProductLiftDefinition(registry))
    eng.register(OperatorSlotAdditivityDefinition())
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )


def prove_pi_twist_h_form(
    N,
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Ψ_Π on the H-twisted Dorfman bracket, form component
    [eq (7.24)-(7.25) with H]: the twist shifts the Nambu double's
    form part by the H-term of the TWISTED anchors,

        form([e₁,e₂]_{Ψ_Π, H}) = form_Nambu + H(U + Πω, V + Πη).

    Declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify

    _, tw_form = twisted_dorfman_h(
        H,
        lambda v, f: pi_twist(N, v, f),
        lambda v, f: pi_twist_inverse(N, v, f),
        U, omega, V, eta,
    )
    _, nb_form = nambu_double(N, U, omega, V, eta)
    shift = h_term(
        H,
        Sum(U, N.sharp_vf(omega)),
        Sum(V, N.sharp_vf(eta)),
    )
    node = _ev(Sum(tw_form, Neg(nb_form), Neg(shift)), slots)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


def prove_pi_twist_h_vec(
    N,
    H: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Ψ_Π on the H-twisted Dorfman bracket, vector component: the
    form-side H-shift feeds back through ``Ψ⁻¹`` as ``−Π(H-term)``,

        vec([e₁,e₂]_{Ψ_Π, H}) = vec_Nambu + R′(ω,η)
                                − Π( H(U + Πω, V + Πη) ).

    Acting on ``h``; declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify

    tw_vec, _ = twisted_dorfman_h(
        H,
        lambda v, f: pi_twist(N, v, f),
        lambda v, f: pi_twist_inverse(N, v, f),
        U, omega, V, eta,
    )
    nb_vec, _ = nambu_double(N, U, omega, V, eta)
    shift = Neg(
        N.sharp_vf(
            h_term(
                H,
                Sum(U, N.sharp_vf(omega)),
                Sum(V, N.sharp_vf(eta)),
            )
        )
    )
    node = Act(
        Sum(
            tw_vec,
            Neg(nb_vec),
            Neg(r_twist(N, omega, eta)),
            Neg(shift),
        ),
        h,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_twist_engine(N, registry),
        max_steps=max_steps,
    )


class FormProductToWedgeDefinition(Definition):
    """``α · β → α ∧ β`` for a bare :class:`Product` with at least
    two determinable form-degree (≥ 1) factors — scoped to the twist
    engine (the head-position case is central
    :class:`HeadFormProductLiftDefinition`; here the shapes sit
    INSIDE sharp slots, where Leibniz splits leave ``d(f)·x``
    products). Scalar (degree-0) factors stay in front."""

    name = "form product is a wedge: α · β = α ∧ β (slot-reachable)"

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        from jacopy.core.expr import Product

        self._registry = registry
        self.anchor = Product

    def _split(self, expr: Expr):
        from jacopy.algebra.derivation import degree_of
        from jacopy.core.expr import Product

        if not (
            isinstance(expr, Product) and len(expr.children) >= 2
        ):
            return None
        scalars, formfs = [], []
        for c in expr.children:
            if hasattr(c, "wedge_degree"):
                return None
            try:
                k = degree_of(c, self._registry).as_int()
            except ValueError:
                return None
            if k is None:
                return None
            (scalars if k == 0 else formfs).append(c)
        if len(formfs) < 2:
            return None
        return scalars, formfs

    def matches(self, expr: Expr) -> bool:
        return self._split(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Product
        from jacopy.core.wedge import Wedge

        scalars, formfs = self._split(expr)
        wedge = Wedge(*formfs)
        if not scalars:
            return wedge
        return Product(*scalars, wedge)


class WedgeSumScalarDefinition(Definition):
    """Wedge slot normalization, scoped to the twist engine: a
    ``Sum`` factor distributes (``α ∧ (x + y) = α∧x + α∧y``) and a
    degree-0 factor pulls out as a scalar coefficient (``α ∧ f =
    f·α`` — the wedge with a 0-form IS scalar multiplication). The
    algorithm passes do this in visible trees; inside sharp slots
    only Definitions fire."""

    name = "wedge normalization: Sum distributes, 0-forms pull out"

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        from jacopy.core.wedge import Wedge

        self._registry = registry
        self.anchor = Wedge

    def _site(self, expr: Expr):
        from jacopy.algebra.derivation import degree_of

        for i, c in enumerate(expr.children):
            if isinstance(c, Sum):
                return i, "sum"
            if hasattr(c, "wedge_degree"):
                continue
            try:
                k = degree_of(c, self._registry).as_int()
            except ValueError:
                continue
            if k == 0:
                return i, "scalar"
        return None

    def matches(self, expr: Expr) -> bool:
        from jacopy.core.wedge import Wedge

        return (
            isinstance(expr, Wedge)
            and len(expr.children) >= 2
            and self._site(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Product
        from jacopy.core.wedge import Wedge

        i, kind = self._site(expr)
        kids = list(expr.children)
        if kind == "sum":
            return Sum(
                *(
                    Wedge(*(kids[:i] + [t] + kids[i + 1 :]))
                    for t in kids[i].children
                )
            )
        scalar = kids[i]
        rest = kids[:i] + kids[i + 1 :]
        core = rest[0] if len(rest) == 1 else Wedge(*rest)
        return Product(scalar, core)
