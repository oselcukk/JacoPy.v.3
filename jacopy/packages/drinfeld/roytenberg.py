"""
The general (8-component) Roytenberg bracket (Phase 6.F) —
[2409.11973 eq (5.2)], concretely instantiated on ``A = TM``,
``Z = Λᵖ T*M``:

    [U+ω, V+η]_E = [U,V]_A + ℒ̃_ω V + 𝒦̃_η U + R(ω,η)
                 ⊕ [ω,η]_Z + ℒ_U η + 𝒦_V ω + H(U,V),

with the tilde calculus induced by a ``(p+1)``-vector Π (6.D), the
H-twist ``H(U,V) = ι_V ι_U H`` of a ``(p+2)``-form and the R-twist
``R(ω,η) = [Πω,Πη] − Π[ω,η]_Kos`` (the 6.E ``r_twist`` — the
NATURAL nonzero R of a non-integrable Π). Decompositions (5.4)/(5.5)
into twisted Dorfman ⊕ twisted tilde-Dorfman are structural.

Mechanical theorems:

* **Twisted linearity (5.6)-(5.7)**: the concrete ``H`` is
  C∞-bilinear (interior tensoriality) — both symbols vanish; the
  R-twist is C∞ in the second entry (6.E) with the (7.26) anomaly in
  the first.
* **H-closure (5.13)**: THE new Jacobi requirement of the twisted
  case. The defect of (5.13) is EXACTLY the ``dH`` evaluation:

      H(U,[V,W]) − H([U,V],W) − H(V,[U,W])
        + ℒ_U H(V,W) − 𝒦_W H(U,V) − ℒ_V H(U,W) = −ι_W ι_V ι_U dH,

  declaration-free — so the condition holds iff ``dH`` dies on the
  triple; in particular ``H = dB`` (the Ševera image) satisfies it
  mechanically (``d² = 0``).
* **(5.16)**: the third twisted Jacobi compatibility — for the
  antisymmetric concrete ``H`` the ``g_H`` term vanishes and the
  condition reduces to the proven (D.10); re-exported here in its
  twisted reading.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.drinfeld.calculus_conditions import kappa
from jacopy.packages.drinfeld.double import lie_tilde_nambu
from jacopy.packages.drinfeld.tilde_calculus import (
    _tilde_engine,
    kappa_tilde_nambu,
)
from jacopy.packages.drinfeld.twist import h_term, r_twist
from jacopy.packages.poisson.nambu import nambu_koszul_bracket


def _L(X: Expr, x: Expr) -> Expr:
    return Act(CARTAN_TM.lie(X), x)


def roytenberg_bracket(
    N,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    H: Optional[Expr] = None,
    with_tilde: bool = True,
    with_r: bool = True,
) -> Tuple[Expr, Expr]:
    """The concrete 8-component Roytenberg bracket [eq (5.2)] as a
    ``(vector, form)`` pair. ``with_tilde``/``with_r``/``H`` switch
    the tilde side, the R-twist and the H-twist independently — all
    off reproduces the standard Dorfman double."""
    vec_terms = [lie_bracket(U, V)]
    form_terms = [
        _L(U, eta),
        kappa(V, omega),
    ]
    if with_tilde:
        vec_terms.append(lie_tilde_nambu(N, omega, V))
        vec_terms.append(kappa_tilde_nambu(N, eta, U))
        form_terms.insert(
            0, nambu_koszul_bracket(N, omega, eta)
        )
    if with_r:
        vec_terms.append(r_twist(N, omega, eta))
    if H is not None:
        form_terms.append(h_term(H, U, V))
    return Sum(*vec_terms), Sum(*form_terms)


def prove_h_bilinearity(
    H: Expr,
    U: Expr,
    V: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, ProofChain]:
    """(5.6)-(5.7) for the concrete ``H(U,V) = ι_V ι_U H``: the
    H-twist is C∞-linear in BOTH slots (interior tensoriality), so
    both symbol maps vanish — the twisted linearity conditions hold
    with zero anomaly. Returns the two chains (second slot, first
    slot), evaluated on the given vector slots."""
    from jacopy.core.expr import Product
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        InteriorVectorLinearityDefinition,
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.packages.drinfeld.double import _ev

    def eng():
        e = tangent_engine(registry=registry)
        e.register(InteriorVectorLinearityDefinition(registry))
        e.register(ActExpansionDefinition(registry))
        e.register(MultiEvalArgLinearityDefinition(registry))
        return e

    chains = []
    for lhs, rhs in (
        (
            h_term(H, U, Product(f, V)),
            Product(f, h_term(H, U, V)),
        ),
        (
            h_term(H, Product(f, U), V),
            Product(f, h_term(H, U, V)),
        ),
    ):
        node = _ev(Sum(lhs, Neg(rhs)), slots)
        chains.append(
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=registry,
                engine=eng(),
                max_steps=max_steps,
            )
        )
    return chains[0], chains[1]


def h_closure_defect(H: Expr, U: Expr, V: Expr, W: Expr) -> Expr:
    """The (5.13) defect,

        H(U,[V,W]) − H([U,V],W) − H(V,[U,W])
          + ℒ_U H(V,W) − 𝒦_W H(U,V) − ℒ_V H(U,W)."""
    return Sum(
        h_term(H, U, lie_bracket(V, W)),
        Neg(h_term(H, lie_bracket(U, V), W)),
        Neg(h_term(H, V, lie_bracket(U, W))),
        _L(U, h_term(H, V, W)),
        Neg(kappa(W, h_term(H, U, V))),
        Neg(_L(V, h_term(H, U, W))),
    )


def prove_h_closure_measures_dh(
    N,
    H: Expr,
    U: Expr,
    V: Expr,
    W: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """THE (5.13) theorem, declaration-free: the twisted-Jacobi
    H-condition measures exactly the closedness of ``H``,

        defect(5.13) = ι_W ι_V ι_U dH

    (sign found by the engine via a full ± sweep — the unique closing
    combination), evaluated on the given slots. Corollary (tested
    separately): ``H = dB`` — the Ševera image — satisfies (5.13) by
    ``d² = 0``."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.double import _ev

    node = _ev(
        Sum(
            h_closure_defect(H, U, V, W),
            Neg(
                Act(
                    Interior(W),
                    Act(Interior(V), Act(Interior(U), d(H))),
                )
            ),
        ),
        slots,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_h_closure_for_exact_h(
    N,
    B: Expr,
    U: Expr,
    V: Expr,
    W: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Corollary: for ``H = dB`` the (5.13) condition HOLDS
    mechanically (``d² = 0``) — the Ševera-twisted bracket keeps the
    Jacobi requirement. Declaration-free."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.double import _ev

    node = _ev(h_closure_defect(d(B), U, V, W), slots)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_h_twisted_jacobi_measures_dh(
    N,
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
):
    """THE 6.F.2 theorem — the classical "twisted Courant ⟺ dH = 0"
    mechanically: the form-component Leibniz-Jacobi defect of the
    H-TWISTED Dorfman bracket is exactly the ``dH`` evaluation,

        form( [e₁,[e₂,e₃]]_H − [[e₁,e₂]_H,e₃]_H − [e₂,[e₁,e₃]]_H )
            = ι_W ι_V ι_U dH

    (sign fixed by the engine, consistent with the (5.13) sweep),
    declaration-free (generalizes the 6.C Leibniz-Jacobi, which is
    the ``H = 0`` case; the vector components never see ``H``).
    Corollary: ``H = dB`` restores Jacobi (``d² = 0``) — the Ševera
    twist of a Courant structure is again Courant. Evaluated on the
    given slots; VF-Jacobi enters as cited instances."""
    from jacopy.central.tangent.cartan import (
        prove_with_bracket_identities,
    )
    from jacopy.packages.drinfeld.double import _ev
    from jacopy.packages.drinfeld.twist import dorfman_double_h

    def nest(U1, o1, U2, o2, U3, o3):
        iv, if_ = dorfman_double_h(H, U2, o2, U3, o3)
        return dorfman_double_h(H, U1, o1, iv, if_)[1]

    v12, f12 = dorfman_double_h(H, U, omega, V, eta)
    defect = Sum(
        nest(U, omega, V, eta, W, mu),
        Neg(dorfman_double_h(H, v12, f12, W, mu)[1]),
        Neg(nest(V, eta, U, omega, W, mu)),
        Neg(
            Act(
                Interior(W),
                Act(Interior(V), Act(Interior(U), d(H))),
            )
        ),
    )
    node = _ev(defect, slots)
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_fi_collapses_twisted_compat(
    N,
    H: Expr,
    U: Expr,
    eta: Expr,
    mu: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """The twisted Jacobi compatibility (5.14) differs from the
    proven untwisted (D.8) by the single term ``−H(U, R(η,μ))``.
    Under the declared FI the R-twist dies INSIDE the H-slot
    (the Interior slot protocol carries the node-face declaration
    into ``ι_{R(η,μ)}``), so the twisted family COLLAPSES to the
    untwisted one:

        H(U, R(η,μ)) = 0        (FI declared).

    Evaluated on the given slots."""
    from jacopy.proof.strategies import ExpandAndSimplify

    node = _ev_forms(
        h_term(H, U, r_twist(N, eta, mu)), slots
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=True),
        max_steps=max_steps,
    )


def _ev_forms(expr: Expr, slots) -> Expr:
    from jacopy.packages.drinfeld.double import _ev

    return _ev(expr, slots)


def twisted_calculus_obstruction_one(
    N, H: Expr, U: Expr, V: Expr, mu: Expr
) -> Expr:
    """The RHS of the first twisted calculus condition (5.9) for the
    concrete data — the obstruction that must vanish for the twisted
    triplet to remain a calculus:

        −H(U, 𝒦̃_μ V) + H(V, 𝒦̃_μ U) + [H(U,V), μ]_Kos.

    (The LHS is the usual C.1 commutator, zero by 6.A.) Generic
    ``(Π, H)`` do NOT satisfy it — pinned by an honest-fail test."""
    return Sum(
        Neg(h_term(H, U, kappa_tilde_nambu(N, mu, V))),
        h_term(H, V, kappa_tilde_nambu(N, mu, U)),
        nambu_koszul_bracket(N, h_term(H, U, V), mu),
    )


# ------------------------------------------------------------------- #
# 6.F.3 — proto bialgebroid master equations, bosonic dictionary       #
# ------------------------------------------------------------------- #
#
# The graded master equation {Θ,Θ} = 0 with Θ = μ + γ + φ + ψ splits
# into (3.11)-(3.15). Bosonic reading for the concrete data
# (A = TM: μ = Lie/de Rham; γ = Π; φ = H; ψ = 0 — quasi-Lie case):
#
#   (3.14) {μ,φ} = 0        ⟺  dH = 0        — measured EXACTLY by
#          the (5.13)/Jacobi theorems above (defect = ιιι dH);
#   (3.11) ½{μ,μ} + {γ,φ}   ⟺  the H-TWISTED fundamental identity:
#          [Πω,Πη] = Π[ω,η]_Kos + Π(H(Πω,Πη))  (twisted-Poisson);
#          consumed as the opt-in TwistedFIDeclaration below;
#   (3.12)/(3.13)/(3.15) need a free trivector ψ (an R-side 3-tensor
#          independent of Π) — deferred until a consumer arrives
#          (definition policy; our R is DERIVED: r_twist).


def r_twist_h(N, H: Expr, omega: Expr, eta: Expr) -> Expr:
    """The H-corrected R-twist [eq (7.25) with H]:

        R′_H(ω,η) = [Πω,Πη] − Π[ω,η]_Kos − Π( H(Πω, Πη) ).

    Its vanishing is the H-TWISTED fundamental identity — the
    bosonic (3.11)."""
    return Sum(
        r_twist(N, omega, eta),
        Neg(
            N.sharp_vf(
                h_term(H, N.sharp_vf(omega), N.sharp_vf(eta))
            )
        ),
    )


class TwistedFIDeclaration(Definition):
    """DECLARED H-twisted fundamental identity (bosonic (3.11), the
    twisted-Poisson condition):

        [Πω, Πη]_Lie → Π[ω,η]_Kos + Π( ι_{Πη} ι_{Πω} H ).

    Opt-in; reduces to :class:`NambuMorphismDeclaration` at H = 0."""

    anchor = None  # set in __init__ (LieBracketVF)

    def __init__(self, structure, H: Expr) -> None:
        from jacopy.algebra.lie_bracket_vf import LieBracketVF
        from jacopy.packages.poisson.nambu import NambuSharpVF

        self._N = structure
        self._H = H
        self._VF = LieBracketVF
        self._Sharp = NambuSharpVF
        self.anchor = LieBracketVF
        self.name = (
            "declared H-twisted FI: [Πω,Πη] = Π[ω,η]_Kos + Π H(Πω,Πη)"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, self._VF):
            return False
        X, Y = expr.X, expr.Y
        return (
            isinstance(X, self._Sharp)
            and isinstance(Y, self._Sharp)
            and X.pi == self._N.pi
            and Y.pi == self._N.pi
        )

    def rewrite(self, expr: Expr) -> Expr:
        om, et = expr.X.omega, expr.Y.omega
        return Sum(
            self._N.sharp_vf(
                nambu_koszul_bracket(self._N, om, et)
            ),
            self._N.sharp_vf(
                h_term(
                    self._H,
                    self._N.sharp_vf(om),
                    self._N.sharp_vf(et),
                )
            ),
        )


def prove_twisted_fi_kills_r_twist_h(
    N,
    H: Expr,
    omega: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Consistency of the bosonic (3.11): under the declared
    H-twisted FI the corrected R-twist ``R′_H`` vanishes — acting on
    the probe ``h``. (At H = 0 this is the 6.E statement "FI kills
    the R-twist".)"""
    from jacopy.proof.strategies import ExpandAndSimplify

    from jacopy.packages.drinfeld.tilde_calculus import (
        FISharpPairingSwapDefinition,
    )

    eng = _tilde_engine(N, registry, declare_fi=False)
    eng.register(TwistedFIDeclaration(N, H))
    # The pairing-swap consequence holds under the TWISTED FI too:
    # the H-parts are antisymmetric and cancel in the symmetric
    # combination, leaving Π(d g_Z) = 0 unchanged.
    eng.register(FISharpPairingSwapDefinition(N))
    node = Act(r_twist_h(N, H, omega, eta), h)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )


def prove_r_twist_h_correction_bilinear(
    N,
    H: Expr,
    omega: Expr,
    eta: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, ProofChain]:
    """The H-correction ``Π H(Πω, Πη)`` of ``R′_H`` is C∞-BILINEAR
    (sharp + interior tensoriality) — hence the first-entry
    linearity anomaly of ``R′_H`` is exactly the H-INDEPENDENT
    (7.26) anomaly ``−Π(df ∧ g_Z)``. Returns the two chains (first
    slot, second slot); probe ``h``."""
    from jacopy.core.expr import Product
    from jacopy.proof.strategies import ExpandAndSimplify

    def corr(a, b):
        return N.sharp_vf(
            h_term(H, N.sharp_vf(a), N.sharp_vf(b))
        )

    chains = []
    for lhs, rhs in (
        (
            corr(Product(f, omega), eta),
            Product(f, corr(omega, eta)),
        ),
        (
            corr(omega, Product(f, eta)),
            Product(f, corr(omega, eta)),
        ),
    ):
        node = Act(Sum(lhs, Neg(rhs)), h)
        chains.append(
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=registry,
                engine=_tilde_engine(
                    N, registry, declare_fi=False
                ),
                max_steps=max_steps,
            )
        )
    return chains[0], chains[1]


def jacobiator_a_obstruction(
    N, H: Expr, U: Expr, V: Expr, W: Expr
) -> Expr:
    """The RHS of the twisted Jacobiator requirement (5.12) for the
    concrete data (the A-bracket is Lie, so its Jacobiator vanishes
    and the requirement forces this combination to vanish):

        −𝒦̃_{H(V,W)} U + ℒ̃_{H(U,V)} W + 𝒦̃_{H(U,W)} V.

    Generic ``(Π, H)`` do NOT satisfy it (honest-fail pinned) —
    mixing the H-twist into the tilde slots is a genuine
    constraint."""
    return Sum(
        Neg(kappa_tilde_nambu(N, h_term(H, V, W), U)),
        lie_tilde_nambu(N, h_term(H, U, V), W),
        kappa_tilde_nambu(N, h_term(H, U, W), V),
    )
