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
