"""
The calculus conditions and the 𝒦-operator (Phase 6.A) — PDF item
13a; drinfeld paper [arXiv:2312.06584 §4, eqs (4.14), (4.32)-(4.34),
(4.4)-(4.5)].

A **calculus on Z induced by A** is a triplet ``(ℒ, ι, d)`` obeying
three CALCULUS CONDITIONS; the associated second action is

    𝒦_V := −ℒ_V + d ∘ ι_V                     [eq (4.14)],

and a calculus pair with its dual is what a bialgebroid carries
(Phase 6.B). The USUAL Cartan calculus (A = TM, Z = Λ^p T*M) is the
prototype instantiation [App A] — this module proves, mechanically
and for concrete p, that it IS one:

* **condition 1**: ``ℒ_U ℒ_V μ − ℒ_V ℒ_U μ − ℒ_{[U,V]} μ = 0``,
* **condition 2**: ``ℒ_U dι_W η − dι_{[U,W]} η − dι_W ℒ_U η = 0``,
* **condition 3**: ``ℒ_W dι_V ω − dι_W dι_V ω = 0``,

plus the 𝒦-composition laws [eqs (4.4)-(4.5)]:

* ``𝒦_U 𝒦_V ω = −𝒦_U ℒ_V ω``,
* ``𝒦_{[U,V]} ω = 𝒦_V 𝒦_U ω + ℒ_U 𝒦_V ω``.

All proofs run through the bracket-identity repair loop (cited
VF-Jacobi instances enter as needed — 0-4 per statement at
p = 1, 2)."""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.cartan import (
    prove_with_bracket_identities,
)
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.lie_bracket import lie_bracket


def lie(X: Expr, x: Expr) -> Expr:
    """``ℒ_X x`` in the TM instantiation."""
    return Act(CARTAN_TM.lie(X), x)


def iota(X: Expr, x: Expr) -> Expr:
    """``ι_X x``."""
    return Act(Interior(X), x)


def kappa(X: Expr, x: Expr) -> Expr:
    """``𝒦_X x := −ℒ_X x + d(ι_X x)`` [drinfeld eq (4.14)] — the
    calculus's second action (for the usual Cartan calculus
    ``𝒦_X = −ι_X ∘ d`` by the magic formula, itself a theorem)."""
    return Sum(Neg(lie(X, x)), d(iota(X, x)))


def _ev(expr: Expr, slots) -> Expr:
    if len(slots) == 1:
        return Pairing(expr, slots[0])
    return MultiEval(
        expr, *slots, alternating=True, slot_kind="vector"
    )


def prove_calculus_condition_one(
    U: Expr,
    V: Expr,
    mu: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """``ℒ_U ℒ_V μ = ℒ_V ℒ_U μ + ℒ_{[U,V]} μ`` evaluated on the
    given slots (cited VF-Jacobi instances via the repair loop)."""
    node = _ev(
        Sum(
            lie(U, lie(V, mu)),
            Neg(lie(V, lie(U, mu))),
            Neg(Act(CARTAN_TM.lie(lie_bracket(U, V)), mu)),
        ),
        slots,
    )
    return prove_with_bracket_identities(
        node, Integer(0), f, registry=registry
    )


def prove_calculus_condition_two(
    U: Expr,
    W: Expr,
    eta: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """``ℒ_U dι_W η = dι_{[U,W]} η + dι_W ℒ_U η`` evaluated."""
    node = _ev(
        Sum(
            lie(U, d(iota(W, eta))),
            Neg(d(iota(lie_bracket(U, W), eta))),
            Neg(d(iota(W, lie(U, eta)))),
        ),
        slots,
    )
    return prove_with_bracket_identities(
        node, Integer(0), f, registry=registry
    )


def prove_calculus_condition_three(
    W: Expr,
    V: Expr,
    omega: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """``ℒ_W dι_V ω = dι_W dι_V ω`` evaluated (the ``d²``-flavoured
    condition: via magic the difference is ``ι_W d(dι_V ω)``)."""
    node = _ev(
        Sum(
            lie(W, d(iota(V, omega))),
            Neg(d(iota(W, d(iota(V, omega))))),
        ),
        slots,
    )
    return prove_with_bracket_identities(
        node, Integer(0), f, registry=registry
    )


def prove_kappa_kappa(
    U: Expr,
    V: Expr,
    omega: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """[drinfeld eq (4.5)]: ``𝒦_U 𝒦_V ω = −𝒦_U ℒ_V ω``."""
    node = _ev(
        Sum(kappa(U, kappa(V, omega)), kappa(U, lie(V, omega))),
        slots,
    )
    return prove_with_bracket_identities(
        node, Integer(0), f, registry=registry
    )


def prove_kappa_bracket(
    U: Expr,
    V: Expr,
    omega: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """[drinfeld eq (4.5)]:
    ``𝒦_{[U,V]} ω = 𝒦_V 𝒦_U ω + ℒ_U 𝒦_V ω``."""
    node = _ev(
        Sum(
            kappa(lie_bracket(U, V), omega),
            Neg(kappa(V, kappa(U, omega))),
            Neg(lie(U, kappa(V, omega))),
        ),
        slots,
    )
    return prove_with_bracket_identities(
        node, Integer(0), f, registry=registry
    )
