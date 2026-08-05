"""
The Drinfel'd double bracket on ``E = A ⊕ Z`` (Phase 6.C) —
drinfeld paper [arXiv:2312.06584] eqs (4.11), (4.17); PDF items 13
and 14 (the Dorfman bracket is BORN here as the standard double).

Sections of ``E = TM ⊕ T*M`` are represented component-wise as pairs
``(U, ω)``; the doubled bracket [eq (4.17)] splits into a vector and
a form component. Two instantiations:

* **Standard double** (zero tilde side — the trivial Lie-bialgebroid
  structure on ``T*M``): the form component is

      form([U+ω, V+η]) = ℒ_U η − ℒ_V ω + dι_V ω,

  i.e. exactly the **standard Dorfman bracket**.
* **Poisson double** (tilde side induced by a Poisson ``π`` — the
  triangular/LWX case): adds ``ℒ̃_ω V − ℒ̃_η U + d̃ι̃_η U`` to the
  vector component and the Koszul bracket ``[ω,η]_π`` to the form
  component.

Mechanical theorems (standard double):

* **right-Leibniz**: ``[e₁, f·e₂] = f[e₁,e₂] + (ρ(e₁)f)·e₂`` with
  ``ρ(U+ω) = U`` — component-wise.
* **exact symmetric part**: ``[e₁,e₂] + [e₂,e₁] =
  (0, d(ι_Uη + ι_Vω))`` — the Dorfman symmetric part is the
  differential of the canonical pairing.
* **Leibniz-Jacobi**: ``[e₁,[e₂,e₃]] = [[e₁,e₂],e₃] + [e₂,[e₁,e₃]]``
  — the vector component is the VF Jacobi; the form component closes
  through the bracket-identity repair loop (92 steps, 3 cited
  VF-Jacobi instances).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
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


def _L(X: Expr, x: Expr) -> Expr:
    return Act(CARTAN_TM.lie(X), x)


def _iota(X: Expr, x: Expr) -> Expr:
    return Act(Interior(X), x)


def dorfman_double(
    U: Expr, omega: Expr, V: Expr, eta: Expr
) -> Tuple[Expr, Expr]:
    """The STANDARD double ``[U+ω, V+η]`` on ``TM ⊕ T*M`` as a
    ``(vector, form)`` component pair — the Dorfman bracket:

        ( [U,V]_Lie ,  ℒ_U η − ℒ_V ω + dι_V ω )."""
    return (
        lie_bracket(U, V),
        Sum(_L(U, eta), Neg(_L(V, omega)), d(_iota(V, omega))),
    )


def poisson_double(
    P, U: Expr, omega: Expr, V: Expr, eta: Expr
) -> Tuple[Expr, Expr]:
    """The POISSON (triangular/LWX) double [eq (4.17) with the
    π-induced tilde calculus]:

        vector: [U,V] + ℒ̃_ω V − ℒ̃_η U + d̃ι̃_η U,
        form:   [ω,η]_π + ℒ_U η − ℒ_V ω + dι_V ω,

    with ``ℒ̃_ω V = [π♯ω, V] + π♯(ι_V dω)`` and ``d̃ = −π♯d``."""
    from jacopy.packages.drinfeld.poisson_bialgebroid import (
        lie_tilde_vf,
    )
    from jacopy.packages.poisson.core import SharpVF
    from jacopy.packages.poisson.koszul import KoszulBracket

    dtilde_iota = Neg(SharpVF(P.pi, d(Pairing(eta, U))))
    vec = Sum(
        lie_bracket(U, V),
        lie_tilde_vf(P, omega, V),
        Neg(lie_tilde_vf(P, eta, U)),
        dtilde_iota,
    )
    form = Sum(
        KoszulBracket(P.pi, omega, eta),
        _L(U, eta),
        Neg(_L(V, omega)),
        d(_iota(V, omega)),
    )
    return vec, form


def canonical_pairing(
    U: Expr, omega: Expr, V: Expr, eta: Expr
) -> Expr:
    """``⟨U+ω, V+η⟩₊ = ι_U η + ι_V ω`` — the canonical symmetric
    pairing of the double."""
    return Sum(_iota(U, eta), _iota(V, omega))


def _ev(expr: Expr, slots) -> Expr:
    from jacopy.core.multi_eval import MultiEval

    if len(slots) == 1:
        return Pairing(expr, slots[0])
    return MultiEval(
        expr, *slots, alternating=True, slot_kind="vector"
    )


def _double_engine(registry):
    """Tangent engine + the theorem-classified Lie-bracket Leibniz
    and the MultiEval argument linearity (degree ≥ 2 form slots need
    both — the double is DEGREE-GENERAL, verified up to p = 7)."""
    from jacopy.central.calculus import (
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )

    eng = tangent_engine(registry=registry)
    eng.register(LieBracketLeibnizDefinition(registry))
    eng.register(MultiEvalArgLinearityDefinition(registry))
    return eng


def prove_dorfman_right_leibniz_form(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, List[Theorem]]:
    """Form component of ``[e₁, f·e₂] = f[e₁,e₂] + (Uf)·e₂``
    (``ρ(U+ω) = U`` — the anchor of the double is the projection),
    evaluated on the given vector slots. Degree-general: a p-form
    section takes p slots (verified up to p = 7)."""
    _, lhs_form = dorfman_double(
        U, omega, Product(f, V), Product(f, eta)
    )
    _, base_form = dorfman_double(U, omega, V, eta)
    rhs_form = Sum(
        Product(f, base_form), Product(Act(U, f), eta)
    )
    node = _ev(Sum(lhs_form, Neg(rhs_form)), slots)
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=_double_engine(registry),
        max_steps=max_steps,
    )


def prove_dorfman_symmetric_part(
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, List[Theorem]]:
    """``[e₁,e₂] + [e₂,e₁] = (0, d⟨e₁,e₂⟩₊)`` — form component
    evaluated on the given slots (the vector component is the Lie
    bracket's antisymmetry). Degree-general; verified up to p = 7,
    composite vector inputs included."""
    _, f12 = dorfman_double(U, omega, V, eta)
    _, f21 = dorfman_double(V, eta, U, omega)
    node = _ev(
        Sum(
            f12,
            f21,
            Neg(d(canonical_pairing(U, omega, V, eta))),
        ),
        slots,
    )
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=_double_engine(registry),
        max_steps=max_steps,
    )


def prove_dorfman_jacobi_form(
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
    """Form component of the Leibniz-Jacobi identity of the standard
    Dorfman bracket,

        [e₁,[e₂,e₃]] = [[e₁,e₂],e₃] + [e₂,[e₁,e₃]],

    evaluated on the given slots (the vector component is the VF
    Jacobi, entering as cited instances via the repair loop).
    Degree-general: closes at p = 7 too (2483 steps, 119 cited
    VF-Jacobi instances, ~4 min — keep heavy degrees out of default
    test suites)."""

    def nest(U1, o1, U2, o2, U3, o3):
        iv, if_ = dorfman_double(U2, o2, U3, o3)
        return dorfman_double(U1, o1, iv, if_)[1]

    v12, f12 = dorfman_double(U, omega, V, eta)
    lhs = Sum(
        nest(U, omega, V, eta, W, mu),
        Neg(dorfman_double(v12, f12, W, mu)[1]),
        Neg(nest(V, eta, U, omega, W, mu)),
    )
    node = _ev(lhs, slots)
    return prove_with_bracket_identities(
        node,
        Integer(0),
        f,
        registry=registry,
        engine=_double_engine(registry),
        max_steps=max_steps,
    )


def _poisson_double_engine(P, registry):
    """Showcase (Koszul/sharp) engine + Lie-bracket Leibniz +
    MultiEval argument linearity + interior vector-slot linearity —
    the 6.D layer for the POISSON double's theorems. No ``[π,π]``
    declaration: the properties below hold for ANY bivector."""
    from jacopy.central.calculus import (
        InteriorVectorLinearityDefinition,
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )
    from jacopy.packages.poisson.showcase import showcase_engine

    eng = showcase_engine(P, registry=registry, declare_poisson=False)
    eng.register(LieBracketLeibnizDefinition(registry))
    eng.register(MultiEvalArgLinearityDefinition(registry))
    eng.register(InteriorVectorLinearityDefinition(registry))
    return eng


def poisson_double_anchor_action(P, U: Expr, omega: Expr, f: Expr) -> Expr:
    """``ρ(U+ω)(f) = U(f) + (π♯ω)(f)`` — the anchor of the LWX double
    is the SUM of the two anchors [eq (4.19)]."""
    from jacopy.packages.poisson.core import SharpVF

    return Sum(Act(U, f), Act(SharpVF(P.pi, omega), f))


def prove_poisson_double_symmetric_part_form(
    P,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Form component of ``[e₁,e₂] + [e₂,e₁] = 𝒟⟨e₁,e₂⟩₊`` for the
    POISSON double, with ``𝒟 = d + d̃``: the form part of ``𝒟`` is
    the usual ``d`` of the canonical pairing (the Koszul bracket's
    symmetric part cancels — no ``[π,π]`` declaration needed)."""
    from jacopy.proof.strategies import ExpandAndSimplify

    _, f12 = poisson_double(P, U, omega, V, eta)
    _, f21 = poisson_double(P, V, eta, U, omega)
    node = _ev(
        Sum(f12, f21, Neg(d(canonical_pairing(U, omega, V, eta)))),
        slots,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_poisson_double_engine(P, registry),
        max_steps=max_steps,
    )


def prove_poisson_double_symmetric_part_vec(
    P,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    gamma: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Vector component of the same identity: the vector part of
    ``𝒟⟨e₁,e₂⟩₊`` is ``d̃⟨e₁,e₂⟩₊ = −π♯d⟨e₁,e₂⟩₊``. Tested by
    pairing against the probe 1-form ``γ``."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.poisson.core import SharpVF

    v12, _ = poisson_double(P, U, omega, V, eta)
    v21, _ = poisson_double(P, V, eta, U, omega)
    dtilde = Neg(
        SharpVF(P.pi, d(canonical_pairing(U, omega, V, eta)))
    )
    node = Pairing(gamma, Sum(v12, v21, Neg(dtilde)))
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_poisson_double_engine(P, registry),
        max_steps=max_steps,
    )


def prove_poisson_double_right_leibniz_form(
    P,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    f: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Form component of ``[e₁, f·e₂] = f[e₁,e₂] + (ρ(e₁)f)·e₂``
    for the POISSON double, ``ρ(U+ω) = U + π♯ω``. Holds for ANY
    bivector π (no declaration)."""
    from jacopy.proof.strategies import ExpandAndSimplify

    _, lhs_form = poisson_double(
        P, U, omega, Product(f, V), Product(f, eta)
    )
    _, base_form = poisson_double(P, U, omega, V, eta)
    rhs_form = Sum(
        Product(f, base_form),
        Product(poisson_double_anchor_action(P, U, omega, f), eta),
    )
    node = _ev(Sum(lhs_form, Neg(rhs_form)), slots)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_poisson_double_engine(P, registry),
        max_steps=max_steps,
    )


def prove_poisson_double_right_leibniz_vec(
    P,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    f: Expr,
    gamma: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """Vector component of the same right-Leibniz rule, paired
    against the probe 1-form ``γ`` (needed the interior vector-slot
    linearity: ``ι_{f·V}(dω) = f·ι_V(dω)``)."""
    from jacopy.proof.strategies import ExpandAndSimplify

    lhs_vec, _ = poisson_double(
        P, U, omega, Product(f, V), Product(f, eta)
    )
    base_vec, _ = poisson_double(P, U, omega, V, eta)
    rhs_vec = Sum(
        Product(f, base_vec),
        Product(poisson_double_anchor_action(P, U, omega, f), V),
    )
    node = Pairing(gamma, Sum(lhs_vec, Neg(rhs_vec)))
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_poisson_double_engine(P, registry),
        max_steps=max_steps,
    )
