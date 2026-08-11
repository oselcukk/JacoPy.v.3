"""
Bracket-morphism compatibility (Phase 6.I.2) — paper eqs
(4.18)-(4.20), concretely for the natural map

    φ = φ_A ⊕ φ_Z = id ⊕ Π :  E = TM ⊕ Λᵖ T*M → TM,

i.e. the TOTAL ANCHOR ``ρ(U+ω) = U + Πω`` of the Nambu double.
Theorems:

* **(4.20)**: ``φ_A(𝒦̃_η U) + φ_Z(ℒ_U η) = [φ_A U, φ_Z η]`` —
  concretely ``𝒦̃_η U + Π(ℒ_U η) = [U, Πη]``: the compatibility
  condition IS the ``𝒦̃`` definition face. Declaration-free.
* **(4.20)-dual**: ``ℒ̃_ω V + Π(𝒦_V ω) = [Πω, V]`` — via the magic
  formula. Declaration-free.
* **φ_Z morphism**: ``Π[ω,η]_Kos = [Πω, Πη]`` — the declared FI
  (honest-fail without).
* **(4.19) — the capstone**: the total anchor is a BRACKET MORPHISM
  of the double,

      ρ([e₁, e₂]_double) = [ρ(e₁), ρ(e₂)]_Lie,

  under the declared FI (honest-fail without) — the double-level
  generalization of the Phase 5 Hamiltonian anchor-morphism.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.tangent.exterior import CARTAN_TM
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.drinfeld.calculus_conditions import kappa
from jacopy.packages.drinfeld.double import (
    lie_tilde_nambu,
    nambu_double,
)
from jacopy.packages.drinfeld.tilde_calculus import (
    _tilde_engine,
    kappa_tilde_nambu,
)
from jacopy.packages.poisson.nambu import nambu_koszul_bracket


def _L(X: Expr, x: Expr) -> Expr:
    return Act(CARTAN_TM.lie(X), x)


def _prove(N, node, registry, declare_fi, max_steps):
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=declare_fi),
        max_steps=max_steps,
    )


def prove_morphism_compat_condition(
    N,
    U: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """(4.20): ``𝒦̃_η U + Π(ℒ_U η) = [U, Πη]`` — the bracket
    morphism compatibility condition for ``φ = id ⊕ Π`` is the
    ``𝒦̃`` definition face. Declaration-free; probe ``h``."""
    node = Act(
        Sum(
            kappa_tilde_nambu(N, eta, U),
            N.sharp_vf(_L(U, eta)),
            Neg(lie_bracket(U, N.sharp_vf(eta))),
        ),
        h,
    )
    return _prove(N, node, registry, False, max_steps)


def prove_morphism_compat_condition_dual(
    N,
    omega: Expr,
    V: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """(4.20)-dual: ``ℒ̃_ω V + Π(𝒦_V ω) = [Πω, V]`` — closes via
    the magic formula. Declaration-free; probe ``h``."""
    node = Act(
        Sum(
            lie_tilde_nambu(N, omega, V),
            N.sharp_vf(kappa(V, omega)),
            Neg(lie_bracket(N.sharp_vf(omega), V)),
        ),
        h,
    )
    return _prove(N, node, registry, False, max_steps)


def prove_phi_z_is_morphism(
    N,
    omega: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_fi: bool = True,
    max_steps: int = 30000,
) -> ProofChain:
    """``Π[ω,η]_Kos = [Πω, Πη]`` — φ_Z = Π is a bracket morphism
    exactly under the declared FI (honest-fail without). Probe
    ``h``."""
    node = Act(
        Sum(
            N.sharp_vf(nambu_koszul_bracket(N, omega, eta)),
            Neg(
                lie_bracket(
                    N.sharp_vf(omega), N.sharp_vf(eta)
                )
            ),
        ),
        h,
    )
    return _prove(N, node, registry, declare_fi, max_steps)


def prove_total_anchor_is_bracket_morphism(
    N,
    U1: Expr,
    om1: Expr,
    U2: Expr,
    om2: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_fi: bool = True,
    max_steps: int = 60000,
) -> ProofChain:
    """(4.19), THE capstone: the total anchor ``ρ(U+ω) = U + Πω``
    is a bracket morphism of the Nambu double,

        ρ([e₁, e₂]_double) = [ρ(e₁), ρ(e₂)]_Lie

    under the declared FI (honest-fail without) — the double-level
    generalization of the Phase 5 Hamiltonian anchor-morphism.
    Probe ``h``."""
    vec, form = nambu_double(N, U1, om1, U2, om2)
    lhs = Sum(vec, N.sharp_vf(form))
    rhs = lie_bracket(
        Sum(U1, N.sharp_vf(om1)),
        Sum(U2, N.sharp_vf(om2)),
    )
    node = Act(Sum(lhs, Neg(rhs)), h)
    return _prove(N, node, registry, declare_fi, max_steps)
