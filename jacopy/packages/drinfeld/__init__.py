"""Calculus on algebroids & Drinfel'd doubles (PDF item 13; Phase 6).

Primary sources: arXiv:2312.06584 (untwisted bialgebroids) and
arXiv:2409.11973 (twists, general Roytenberg bracket)."""

from jacopy.packages.drinfeld.poisson_bialgebroid import (
    kappa_cartan,
    kappa_tilde,
    lie_tilde_vf,
    prove_compat_condition_one,
    prove_compat_condition_three_p1,
    prove_compat_condition_two,
)
from jacopy.packages.drinfeld.calculus_conditions import (
    kappa,
    prove_calculus_condition_one,
    prove_calculus_condition_three,
    prove_calculus_condition_two,
    prove_kappa_bracket,
    prove_kappa_kappa,
)

__all__ = [
    "kappa_cartan",
    "kappa_tilde",
    "lie_tilde_vf",
    "prove_compat_condition_one",
    "prove_compat_condition_three_p1",
    "prove_compat_condition_two",
    "kappa",
    "prove_calculus_condition_one",
    "prove_calculus_condition_two",
    "prove_calculus_condition_three",
    "prove_kappa_bracket",
    "prove_kappa_kappa",
]
