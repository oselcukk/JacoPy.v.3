"""
Index antisymmetrisation (research interface, Phase 8 brick 5): the
bracket notation of the papers,

    T^{[a₁…a_n]} := (1/n!) Σ_σ sign(σ) T^{a_{σ(1)}…a_{σ(n)}},

as an operation on frame-component expressions. Renaming is done
through the frame layer's own index-substitution protocol
(``substitute_atom`` with :class:`FrameIndex` atoms), simultaneously —
so ``(a₁ a₂) → (a₂ a₁)`` does not collide.
"""

from __future__ import annotations

from itertools import permutations
from math import factorial
from typing import Dict, Sequence

from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum


def permutation_sign(perm: Sequence[int]) -> int:
    """``+1`` / ``−1`` for a permutation given as a tuple of positions."""
    perm = list(perm)
    sign = 1
    for i in range(len(perm)):
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            sign = -sign
    return sign


def rename_indices(expr: Expr, mapping: Dict[str, str]) -> Expr:
    """Simultaneous renaming of frame-index names."""
    from jacopy.central.objects.frame import FrameIndex

    temps = {a: f"§{k}§" for k, a in enumerate(mapping)}
    out = expr
    for a, t in temps.items():
        out = out.substitute_atom(FrameIndex(a), FrameIndex(t))
    for a, t in temps.items():
        out = out.substitute_atom(FrameIndex(t), FrameIndex(mapping[a]))
    return out


def antisymmetrize(
    expr: Expr, names: Sequence[str], *, unit_weight: bool = False
) -> Expr:
    """``expr^{[names]}``: the signed sum over all permutations of the
    named indices, divided by ``n!`` unless ``unit_weight=True``."""
    names = list(names)
    n = len(names)
    if n < 2:
        return expr
    terms = []
    for perm in permutations(range(n)):
        renamed = rename_indices(
            expr, {names[i]: names[perm[i]] for i in range(n)}
        )
        terms.append(
            renamed if permutation_sign(perm) > 0 else Neg(renamed)
        )
    total = Sum(*terms)
    if unit_weight:
        return total
    return Product(Rational(1, factorial(n)), total)


def swap_indices(expr: Expr, a: str, b: str) -> Expr:
    """Transposition ``a ↔ b``."""
    return rename_indices(expr, {a: b, b: a})
