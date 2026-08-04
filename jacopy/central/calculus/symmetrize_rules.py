"""
Permutation unfolding for (anti)symmetrization under evaluation — the
Phase 1 deferral of PDF item 8x (Phase 2.F).

The canonical definitions, with the ``1/k!`` normalization fixed in
Phase 1 (``T_(ab) = ½(T_ab + T_ba)``, ``T_[ab] = ½(T_ab − T_ba)``):

    Sym(T)(Y_1, …, Y_k) = (1/k!) Σ_σ T(Y_σ(1), …, Y_σ(k)),
    Alt(T)(Y_1, …, Y_k) = (1/k!) Σ_σ sgn(σ) T(Y_σ(1), …, Y_σ(k)).

On a single slot both are the identity (``⟨Sym(T), X⟩ = ⟨T, X⟩``).
Calculus-independent (no anchor/bracket involved).
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

from jacopy.core.expr import Expr, Neg, Product, Rational, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.symmetrize import Antisymmetrization, Symmetrization
from jacopy.proof.expansion import Definition


def _perm_sign(perm) -> int:
    """Parity (0 even / 1 odd) of an index permutation tuple."""
    perm = list(perm)
    swaps = 0
    for i in range(len(perm)):
        while perm[i] != i:
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
            swaps += 1
    return swaps % 2


class SymAltEvalDefinition(Definition):
    """Unfold ``Sym(T)(Y…)`` / ``Alt(T)(Y…)`` into the permutation sum."""

    name = (
        "(anti)symmetrization: Sym/Alt(T)(Y_1,…,Y_k) = "
        "(1/k!) Σ_σ (±) T(Y_σ(1),…,Y_σ(k))"
    )
    anchor = (Pairing, MultiEval)

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Pairing):
            return isinstance(
                expr.alpha, (Symmetrization, Antisymmetrization)
            )
        if isinstance(expr, MultiEval):
            return isinstance(
                expr.head, (Symmetrization, Antisymmetrization)
            )
        return False

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            # One slot: both projections are the identity.
            return Pairing(expr.alpha.arg, expr.X)
        head = expr.head
        inner = head.arg
        args = expr.args
        k = len(args)
        signed = isinstance(head, Antisymmetrization)
        terms = []
        for perm in permutations(range(k)):
            permuted = tuple(args[i] for i in perm)
            node: Expr = MultiEval(
                inner,
                *permuted,
                # The projection's own symmetry is what we are
                # unfolding; the inner evaluation carries no declared
                # symmetry of its own.
                alternating=False,
                slot_kind=expr.slot_kind,
            )
            if signed and _perm_sign(perm):
                node = Neg(node)
            terms.append(node)
        return Product(Rational(1, factorial(k)), Sum(*terms))