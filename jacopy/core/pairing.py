"""
Canonical pairing ``⟨α, X⟩`` — evaluation of a 1-form against a vector
field into a scalar.

This is a core *evaluation primitive* (just like
:class:`~jacopy.core.multi_eval.MultiEval`): it takes two arguments and
produces a degree-0 (scalar) expression. Structurally it stores the
pair symmetrically — the semantic order (first slot: 1-form; second
slot: vector field) is the caller's responsibility.

The pairing is bilinear; algorithms that distribute over
:class:`~jacopy.core.expr.Sum` or pull out scalar factors process this
node like any other Expr by visiting its children. From the
product-rule layer's point of view it is a leaf.

Why it lives in ``core``: ``degree_of`` (the algebra layer) recognises
it as a scalar-producer; keeping it in the core avoids making it
depend on any geometry package.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Expr


class Pairing(Expr):
    """Canonical pairing ``⟨α, X⟩``.

    Children ``(alpha, X)`` are stored in the order given; equality is
    structural. The semantic order (1-form, vector field) is under the
    caller's control.
    """

    __slots__ = ("_alpha", "_X")

    def __init__(self, alpha: Expr, X: Expr) -> None:
        if not isinstance(alpha, Expr):
            raise TypeError("Pairing first argument must be an Expr")
        if not isinstance(X, Expr):
            raise TypeError("Pairing second argument must be an Expr")
        self._alpha = alpha
        self._X = X

    @property
    def alpha(self) -> Expr:
        return self._alpha

    @property
    def X(self) -> Expr:
        return self._X

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._alpha, self._X)

    def _key(self) -> Any:
        return (self._alpha, self._X)

    def _repr_inner(self) -> str:
        return f"⟨{self._alpha._repr_inner()}, {self._X._repr_inner()}⟩"


def pairing(alpha: Expr, X: Expr) -> Pairing:
    """Build the pairing ``⟨α, X⟩`` (readable helper function)."""
    return Pairing(alpha, X)
