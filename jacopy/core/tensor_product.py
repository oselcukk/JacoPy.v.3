"""
Tensor product ``⊗`` — PDF item 8u.

``a ⊗ b`` combines two tensor-like objects; unlike the wedge it is
**not** graded-commutative (``a ⊗ b ≠ ± b ⊗ a`` in general) and applies
no symmetry. Degrees add: ``|a ⊗ b| = |a| + |b|``; the ``(q, r)``
shapes also add (``(q₁+q₂, r₁+r₂)``).

This is a core algebraic primitive (in a position symmetric to Wedge);
:func:`~jacopy.algebra.derivation.degree_of` recognises it as the sum
of its children's degrees.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Expr


class TensorProduct(Expr):
    r"""n-ary tensor product ``a_1 ⊗ a_2 ⊗ … ⊗ a_n``.

    Children are stored in the order given; no reordering is applied
    (the tensor product is non-commutative). :meth:`make` flattens
    nested products (associativity) and collapses to a single child.
    """

    __slots__ = ("_children",)

    def __init__(self, *children: Expr) -> None:
        if len(children) < 2:
            raise ValueError(
                "TensorProduct requires at least two factors"
            )
        for c in children:
            if not isinstance(c, Expr):
                raise TypeError("TensorProduct children must be Expr")
        self._children = tuple(children)

    @property
    def children(self) -> Tuple[Expr, ...]:
        return self._children

    @classmethod
    def make(cls, *args: Expr) -> Expr:
        """Smart constructor: flatten nested ``⊗``, return a lone factor."""
        flat: list[Expr] = []
        for a in args:
            if isinstance(a, TensorProduct):
                flat.extend(a._children)
            else:
                flat.append(a)
        if len(flat) == 1:
            return flat[0]
        return cls(*flat)

    def _key(self) -> Any:
        return self._children

    def _repr_inner(self) -> str:
        return " ⊗ ".join(c._repr_inner() for c in self._children)


def tensor_product(*args: Expr) -> Expr:
    """Build the tensor product ``a ⊗ b ⊗ …`` (via the smart constructor)."""
    return TensorProduct.make(*args)
