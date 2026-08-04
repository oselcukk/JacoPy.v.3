"""
Hodge star ``⋆`` — the p-form ↔ (n−p)-form isomorphism with respect to
a metric (PDF item 8l).

On an ``n``-dimensional bundle the Hodge star sends a p-form to an
(n−p)-form: ``⋆: Λ^p → Λ^{n-p}``. The degree transformation is not
linear (``|⋆ω| = n − |ω|``), so the star is modelled not as an
operator application (``Act``) but as its own
:class:`~jacopy.core.expr.Expr` node; it carries the dimension ``n``
as a :class:`~jacopy.core.symbolic_degree.Degree` (a symbolic
dimension is also possible).

This core node is structural — the information about which metric the
star is taken with respect to is supplied when ``⋆_g`` is built in the
:mod:`~jacopy.central.objects.metric` layer. ``degree_of`` recognises
this node as ``n − |arg|``.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Expr
from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree


class HodgeStar(Expr):
    """``⋆ω`` — the Hodge dual of a p-form ``ω``, an (n−p)-form.

    Parameters
    ----------
    arg
        The form the star is applied to.
    dim
        The dimension ``n`` of the bundle (int or :class:`Degree`; use
        ``Degree.var("n")`` for a symbolic dimension).
    """

    __slots__ = ("_arg", "_dim")

    def __init__(self, arg: Expr, dim: DegreeLike) -> None:
        if not isinstance(arg, Expr):
            raise TypeError("HodgeStar argument must be an Expr")
        self._arg = arg
        self._dim = as_degree(dim)

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def dim(self) -> Degree:
        return self._dim

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._arg,)

    def _key(self) -> Any:
        return (self._arg, self._dim)

    def _repr_inner(self) -> str:
        return f"⋆{self._arg._repr_inner()}"


def hodge_star(arg: Expr, dim: DegreeLike) -> HodgeStar:
    """Build the Hodge dual ``⋆ω`` (with dimension ``dim`` given)."""
    return HodgeStar(arg, dim)
