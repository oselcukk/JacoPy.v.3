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

from typing import Any, Optional, Tuple

from jacopy.core.expr import Expr
from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree


class HodgeStar(Expr):
    """``⋆_g ω`` — the Hodge dual of a p-form ``ω``, an (n−p)-form.

    Parameters
    ----------
    arg
        The form the star is applied to.
    dim
        The dimension ``n`` of the bundle (int or :class:`Degree`; use
        ``Degree.var("n")`` for a symbolic dimension).
    metric
        The metric the star is taken with respect to (an Expr, usually
        the central :class:`~jacopy.central.objects.metric.Metric`
        node), or ``None`` for the bare structural star. The metric is
        PART OF THE OPERATION'S IDENTITY: ``⋆_g dx = 1`` while
        ``⋆_{4g} dx = ½`` in oriented dimension 1, so it participates
        in ``_key`` — two stars with different metrics are never equal
        (2026-09-07 audit, finding 5).
    """

    __slots__ = ("_arg", "_dim", "_metric")

    def __init__(
        self,
        arg: Expr,
        dim: DegreeLike,
        *,
        metric: Optional[Expr] = None,
    ) -> None:
        if not isinstance(arg, Expr):
            raise TypeError("HodgeStar argument must be an Expr")
        if metric is not None and not isinstance(metric, Expr):
            raise TypeError("HodgeStar metric must be an Expr")
        self._arg = arg
        self._dim = as_degree(dim)
        self._metric = metric

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def dim(self) -> Degree:
        return self._dim

    @property
    def metric(self) -> Optional[Expr]:
        return self._metric

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._arg,)

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "HodgeStar":
        """Preserve dimension and metric context — the default
        ``type(self)(*children)`` rebuild dropped ``dim`` and crashed
        every generic tree pass (2026-09-07 audit, finding 6)."""
        (arg,) = new_children
        return HodgeStar(arg, self._dim, metric=self._metric)

    def _key(self) -> Any:
        return (self._arg, self._dim, self._metric)

    def _repr_inner(self) -> str:
        if self._metric is not None:
            return (
                f"⋆_{self._metric._repr_inner()}"
                f"{self._arg._repr_inner()}"
            )
        return f"⋆{self._arg._repr_inner()}"


def hodge_star(
    arg: Expr,
    dim: DegreeLike,
    *,
    metric: Optional[Expr] = None,
) -> HodgeStar:
    """Build the Hodge dual ``⋆ω`` (with dimension ``dim`` given)."""
    return HodgeStar(arg, dim, metric=metric)
