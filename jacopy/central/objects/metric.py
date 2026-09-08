"""
Metric ``g``, inverse metric ``g⁻¹`` and the Hodge star ``⋆_g``
(PDF items 8k-l).

* :class:`Metric` — a ``(0, 2)`` symmetric, non-degenerate tensor.
  ``g(X, Y)`` maps two vector fields to a scalar (symmetric → not
  alternating). Produces the musical (flat) map ``g^♭`` and its
  inverse.
* :class:`InverseMetric` — the ``(2, 0)`` tensor ``g⁻¹``;
  ``g⁻¹(α, β)`` maps two 1-forms to a scalar. Produces the musical
  (sharp) map ``g^♯``.
* :func:`hodge` — ``⋆_g ω``; the bundle dimension ``n`` is read from
  the metric (symbolic ``n`` when ``dim=None``), the result is an
  ``(n−p)``-form (:class:`~jacopy.core.hodge.HodgeStar`). When both
  the dimension and the argument's degree are concrete, the range
  ``0 ≤ p ≤ n`` is enforced.

**Central-code view.** On ``TM`` these are the ordinary metric and
Hodge star; for a general bundle ``E`` the metric is a fiber metric on
``E``. Since the metric is non-degenerate, ``♭`` and ``♯`` are mutually
inverse (musical compatibility); that proof is handled in Phase 4/5.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.core.expr import Atom, Expr
from jacopy.core.hodge import HodgeStar
from jacopy.core.multi_eval import MultiEval
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects.bundle import Bundle, TM
from jacopy.central.objects.musical import Flat, Sharp


class Metric(Atom):
    """``g`` — a ``(0, 2)`` symmetric non-degenerate metric.

    Parameters
    ----------
    name
        Display name (e.g. ``"g"``).
    bundle
        Owning bundle; defaults to :data:`TM`. The bundle's dimension
        (``bundle.dim``) is used by the Hodge star.
    """

    __slots__ = ("_name", "_bundle")

    def __init__(self, name: str = "g", *, bundle: Optional[Bundle] = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty str")
        if bundle is not None and not isinstance(bundle, Bundle):
            raise TypeError("bundle must be a Bundle instance")
        self._name = name
        self._bundle = bundle if bundle is not None else TM

    @property
    def name(self) -> str:
        return self._name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def signature(self) -> Tuple[int, int]:
        """``(0, 2)`` — two covariant slots."""
        return (0, 2)

    def _key(self) -> Any:
        return ("Metric", self._name, self._bundle)

    def _repr_inner(self) -> str:
        return self._name

    def __call__(self, X: Expr, Y: Expr) -> MultiEval:
        """``g(X, Y)`` — symmetric evaluation (a scalar)."""
        if not isinstance(X, Expr) or not isinstance(Y, Expr):
            raise TypeError("g(X, Y) arguments must be Expr")
        return MultiEval(self, X, Y, alternating=False, slot_kind="vector")

    def flat(self) -> Flat:
        """The musical (lowering) map ``g^♭ : TM → T*M``."""
        return Flat(self)

    def inverse(self, name: Optional[str] = None) -> "InverseMetric":
        """The inverse metric ``g⁻¹`` — a ``(2, 0)`` tensor."""
        inv_name = name if name is not None else f"{self._name}⁻¹"
        return InverseMetric(inv_name, bundle=self._bundle)


class InverseMetric(Atom):
    """``g⁻¹`` — the ``(2, 0)`` inverse metric.

    ``g⁻¹(α, β)`` maps two 1-forms to a scalar (symmetric). Produces
    the musical (raising) map ``g^♯``.
    """

    __slots__ = ("_name", "_bundle")

    def __init__(self, name: str = "g⁻¹", *, bundle: Optional[Bundle] = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("InverseMetric name must be a non-empty str")
        if bundle is not None and not isinstance(bundle, Bundle):
            raise TypeError("bundle must be a Bundle instance")
        self._name = name
        self._bundle = bundle if bundle is not None else TM

    @property
    def name(self) -> str:
        return self._name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def signature(self) -> Tuple[int, int]:
        """``(2, 0)`` — two contravariant slots."""
        return (2, 0)

    def _key(self) -> Any:
        return ("InverseMetric", self._name, self._bundle)

    def _repr_inner(self) -> str:
        return self._name

    def __call__(self, alpha: Expr, beta: Expr) -> MultiEval:
        """``g⁻¹(α, β)`` — symmetric evaluation (a scalar)."""
        if not isinstance(alpha, Expr) or not isinstance(beta, Expr):
            raise TypeError("g⁻¹(α, β) arguments must be Expr")
        return MultiEval(self, alpha, beta, alternating=False, slot_kind="covector")

    def sharp(self) -> Sharp:
        """The musical (raising) map ``g^♯ : T*M → TM``."""
        return Sharp(self)


def metric(name: str = "g", *, bundle: Optional[Bundle] = None) -> Metric:
    """Create a metric ``g``."""
    return Metric(name, bundle=bundle)


def hodge(omega: Expr, g: Metric) -> HodgeStar:
    """``⋆_g ω`` — the Hodge dual of ``ω`` w.r.t. ``g``
    (p-form → (n−p)-form).

    The bundle dimension ``n`` is read from the metric; when
    ``bundle.dim is None`` a symbolic ``n`` (``Degree.var("n")``) is
    used. NOTE (variable capture): the symbolic dimension is literally
    the degree variable ``n`` — a form whose own symbolic degree uses
    ``Degree.var("n")`` will unify with it (``|⋆ω| = n − n = 0``).
    That is correct exactly when you *mean* ``p = dim`` (a top-form);
    otherwise name your form's degree variable something else
    (``Degree.var("p")``). When both ``n`` and the degree ``p`` of ``omega`` are
    concrete integers, ``0 ≤ p ≤ n`` is enforced (the Hodge star is
    only defined on that range); when either is symbolic or
    undeterminable the check is skipped (symbolic proof mode).
    """
    if not isinstance(omega, Expr):
        raise TypeError("hodge argument must be an Expr")
    if not isinstance(g, Metric):
        # PDF item 5 compatibility bridge (2026-09-08 audit,
        # compliance finding 3): accept the metric-affine package's
        # metric CONTEXT too — same defining semantics (symmetric,
        # non-degenerate (0,2) by definition), shared by name. The
        # central→packages import stays function-local (lazy) to
        # respect the layering.
        try:
            from jacopy.packages.metric_affine.metric import (
                Metric as AffineMetric,
            )
        except ImportError:  # pragma: no cover
            AffineMetric = ()
        if isinstance(g, AffineMetric):
            g = Metric(g.name)
        else:
            raise TypeError(
                "hodge second argument must be a Metric "
                "(central atom or metric-affine context)"
            )
    dim = g.bundle.dim
    n = Degree.var("n") if dim is None else Degree.const(dim)
    if dim is not None:
        # Late import: the algebra layer late-imports core.hodge, so an
        # eager top-level import here would tighten the cycle.
        from jacopy.algebra.derivation import degree_of
        try:
            p = degree_of(omega).as_int()
        except ValueError:
            p = None
        if p is not None and not (0 <= p <= dim):
            raise ValueError(
                f"hodge: argument has degree {p}, outside the valid "
                f"range 0..{dim} for a bundle of dimension {dim}"
            )
    return HodgeStar(omega, n, metric=g)
