"""
(q, r)-type tensors, scaling by a function ``fT``, and signature
tracking — PDF items 8j, 8n, 8q.

A **(q, r)-type tensor** has ``q`` contravariant (upper) and ``r``
covariant (lower) slots. Over a bundle ``E`` it is a multilinear map
``T: (E^*)^q × E^r → C^∞(M)``; taking ``E = TM`` it is the usual
(q, r)-tensor.

:class:`Tensor` does not carry a single grading number — its shape is
the pair ``(q, r)`` (a p-form is the antisymmetric special case
``(0, p)``, a p-vector the special case ``(p, 0)``). For that reason a
Tensor carries **no form degree** and stays outside ``degree_of``.

The type-level bookkeeping that *does* apply to tensors is the
**signature** ``(q, r)``, exposed by :func:`signature_of`. It walks the
expression tree through the type-preserving wrappers — most importantly
the covariant derivative ``∇_X T`` (PDF item 8n: a connection maps a
(q, r)-tensor to a (q, r)-tensor), ``fT`` scaling, negation and
(anti)symmetrization — so that type preservation is checkable even
though ``degree_of`` does not apply.

``fT`` (item 8q): multiplying a tensor by a smooth function is the
``C^∞(M)``-module structure. :func:`scale` builds it as
``Product(f, T)`` and works for any tensor-like object (Form, PVector,
VectorField, Tensor).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.core.expr import Atom, Expr, Neg, Product
from jacopy.core.multi_eval import MultiEval
from jacopy.central.objects.bundle import Bundle, TM


class Tensor(Atom):
    """A (q, r)-type tensor; on ``TM`` the usual tensor.

    Parameters
    ----------
    name
        Display name (e.g. ``"T"``).
    upper
        Number of contravariant (upper) slots ``q ≥ 0``.
    lower
        Number of covariant (lower) slots ``r ≥ 0``.
    bundle
        Owning bundle; defaults to :data:`TM`.

    Notes
    -----
    Equality is structural over ``(name, q, r, bundle)``. A tensor does
    not carry a single form degree, so it has **no** ``degree``
    property; ``degree_of`` does not recognize it (in the form-grading
    sense). Type-level tracking is done via :func:`signature_of`.
    """

    __slots__ = ("_name", "_upper", "_lower", "_bundle")

    def __init__(
        self,
        name: str,
        *,
        upper: int,
        lower: int,
        bundle: Optional[Bundle] = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor name must be a non-empty str")
        if not isinstance(upper, int) or upper < 0:
            raise ValueError("upper (q) must be a non-negative int")
        if not isinstance(lower, int) or lower < 0:
            raise ValueError("lower (r) must be a non-negative int")
        if bundle is not None and not isinstance(bundle, Bundle):
            raise TypeError("bundle must be a Bundle instance")
        self._name = name
        self._upper = upper
        self._lower = lower
        self._bundle = bundle if bundle is not None else TM

    @property
    def name(self) -> str:
        return self._name

    @property
    def upper(self) -> int:
        """Number of contravariant slots ``q``."""
        return self._upper

    @property
    def lower(self) -> int:
        """Number of covariant slots ``r``."""
        return self._lower

    @property
    def signature(self) -> Tuple[int, int]:
        """The pair ``(q, r)``."""
        return (self._upper, self._lower)

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    def _key(self) -> Any:
        return (self._name, self._upper, self._lower, self._bundle)

    def _repr_inner(self) -> str:
        return self._name

    def __call__(self, *args: Expr) -> MultiEval:
        """``T(α¹, …, α^q, X₁, …, X_r)`` — full multilinear evaluation.

        Exactly ``q + r`` arguments are required: first the ``q``
        covector (1-form) slots, then the ``r`` vector-field slots
        (PDF item 8j: appropriate domain and range). The result is a
        plain multilinear :class:`~jacopy.core.multi_eval.MultiEval`
        (``alternating=False``, ``slot_kind="mixed"``), a scalar.
        """
        expected = self._upper + self._lower
        if len(args) != expected:
            raise TypeError(
                f"{self._name} is a ({self._upper}, {self._lower})-tensor: "
                f"evaluation takes exactly {expected} argument(s) "
                f"({self._upper} covector(s) then {self._lower} "
                f"vector(s)), got {len(args)}"
            )
        for a in args:
            if not isinstance(a, Expr):
                raise TypeError("Tensor call arguments must be Expr")
        return MultiEval(self, *args, alternating=False, slot_kind="mixed")


def tensors(
    names: str,
    *,
    upper: int,
    lower: int,
    bundle: Optional[Bundle] = None,
) -> Tuple[Tensor, ...]:
    """Create one or more (q, r)-tensors.

    ``tensors("S T", upper=1, lower=2)`` returns two (1,2)-tensors.
    Even for a single name the return value is a tuple. No registry is
    needed.
    """
    if not isinstance(names, str):
        raise TypeError("names must be a whitespace-separated str")
    parts = tuple(names.split())
    if not parts:
        raise ValueError("at least one name is required")
    return tuple(
        Tensor(n, upper=upper, lower=lower, bundle=bundle) for n in parts
    )


def scale(f: Expr, T: Expr) -> Product:
    """Scaling by a function ``f·T`` (PDF item 8q).

    Multiplies any tensor-like object (Form, PVector, VectorField,
    Tensor) by a smooth function ``f ∈ C^∞(M)``; the result is
    ``Product(f, T)``. The shape of the scaled object (form degree or
    ``(q, r)`` signature) is unchanged — only a scalar coefficient is
    attached.
    """
    if not isinstance(f, Expr) or not isinstance(T, Expr):
        raise TypeError("scale arguments must be Expr")
    return Product(f, T)


def signature_of(expr: Expr) -> Optional[Tuple[int, int]]:
    """Return the ``(q, r)`` signature of ``expr``, or ``None``.

    The tensor-type analogue of
    :func:`~jacopy.algebra.derivation.degree_of`: it tracks the
    ``(q, r)`` shape through type-preserving nodes, so statements like
    "``∇_X`` maps a (q, r)-tensor to a (q, r)-tensor" (PDF item 8n)
    are checkable. Recognized cases:

    * :class:`Tensor` → its ``(q, r)``;
      :class:`~jacopy.central.objects.metric.Metric` → ``(0, 2)``;
      :class:`~jacopy.central.objects.metric.InverseMetric` → ``(2, 0)``.
    * :class:`~jacopy.central.objects.vector_field.VectorField` →
      ``(1, 0)``.
    * :class:`~jacopy.central.objects.form.Form` with concrete degree
      ``p`` → ``(0, p)``;
      :class:`~jacopy.central.objects.multivector.PVector` with
      concrete ``p`` → ``(p, 0)``.
    * ``∇_X T`` (:class:`Act` with a
      :class:`~jacopy.central.objects.connection.CovariantOp` head) →
      signature of ``T`` (the connection preserves the type).
    * :class:`Neg`, ``Sym``/``Alt`` → signature of the argument.
    * :class:`Product` with exactly one signatured factor → that
      factor's signature (covers ``fT``; the remaining factors are
      treated as scalar coefficients). More than one signatured factor
      is ambiguous at this layer → ``None``.

    Returns ``None`` whenever the signature cannot be determined
    (symbolic degree, unrecognized node, ambiguous product); callers in
    symbolic contexts may then proceed without signature enforcement.
    """
    if not isinstance(expr, Expr):
        raise TypeError("signature_of argument must be an Expr")
    if isinstance(expr, Tensor):
        return expr.signature
    # Late imports keep this module free of eager sibling dependencies
    # (metric imports musical, connection imports bundle, ...).
    from jacopy.central.objects.metric import InverseMetric, Metric
    if isinstance(expr, (Metric, InverseMetric)):
        return expr.signature
    from jacopy.central.objects.vector_field import VectorField
    if isinstance(expr, VectorField):
        return (1, 0)
    from jacopy.central.objects.form import Form
    if isinstance(expr, Form):
        p = expr.degree.as_int()
        return None if p is None else (0, p)
    from jacopy.central.objects.multivector import PVector
    if isinstance(expr, PVector):
        p = expr.degree.as_int()
        return None if p is None else (p, 0)
    if isinstance(expr, Neg):
        return signature_of(expr.arg)
    from jacopy.core.symmetrize import Antisymmetrization, Symmetrization
    if isinstance(expr, (Symmetrization, Antisymmetrization)):
        return signature_of(expr.arg)
    from jacopy.algebra.derivation import Act
    from jacopy.central.objects.connection import CovariantOp
    if isinstance(expr, Act) and isinstance(expr.op, CovariantOp):
        return signature_of(expr.arg)
    if isinstance(expr, Product):
        found: Optional[Tuple[int, int]] = None
        for c in expr.children:
            sig = signature_of(c)
            if sig is not None:
                if found is not None:
                    return None
                found = sig
        return found
    return None
