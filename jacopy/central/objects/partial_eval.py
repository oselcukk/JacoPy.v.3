"""
Partial evaluation of a multilinear map — PDF item 8w, "its
generalization to arbitrary numbers" (deferred-items ledger K2,
2026-09-22).

The bilinear case is the musical pair: a map ``g: TM × TM → C^∞`` seen
as ``g^♭ = g(X, ·): TM → T*M``. In general a ``k``-linear map ``T``
with ``j`` of its slots fixed,

    T(a₁, ·, a₃, ·, …)   (``j`` fixed, ``k − j`` open),

is a ``(k − j)``-linear map on the open slots — an element of
``(E^*)^{⊗(k−j)}`` (or of ``E^{⊗(k−j)}`` for covector slots); this is
the coordinate-free index raising/lowering with any number of indices.
:class:`PartialEval` is that object as a node:

* it remembers the head, the total arity, WHICH positions are fixed
  and with what, and the head's ``alternating`` / ``slot_kind`` flags;
* calling it with the remaining arguments merges them into the open
  positions and returns the full :class:`~jacopy.core.multi_eval.MultiEval`
  (:meth:`PartialEval.__call__`);
* its type is derived: an alternating form head loses one degree per
  fixed slot (``degree``), a ``(q, r)`` tensor head loses the fixed
  slots from its signature (:func:`~jacopy.central.objects.tensor.signature_of`).

Engine semantics live in
:mod:`jacopy.central.calculus.partial_eval_rules`:
the collapse ``T(a, ·)(b) → T(a, b)`` (also through a
:class:`~jacopy.core.pairing.Pairing`), multilinearity in the fixed
slots, the identification of the ALTERNATING vector-slot case with the
iterated interior product ``ι_{X_j}⋯ι_{X_1} ω`` (up to the sign of
moving the fixed slots to the front), and the bridge
``g^♭(X) → g(X, ·)``, ``π^♯(α) → π(α, ·)`` from the bilinear musical
atoms of :mod:`jacopy.central.objects.musical`.

Nothing is assumed about the head: an opaque ``Tensor`` stays opaque;
only the bookkeeping (which slot is which) is mechanical.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from jacopy.core.expr import Expr
from jacopy.core.multi_eval import MultiEval, _ALLOWED_SLOT_KINDS
from jacopy.core.symbolic_degree import Degree


class PartialEval(Expr):
    """``T(a₁, ·, a₃, ·)`` — a ``k``-linear map with some slots fixed.

    Parameters
    ----------
    head
        The multilinear object (a form, metric, tensor, multivector …).
    arity
        Its total number of slots ``k``.
    fixed
        ``{position: Expr}`` for the fixed slots (``0 ≤ position < k``);
        at least one slot must stay open (a fully fixed map is a plain
        :class:`MultiEval`) and at least one must be fixed (otherwise
        the object is the head itself).
    alternating, slot_kind
        Forwarded to the :class:`MultiEval` produced on full evaluation
        (the same flags the head's own evaluation carries).
    """

    __slots__ = ("_head", "_arity", "_fixed", "_alternating", "_slot_kind")

    def __init__(
        self,
        head: Expr,
        arity: int,
        fixed: Mapping[int, Expr],
        *,
        alternating: bool = False,
        slot_kind: str = "vector",
    ) -> None:
        if not isinstance(head, Expr):
            raise TypeError("PartialEval head must be an Expr")
        if not isinstance(arity, int) or arity < 2:
            raise ValueError("PartialEval arity must be an int ≥ 2")
        if not isinstance(alternating, bool):
            raise TypeError("alternating must be a bool")
        if slot_kind not in _ALLOWED_SLOT_KINDS:
            raise ValueError(
                f"slot_kind must be one of {_ALLOWED_SLOT_KINDS}, got {slot_kind!r}"
            )
        items = []
        for pos, value in dict(fixed).items():
            if not isinstance(pos, int) or not 0 <= pos < arity:
                raise ValueError(
                    f"fixed slot position {pos!r} out of range 0..{arity - 1}"
                )
            if not isinstance(value, Expr):
                raise TypeError("fixed slot values must be Expr")
            items.append((pos, value))
        items.sort(key=lambda it: it[0])
        if isinstance(head, PartialEval):
            # NESTED partial evaluation flattens onto the original head:
            # T(a, ·, ·)(b, ·) is T(a, b, ·). The outer positions index
            # the inner map's OPEN slots; the flags must agree
            # (2026-09-23 audit, F3: nesting silently dropped the
            # alternating flag).
            if arity != head.n_open:
                raise ValueError(
                    f"the inner map has {head.n_open} open slot(s), got arity {arity}"
                )
            if alternating != head.alternating or slot_kind != head.slot_kind:
                raise ValueError(
                    "nested partial evaluation must keep the inner map's "
                    "alternating / slot_kind flags"
                )
            inner_open = head.open_positions
            merged = dict(head.fixed)
            for pos, value in items:
                merged[inner_open[pos]] = value
            head, arity = head.head, head.arity
            items = sorted(merged.items(), key=lambda it: it[0])
        known = _known_arity(head)
        if known is not None and known != arity:
            # a head with a known signature fixes its own arity: a
            # bilinear metric cannot be viewed as a trilinear map
            # (2026-09-23 audit, F2)
            raise ValueError(
                f"{head._repr_inner()} has {known} slot(s), not {arity}"
            )
        if not items:
            raise ValueError("PartialEval needs at least one fixed slot")
        if len(items) >= arity:
            raise ValueError(
                "PartialEval needs at least one OPEN slot — a fully "
                "evaluated map is a plain MultiEval"
            )
        self._head = head
        self._arity = arity
        self._fixed: Tuple[Tuple[int, Expr], ...] = tuple(items)
        self._alternating = alternating
        self._slot_kind = slot_kind

    # ---- accessors -------------------------------------------------- #

    @property
    def head(self) -> Expr:
        return self._head

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def fixed(self) -> Tuple[Tuple[int, Expr], ...]:
        """``((position, value), …)`` in increasing position."""
        return self._fixed

    @property
    def fixed_args(self) -> Tuple[Expr, ...]:
        return tuple(v for _, v in self._fixed)

    @property
    def fixed_positions(self) -> Tuple[int, ...]:
        return tuple(p for p, _ in self._fixed)

    @property
    def open_positions(self) -> Tuple[int, ...]:
        taken = set(self.fixed_positions)
        return tuple(i for i in range(self._arity) if i not in taken)

    @property
    def n_open(self) -> int:
        return self._arity - len(self._fixed)

    @property
    def alternating(self) -> bool:
        return self._alternating

    @property
    def slot_kind(self) -> str:
        return self._slot_kind

    @property
    def degree(self) -> Degree:
        """Form degree of the OPEN map, when it is a form: an
        alternating head on vector slots gives ``|T| − j``; a single
        open vector slot of ANY head is a 1-form (a ``(0,1)``-tensor,
        e.g. ``g(X, ·)``). A non-alternating map with several open
        slots is a covariant tensor, not a form: the attribute is
        absent (``AttributeError``, so ``getattr(…, None)`` protocols
        pass over it; 2026-09-23 audit, F1)."""
        from jacopy.algebra.derivation import degree_of

        if self._slot_kind == "vector":
            if self._alternating:
                return degree_of(self._head, None) + Degree.const(-len(self._fixed))
            if self.n_open == 1:
                return Degree.const(1)
        raise AttributeError("PartialEval.degree: the open map is not a form")

    @property
    def wedge_degree(self) -> Degree:
        """Multivector degree of the OPEN map, when it is one: an
        alternating head on covector slots gives ``|P| − j``; a single
        open covector slot of any head is a vector (degree 1)."""
        from jacopy.algebra.derivation import degree_of

        if self._slot_kind == "covector":
            if self._alternating:
                lift = getattr(self._head, "wedge_degree", None)
                base = lift if isinstance(lift, Degree) else degree_of(self._head, None)
                return base + Degree.const(-len(self._fixed))
            if self.n_open == 1:
                return Degree.const(1)
        raise AttributeError("PartialEval.wedge_degree: the open map is not a multivector")

    # ---- Expr protocol ---------------------------------------------- #

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._head,) + self.fixed_args

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "PartialEval":
        if len(new_children) != 1 + len(self._fixed):
            raise ValueError("PartialEval._rebuild: wrong number of children")
        head, *values = new_children
        return PartialEval(
            head,
            self._arity,
            {p: v for (p, _), v in zip(self._fixed, values)},
            alternating=self._alternating,
            slot_kind=self._slot_kind,
        )

    def _key(self) -> Any:
        return (
            "PartialEval",
            self._head,
            self._arity,
            self._fixed,
            self._alternating,
            self._slot_kind,
        )

    def _repr_inner(self) -> str:
        fixed = dict(self._fixed)
        slots = ", ".join(
            fixed[i]._repr_inner() if i in fixed else "·" for i in range(self._arity)
        )
        return f"{self._head._repr_inner()}({slots})"

    # ---- evaluation ------------------------------------------------- #

    def merge(self, *rest: Expr) -> Tuple[Expr, ...]:
        """The full argument tuple with ``rest`` filled into the open
        positions (in order)."""
        if len(rest) != self.n_open:
            raise TypeError(
                f"{self._repr_inner()} has {self.n_open} open slot(s), got {len(rest)}"
            )
        for r in rest:
            if not isinstance(r, Expr):
                raise TypeError("PartialEval arguments must be Expr")
        fixed = dict(self._fixed)
        it = iter(rest)
        return tuple(fixed[i] if i in fixed else next(it) for i in range(self._arity))

    def __call__(self, *rest: Expr) -> MultiEval:
        """Fill the open slots: ``T(a, ·)(b) = T(a, b)``."""
        return MultiEval(
            self._head,
            *self.merge(*rest),
            alternating=self._alternating,
            slot_kind=self._slot_kind,
        )

    def with_fixed(self, more: Mapping[int, Expr]) -> "PartialEval":
        """Fix further slots (positions refer to the ORIGINAL arity)."""
        merged: Dict[int, Expr] = dict(self._fixed)
        for p, v in more.items():
            if p in merged:
                raise ValueError(f"slot {p} is already fixed")
            merged[p] = v
        return PartialEval(
            self._head,
            self._arity,
            merged,
            alternating=self._alternating,
            slot_kind=self._slot_kind,
        )


# --------------------------------------------------------------------- #
# Constructors                                                           #
# --------------------------------------------------------------------- #


def _known_arity(head: Expr) -> Optional[int]:
    """The head's slot count when its signature is known (``g``: 2,
    a ``p``-form: ``p``, a ``(q, r)`` tensor: ``q + r``), else ``None``."""
    from jacopy.central.objects.tensor import signature_of

    if isinstance(head, PartialEval):
        return head.n_open
    sig = signature_of(head)
    return None if sig is None else sig[0] + sig[1]


def _flags_for(head: Expr, alternating: Optional[bool], slot_kind: Optional[str]):
    """Default ``alternating``/``slot_kind`` from the head's nature:
    forms and multivectors are alternating (vector / covector slots),
    metrics, inverse metrics and general tensors are not; a partial
    map keeps its own flags."""
    from jacopy.central.objects.form import Form
    from jacopy.central.objects.metric import InverseMetric, Metric
    from jacopy.central.objects.multivector import PVector
    from jacopy.central.objects.tensor import Tensor

    if isinstance(head, PartialEval):
        alternating = head.alternating if alternating is None else alternating
        slot_kind = head.slot_kind if slot_kind is None else slot_kind
        return alternating, slot_kind
    if alternating is None:
        alternating = isinstance(head, (Form, PVector))
    if slot_kind is None:
        if isinstance(head, (PVector, InverseMetric)):
            slot_kind = "covector"
        elif isinstance(head, Tensor):
            slot_kind = "mixed"
        else:
            slot_kind = "vector"
    return alternating, slot_kind


def _arity_for(head: Expr, arity: Optional[int]) -> int:
    known = _known_arity(head)
    if arity is not None:
        if known is not None and known != arity:
            raise ValueError(f"{head._repr_inner()} has {known} slot(s), not {arity}")
        return arity
    if known is None:
        raise ValueError(
            "the head's arity is not determinable — pass arity= explicitly"
        )
    return known


def partial_eval(
    head: Expr,
    *slots: Optional[Expr],
    alternating: Optional[bool] = None,
    slot_kind: Optional[str] = None,
) -> PartialEval:
    """``partial_eval(T, α, None, X, None)`` = ``T(α, ·, X, ·)`` — one
    entry per slot, ``None`` marking an open one. The arity is the
    number of entries; ``alternating``/``slot_kind`` default from the
    head (forms/multivectors alternating; metrics/tensors not)."""
    alt, kind = _flags_for(head, alternating, slot_kind)
    fixed = {i: s for i, s in enumerate(slots) if s is not None}
    return PartialEval(head, len(slots), fixed, alternating=alt, slot_kind=kind)


def musical_view(
    head: Expr,
    *fixed: Expr,
    arity: Optional[int] = None,
    alternating: Optional[bool] = None,
    slot_kind: Optional[str] = None,
) -> PartialEval:
    """``musical_view(T, a₁, …, a_j)`` = ``T(a₁, …, a_j, ·, …, ·)`` —
    the LEADING ``j`` slots fixed, the rest open: the ``E^{k−j} →
    (E^*)^{⊗j}``-style view of PDF item 8w. ``arity`` defaults to the
    head's signature (``g``: 2, a ``p``-form: ``p``, a ``(q, r)``
    tensor: ``q + r``)."""
    k = _arity_for(head, arity)
    alt, kind = _flags_for(head, alternating, slot_kind)
    return PartialEval(head, k, dict(enumerate(fixed)), alternating=alt, slot_kind=kind)
