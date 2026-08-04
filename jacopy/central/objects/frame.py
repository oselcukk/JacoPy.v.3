"""
Frame ``(e_a)`` and coframe ``(e^a)`` (PDF items 8o-p).

A **frame** ``(e_a)`` is a local basis of sections of a bundle — the
usual vector-field frame on ``TM``, a local basis of ``E`` for a
general algebroid. The dual **coframe** ``(e^a)`` is the basis of
1-forms satisfying ``e^a(e_b) = ⟨e^a, e_b⟩ = δ^a_b`` (Kronecker).

Implementation:

* :class:`FrameField` — ``e_a``, a
  :class:`~jacopy.central.objects.vector_field.VectorField` (degree 0)
  carrying an index label (displayed ``"e_a"``).
* :class:`CoframeField` — ``e^a``, a
  :class:`~jacopy.central.objects.form.Form` (degree 1) carrying an
  upper index (displayed ``"e^a"``).
* :class:`Frame` / :class:`Coframe` — context objects (like a Bundle);
  they produce the indexed fields and each other's dual. A frame and
  its dual coframe share the same name and bundle.

Duality ``⟨e^a, e_b⟩ = δ^a_b``: the pairing node
(:class:`~jacopy.core.pairing.Pairing`) is built by
:meth:`Frame.pairing`; its *evaluation* to ``δ^a_b`` is an expansion
rule (definition) and will be added to the engine layer in Phase 2
alongside d/L. :func:`kronecker_delta` supplies the scalar directly
(``δ^a_b``; ``1`` for equal labels).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

from jacopy.core.expr import Atom, Expr, One, Symbol
from jacopy.core.pairing import Pairing
from jacopy.central.objects.bundle import Bundle, TM
from jacopy.central.objects.form import Form
from jacopy.central.objects.vector_field import VectorField

class FrameIndex(Atom):
    """A frame-index ATOM — the bound dummy of an
    :class:`~jacopy.core.indexed_sum.IndexedSum` over frame labels
    (Phase 4.D.2), and the target of index substitution.

    Index substitution is NAME-BASED: indexed atoms (frame fields,
    coefficients, deltas) store their indices as strings; their
    ``substitute_atom`` overrides rename a slot whose string equals
    the dummy's name. ``FrameIndex`` therefore never needs to be
    STORED inside those atoms — it exists as the binder and as the
    substitution carrier (including the α-equivalence sentinels of
    ``IndexedSum``). Pick bound names that do not collide with free
    index labels in scope (variable capture is the caller's to avoid,
    as usual with named binders).
    """

    __slots__ = ("_index_name", "_kind")

    def __init__(self, name: str, kind: str = "bound") -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("FrameIndex name must be a non-empty str")
        self._index_name = name
        self._kind = kind

    @property
    def name(self) -> str:
        return self._index_name

    @property
    def kind(self) -> str:
        return self._kind

    def _key(self) -> Any:
        return ("frame-index", self._index_name)

    def _repr_inner(self) -> str:
        return self._index_name


IndexLike = Union[str, int, FrameIndex]


def _index_str(index: IndexLike) -> str:
    if isinstance(index, FrameIndex):
        return index.name
    if isinstance(index, bool) or not isinstance(index, (str, int)):
        raise TypeError("frame index must be a str, int or FrameIndex")
    s = str(index)
    if not s:
        raise ValueError("frame index must be non-empty")
    return s


def substituted_index_names(
    names: Tuple[str, ...], dummy: Expr, target: Expr
) -> Optional[Tuple[str, ...]]:
    """Rename occurrences of a :class:`FrameIndex` dummy in a tuple of
    index-name slots; ``None`` when nothing changes (or when the
    substitution is not an index renaming)."""
    if not isinstance(dummy, FrameIndex) or not isinstance(
        target, FrameIndex
    ):
        return None
    if dummy.name not in names:
        return None
    return tuple(
        target.name if n == dummy.name else n for n in names
    )


class FrameField(VectorField):
    """``e_a`` — an indexed section of a frame (degree 0)."""

    __slots__ = ("_index",)

    def __init__(
        self,
        frame_name: str,
        index: IndexLike,
        *,
        bundle: Optional[Bundle] = None,
    ) -> None:
        idx = _index_str(index)
        super().__init__(f"{frame_name}_{idx}", bundle=bundle, degree=0)
        self._index = idx

    @property
    def index(self) -> str:
        return self._index

    @property
    def base_name(self) -> str:
        """The owning frame's name (``"e"`` for ``e_a``)."""
        return self._name[: -(len(self._index) + 1)]

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._index,)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        renamed = substituted_index_names(
            (self._index,), dummy, target
        )
        if renamed is None:
            return self
        return FrameField(self.base_name, renamed[0], bundle=self.bundle)


class CoframeField(Form):
    """``e^a`` — an indexed 1-form of a coframe (degree 1)."""

    __slots__ = ("_index",)

    def __init__(
        self,
        coframe_name: str,
        index: IndexLike,
        *,
        bundle: Optional[Bundle] = None,
    ) -> None:
        idx = _index_str(index)
        super().__init__(f"{coframe_name}^{idx}", degree=1, bundle=bundle)
        self._index = idx

    @property
    def index(self) -> str:
        return self._index

    @property
    def base_name(self) -> str:
        """The owning coframe's name (``"e"`` for ``e^a``)."""
        return self._name[: -(len(self._index) + 1)]

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._index,)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        renamed = substituted_index_names(
            (self._index,), dummy, target
        )
        if renamed is None:
            return self
        return CoframeField(self.base_name, renamed[0], bundle=self.bundle)


class Coframe:
    """``(e^a)`` — the dual 1-form basis of a frame (item 8p)."""

    __slots__ = ("_name", "_bundle")

    def __init__(self, name: str = "e", *, bundle: Optional[Bundle] = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Coframe name must be a non-empty str")
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

    def field(self, index: IndexLike) -> CoframeField:
        """``e^a`` — the coframe 1-form with the given upper index."""
        return CoframeField(self._name, index, bundle=self._bundle)

    def fields(self, indices: str) -> Tuple[CoframeField, ...]:
        """``e^a, e^b, …`` for whitespace-separated indices."""
        parts = tuple(indices.split())
        if not parts:
            raise ValueError("at least one index is required")
        return tuple(self.field(i) for i in parts)

    def dual(self) -> "Frame":
        """The dual frame ``(e_a)`` of this coframe (same name + bundle)."""
        return Frame(self._name, bundle=self._bundle)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Coframe)
            and self._name == other._name
            and self._bundle == other._bundle
        )

    def __hash__(self) -> int:
        return hash(("coframe", self._name, self._bundle))

    def __repr__(self) -> str:
        return f"Coframe({self._name!r}, bundle={self._bundle!r})"


class Frame:
    """``(e_a)`` — a local basis of sections of a bundle (item 8o).

    A context object (not an Expr). Produces the indexed frame fields
    (:meth:`field`) and the dual coframe (:meth:`dual`).
    """

    __slots__ = ("_name", "_bundle")

    def __init__(self, name: str = "e", *, bundle: Optional[Bundle] = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Frame name must be a non-empty str")
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

    def field(self, index: IndexLike) -> FrameField:
        """``e_a`` — the frame section with the given lower index."""
        return FrameField(self._name, index, bundle=self._bundle)

    def fields(self, indices: str) -> Tuple[FrameField, ...]:
        """``e_a, e_b, …`` for whitespace-separated indices."""
        parts = tuple(indices.split())
        if not parts:
            raise ValueError("at least one index is required")
        return tuple(self.field(i) for i in parts)

    def dual(self) -> Coframe:
        """The dual coframe ``(e^a)`` (same name + bundle)."""
        return Coframe(self._name, bundle=self._bundle)

    def pairing(self, upper: IndexLike, lower: IndexLike) -> Pairing:
        """``⟨e^a, e_b⟩`` — the duality pairing node (item 8p).

        Its evaluation to ``δ^a_b`` comes with the Phase 2 expansion
        rule; for the scalar use :func:`kronecker_delta`.
        """
        return Pairing(self.dual().field(upper), self.field(lower))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Frame)
            and self._name == other._name
            and self._bundle == other._bundle
        )

    def __hash__(self) -> int:
        return hash(("frame", self._name, self._bundle))

    def __repr__(self) -> str:
        return f"Frame({self._name!r}, bundle={self._bundle!r})"


def frame(name: str = "e", *, bundle: Optional[Bundle] = None) -> Frame:
    """Create a local frame ``(e_a)``."""
    return Frame(name, bundle=bundle)


def coframe(name: str = "e", *, bundle: Optional[Bundle] = None) -> Coframe:
    """Create a local coframe ``(e^a)``."""
    return Coframe(name, bundle=bundle)


class KroneckerDelta(Atom):
    """``δ^a_b`` — the Kronecker delta atom (a *constant* scalar).

    Degree 0 and ``is_constant = True``: every derivation annihilates
    it (``X(δ^a_b) = 0``), which :mod:`jacopy.algorithms.product_rule`
    honours via the ``is_constant`` protocol. This is what makes frame
    identities like ``de^c(e_a, e_b) = −γ^c_ab`` close — the
    ``e_a(δ^c_b)`` terms of the Palais unroll must die.
    """

    __slots__ = ("_upper", "_lower")

    #: Constant-function marker (see product_rule): D(δ^a_b) = 0.
    is_constant = True

    def __init__(self, upper: str, lower: str) -> None:
        self._upper = upper
        self._lower = lower

    @property
    def upper(self) -> str:
        return self._upper

    @property
    def lower(self) -> str:
        return self._lower

    @property
    def degree(self):
        from jacopy.core.symbolic_degree import Degree

        return Degree.const(0)

    def _key(self):
        return (self._upper, self._lower)

    def _repr_inner(self) -> str:
        return f"δ^{self._upper}_{self._lower}"

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._upper, self._lower)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        renamed = substituted_index_names(
            (self._upper, self._lower), dummy, target
        )
        if renamed is None:
            return self
        return kronecker_delta(renamed[0], renamed[1])


def kronecker_delta(upper: IndexLike, lower: IndexLike) -> Expr:
    """``δ^a_b`` — the Kronecker delta (a degree-0 constant scalar).

    Equal index labels give ``1`` (:data:`~jacopy.core.expr.One`);
    distinct labels give the :class:`KroneckerDelta` atom. Distinct
    *abstract* labels stay symbolic because ``a`` and ``b`` might
    still name the same index.

    Note: labels here denote **fixed** indices — equal labels mean the
    same single index, so ``δ^a_a = 1``. This is *not* the
    Einstein-summed trace (``δ^a_a = n`` under summation); summed
    traces belong to the frame-component layer (Phase 4).
    """
    a = _index_str(upper)
    b = _index_str(lower)
    if a == b:
        return One
    return KroneckerDelta(a, b)
