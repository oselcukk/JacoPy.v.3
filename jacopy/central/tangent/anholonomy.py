"""
Anholonomy coefficients ``γ^c_ab`` of the Lie bracket on a frame
(PDF item 9b) and the frame writing of the bracket properties (the
frame half of item 9c) — Phase 2.E.

A frame ``(e_a)`` is generally *anholonomic*: its brackets do not
vanish but decompose back over the frame,

    [e_a, e_b] = γ^c_{ab} e_c        (sum over c).

**Canonical definition (coefficient extraction).** Rather than the
summed decomposition, the definitional content encoded here is the
coefficient itself:

    γ^c_{ab} := ⟨e^c, [e_a, e_b]⟩,

i.e. pairing the bracket against the dual coframe — which is how
components are always extracted, needs no sum over the bundle
dimension, and works for symbolic dimension. The rule
:class:`FrameBracketCoefficientDefinition` rewrites that pairing to
the :class:`AnholonomyCoefficient` scalar atom. The summed
decomposition (an :class:`~jacopy.core.indexed_sum.IndexedSum` over
``c``) is the component-layer view and arrives with the metric-affine
frame components (Phase 4).

**Canonical index order.** ``γ^c_{ab}`` with ``a > b`` (string order)
is normalized to ``−γ^c_{ba}``, and ``γ^c_{aa} = 0``. This bakes in
bracket antisymmetry — a *theorem* (2.A) — as a canonical form;
:func:`prove_gamma_antisymmetry` exhibits the mechanical proof that
sanctions it.

**Holonomic frames.** A coordinate frame has ``[∂_i, ∂_j] = 0``;
declaring a frame holonomic registers
:class:`HolonomicFrameDefinition`, which rewrites its frame brackets
to zero (so every ``γ`` collapses).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Atom, Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.objects.frame import (
    CoframeField,
    Frame,
    FrameField,
    IndexLike,
    _index_str,
)
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket


class AnholonomyCoefficient(Atom):
    """``γ^c_ab`` — an anholonomy (structure) coefficient.

    A genuine **function** on the base (degree 0, *not* constant):
    derivations act on it nontrivially — ``e_d(γ^c_ab)`` terms are what
    the frame Jacobi identity is made of.
    """

    __slots__ = ("_frame_name", "_upper", "_first", "_second")

    def __init__(
        self, frame_name: str, upper: str, first: str, second: str
    ) -> None:
        self._frame_name = frame_name
        self._upper = upper
        self._first = first
        self._second = second

    @property
    def frame_name(self) -> str:
        return self._frame_name

    @property
    def upper(self) -> str:
        return self._upper

    @property
    def lower(self) -> Tuple[str, str]:
        return (self._first, self._second)

    @property
    def degree(self) -> Degree:
        return Degree.const(0)

    def _key(self) -> Any:
        return (self._frame_name, self._upper, self._first, self._second)

    def _repr_inner(self) -> str:
        # Include the frame name for non-default frames so two frames'
        # coefficients stay distinguishable on screen (identity was
        # always distinct; audit 4 was display-only).
        if self._frame_name == "e":
            return f"γ^{self._upper}_{self._first}{self._second}"
        return (
            f"γ({self._frame_name})^{self._upper}"
            f"_{self._first}{self._second}"
        )

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._upper, self._first, self._second)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        from jacopy.central.objects.frame import substituted_index_names

        if self == dummy:
            return target
        renamed = substituted_index_names(
            self.index_names, dummy, target
        )
        if renamed is None:
            return self
        return AnholonomyCoefficient(self._frame_name, *renamed)


def anholonomy_coefficient(
    fr: Frame, upper: IndexLike, first: IndexLike, second: IndexLike
) -> Expr:
    """``γ^c_ab`` in canonical form.

    ``a == b`` gives ``0``; ``a > b`` (string order) gives
    ``−γ^c_{ba}`` — the antisymmetry inherited from the bracket
    (theorem-backed canonical form; see
    :func:`prove_gamma_antisymmetry`).
    """
    if not isinstance(fr, Frame):
        raise TypeError("anholonomy_coefficient expects a Frame")
    c = _index_str(upper)
    a = _index_str(first)
    b = _index_str(second)
    if a == b:
        return Integer(0)
    if a > b:
        return Neg(AnholonomyCoefficient(fr.name, c, b, a))
    return AnholonomyCoefficient(fr.name, c, a, b)


# --------------------------------------------------------------------- #
# Definitional rules                                                     #
# --------------------------------------------------------------------- #


class FrameBracketCoefficientDefinition(Definition):
    """``⟨e^c, [e_a, e_b]⟩ → γ^c_ab`` — the definition of the
    anholonomy coefficient (PDF item 9b).

    Fires on a pairing of a coframe field against a Lie bracket of two
    frame fields of the *same* frame (matching base name and bundle on
    all three legs).
    """

    name = "anholonomy: ⟨e^c, [e_a, e_b]⟩ = γ^c_ab"
    anchor = Pairing

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        alpha, X = expr.alpha, expr.X
        if not isinstance(alpha, CoframeField):
            return False
        if not isinstance(X, LieBracketVF):
            return False
        if not (
            isinstance(X.X, FrameField) and isinstance(X.Y, FrameField)
        ):
            return False
        legs = (X.X, X.Y)
        return all(
            leg.base_name == alpha.base_name and leg.bundle == alpha.bundle
            for leg in legs
        )

    def rewrite(self, expr: Expr) -> Expr:
        alpha = expr.alpha
        br = expr.X
        fr = Frame(alpha.base_name, bundle=alpha.bundle)
        return anholonomy_coefficient(fr, alpha.index, br.X.index, br.Y.index)


class HolonomicFrameDefinition(Definition):
    """``[e_a, e_b] → 0`` for a frame declared holonomic (coordinate
    frame): the opt-in ``γ = 0`` mode."""

    anchor = LieBracketVF

    def __init__(self, fr: Frame) -> None:
        if not isinstance(fr, Frame):
            raise TypeError("HolonomicFrameDefinition expects a Frame")
        self._frame = fr
        self.name = f"holonomic frame {fr.name}: [e_a, e_b] = 0"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        legs = (expr.X, expr.Y)
        return all(
            isinstance(leg, FrameField)
            and leg.base_name == self._frame.name
            and leg.bundle == self._frame.bundle
            for leg in legs
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


# --------------------------------------------------------------------- #
# Frame writing of the bracket properties (item 9c, frame half)          #
# --------------------------------------------------------------------- #


def _engine(registry, holonomic_frames=()):
    from jacopy.central.tangent.engine import tangent_engine

    return tangent_engine(
        registry=registry, holonomic_frames=holonomic_frames
    )


def prove_gamma_antisymmetry(
    fr: Frame,
    upper: IndexLike,
    a: IndexLike,
    b: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``⟨e^c, [e_a, e_b]⟩ = −⟨e^c, [e_b, e_a]⟩`` — bracket
    antisymmetry written on the frame (``γ^c_ab = −γ^c_ba``)."""
    lhs = Pairing(fr.dual().field(upper), lie_bracket(fr.field(a), fr.field(b)))
    rhs = Neg(
        Pairing(fr.dual().field(upper), lie_bracket(fr.field(b), fr.field(a)))
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_coframe_differential(
    fr: Frame,
    upper: IndexLike,
    a: IndexLike,
    b: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``de^c(e_a, e_b) = −γ^c_ab`` — the dual (Maurer-Cartan
    style) form of the anholonomy decomposition.

    Three phases of machinery close it together: the Palais rule (2.C)
    unrolls ``de^c``, frame duality (2.B) turns ``⟨e^c, e_b⟩`` into
    Kronecker deltas whose derivatives die (``is_constant``), and the
    coefficient rule (2.E) names the surviving bracket pairing.
    """
    e_up = fr.dual().field(upper)
    lhs = MultiEval(
        d(e_up), fr.field(a), fr.field(b),
        alternating=True, slot_kind="vector",
    )
    # The target stated via the coefficient itself: −γ^c_ab.
    rhs = Neg(anholonomy_coefficient(fr, upper, a, b))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_frame_jacobi(
    fr: Frame,
    upper: IndexLike,
    a: IndexLike,
    b: IndexLike,
    c: IndexLike,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
):
    """Prove the Jacobi identity written on the frame:

    ``⟨e^e, [e_a,[e_b,e_c]] + [e_b,[e_c,e_a]] + [e_c,[e_a,e_b]]⟩ = 0``.

    The nested brackets are not frame-decomposed (that is the
    Phase 4 component layer); the identity closes through the repair
    loop, which proves the inner vector-level Jacobi sum on the
    generic function ``f`` and cites it. Returns
    ``(chain, used_identities)``.
    """
    from jacopy.central.tangent.cartan import prove_with_bracket_identities

    e_up = fr.dual().field(upper)
    ea, eb, ec = fr.field(a), fr.field(b), fr.field(c)
    lhs = Sum(
        Pairing(e_up, lie_bracket(ea, lie_bracket(eb, ec))),
        Pairing(e_up, lie_bracket(eb, lie_bracket(ec, ea))),
        Pairing(e_up, lie_bracket(ec, lie_bracket(ea, eb))),
    )
    return prove_with_bracket_identities(
        lhs, Integer(0), f, registry=registry
    )


def prove_holonomic_gamma_vanishes(
    fr: Frame,
    upper: IndexLike,
    a: IndexLike,
    b: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """For a frame declared holonomic: ``⟨e^c, [e_a, e_b]⟩ = 0``."""
    lhs = Pairing(fr.dual().field(upper), lie_bracket(fr.field(a), fr.field(b)))
    return ExpandAndSimplify().prove(
        lhs,
        Integer(0),
        registry=registry,
        engine=_engine(registry, holonomic_frames=(fr,)),
    )
