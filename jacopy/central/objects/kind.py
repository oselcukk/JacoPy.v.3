"""
The common TYPE query — kind, exterior degree and owning bundle of an
expression (Faz 8 steps 1c/2a; audit rounds 2026-09-22/23: "degree is
not a type", "a foreign bundle is not unknown", "a known mixed sum is
not unknown").

:func:`kind_of` returns a :class:`Kind` with

* ``kind`` — ``"function"``, ``"vector"``, ``"form"``,
  ``"multivector"``, ``"tensor"``, ``"operator"``, ``"zero"``,
  ``"unknown"`` or ``"mismatch"`` (a sum of DIFFERENT known kinds or
  degrees — known to be ill-typed, distinct from unknown);
* ``degree`` — the EXTERIOR degree (see
  :mod:`jacopy.algebra.grading`): a :class:`~jacopy.core.symbolic_degree.Degree`,
  possibly symbolic, or the one shared :data:`~jacopy.algebra.grading.Unknown`
  object. A function is 0, a vector 1, a ``p``-form / ``p``-vector
  ``p``; a tensor, an operator without a wedge lift, and every
  unknown or mismatched expression are ``Unknown``;
* ``signature`` — ``(q, r)`` for tensors;
* ``bundle`` — the owning :class:`~jacopy.central.objects.bundle.Bundle`
  when the expression carries one (``None`` when not determinable).

Every answer is derived from the node's own metadata (a
:class:`~jacopy.central.objects.form.Form` is a form because it is a
Form — a bivector of degree 2 is NOT a 2-form); bilinear combinations
are looked through (sums, negations, scalar products), the calculus
operators shift degrees (``d``, ``ι``, ``ℒ``, ``∇``, ``ι_P``, the
musical maps), brackets grade through their own hook or the operand
kinds, and anything else is ``unknown`` — never guessed. Consumers
(section slots, the projector provers, the wedge sorter) turn
``mismatch`` and a known wrong kind/degree/bundle into an error or an
inert rule, and ``unknown`` into an explicit "unverified" mark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jacopy.algebra.grading import DegreeOrUnknown, Unknown, as_concrete, is_unknown
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum, Symbol
from jacopy.core.symbolic_degree import Degree


def _as_degree(value) -> DegreeOrUnknown:
    """Coerce ``int | Degree | None | Unknown`` to ``Degree | Unknown``
    (``None`` is accepted at construction only, as a spelling of
    Unknown, so that ``Kind("form", 2)`` and ``Kind("vector")`` read
    naturally)."""
    if isinstance(value, Degree):
        return value
    if value is None or is_unknown(value):
        return Unknown
    if isinstance(value, bool):
        raise TypeError("a Kind degree cannot be a bool")
    if isinstance(value, int):
        return Degree.const(value)
    raise TypeError(f"a Kind degree must be an int, a Degree or Unknown; got {value!r}")


_DEFAULT_DEGREES = {"function": Degree.const(0), "zero": Degree.const(0), "vector": Degree.const(1)}


@dataclass(frozen=True)
class Kind:
    kind: str
    degree: DegreeOrUnknown = Unknown
    signature: Optional[Tuple[int, int]] = None
    bundle: object = None

    def __post_init__(self) -> None:
        deg = _as_degree(self.degree)
        if is_unknown(deg) and self.kind in _DEFAULT_DEGREES:
            deg = _DEFAULT_DEGREES[self.kind]
        object.__setattr__(self, "degree", deg)

    @property
    def known(self) -> bool:
        return self.kind not in ("unknown", "mismatch")

    @property
    def concrete_degree(self) -> Optional[int]:
        """The exterior degree as an ``int``; ``None`` when symbolic or
        unknown (:func:`jacopy.algebra.grading.as_concrete`)."""
        return as_concrete(self.degree)

    def same_type(self, other: "Kind") -> bool:
        """Same kind, signature and degree (equal as polynomials, or
        both unknown); bundles agree when both known."""
        if self.kind != other.kind or self.signature != other.signature:
            return False
        if self.degree != other.degree:
            return False
        if self.bundle is not None and other.bundle is not None and self.bundle != other.bundle:
            return False
        return True


UNKNOWN = Kind("unknown")
ZERO = Kind("zero")
FUNCTION = Kind("function")
MISMATCH = Kind("mismatch")


def _merge_degree(a: DegreeOrUnknown, b: DegreeOrUnknown):
    """The degree of a sum of two same-kind terms: equal polynomials
    → that degree; two CONCRETE different integers → ``"mismatch"``;
    otherwise (symbolic vs anything, or an unknown) → ``Unknown`` —
    not known to agree, not known to differ."""
    if a == b and not is_unknown(a):
        return a
    ca, cb = as_concrete(a), as_concrete(b)
    if ca is not None and cb is not None and ca != cb:
        return "mismatch"
    return Unknown


def _merge(kinds) -> Kind:
    """Combine the kinds of the terms of a sum: zeros drop out, one
    unknown makes the sum unknown, two different known types make it
    a MISMATCH (a known error, not an unknown)."""
    kinds = [k for k in kinds if k.kind != "zero"]
    if not kinds:
        return ZERO
    if any(not k.known for k in kinds):
        if any(k.kind == "mismatch" for k in kinds):
            return MISMATCH
        return UNKNOWN
    first = kinds[0]
    degree = first.degree
    bundle = first.bundle
    for k in kinds[1:]:
        if first.kind != k.kind or first.signature != k.signature:
            return MISMATCH
        merged = _merge_degree(degree, k.degree)
        if merged == "mismatch":
            return MISMATCH
        degree = merged
        if bundle is None:
            bundle = k.bundle
        elif k.bundle is not None and k.bundle != bundle:
            return MISMATCH
    return Kind(first.kind, degree, first.signature, bundle)


def kind_of(expr: Expr, registry=None) -> Kind:  # noqa: C901 - one dispatch table
    """The :class:`Kind` of ``expr`` (see the module docstring)."""
    from jacopy.algebra.derivation import Act, Derivation
    from jacopy.central.calculus.scalars import is_scalar_function

    if not isinstance(expr, Expr):
        raise TypeError("kind_of expects an Expr")

    # --- literals and symbols ---------------------------------------
    if isinstance(expr, (Integer, Rational)):
        return ZERO if expr == Integer(0) else FUNCTION
    if isinstance(expr, Symbol):
        if is_scalar_function(expr, registry):
            return FUNCTION
        return _registry_graded(expr, registry)

    # --- bilinear combinations --------------------------------------
    if isinstance(expr, Neg):
        return kind_of(expr.arg, registry)
    if isinstance(expr, Sum):
        return _merge(kind_of(c, registry) for c in expr.children)
    if isinstance(expr, Product):
        # scalars times ONE geometric element keep that element's type
        # (f·X is a vector, f·ω_p a p-form); two or more geometric or
        # operator factors are a composition (X∘Y) or have no meaning
        # in the exterior algebra — unknown, not graded (roadmap 2a v3)
        kinds = [kind_of(c, registry) for c in expr.children]
        if any(k.kind == "zero" for k in kinds):
            return ZERO
        rest = [k for k in kinds if k.kind != "function"]
        if not rest:
            return FUNCTION
        if len(rest) == 1:
            return rest[0]
        return UNKNOWN

    # --- the central objects ----------------------------------------
    from jacopy.central.objects.form import Form
    from jacopy.central.objects.metric import InverseMetric, Metric
    from jacopy.central.objects.multivector import PVector
    from jacopy.central.objects.tensor import Tensor
    from jacopy.central.objects.vector_field import VectorField

    if isinstance(expr, VectorField):
        return Kind("vector", bundle=expr.bundle)
    if isinstance(expr, Form):
        return Kind("form", expr.degree, bundle=expr.bundle)
    if isinstance(expr, PVector):
        return Kind("multivector", expr.degree, bundle=expr.bundle)
    if isinstance(expr, Tensor):
        return Kind("tensor", signature=expr.signature, bundle=expr.bundle)
    if isinstance(expr, (Metric, InverseMetric)):
        return Kind("tensor", signature=expr.signature, bundle=expr.bundle)

    # --- partial evaluation: the open map's type --------------------
    from jacopy.central.objects.partial_eval import PartialEval

    if isinstance(expr, PartialEval):
        head = kind_of(expr.head, registry)
        bundle = head.bundle
        if expr.alternating and expr.slot_kind == "vector":
            return Kind("form", expr.n_open, bundle=bundle)
        if expr.alternating and expr.slot_kind == "covector":
            return Kind("multivector", expr.n_open, bundle=bundle)
        from jacopy.central.objects.tensor import signature_of

        sig = signature_of(expr)
        if sig == (0, 1):
            return Kind("form", 1, bundle=bundle)
        if sig == (1, 0):
            return Kind("vector", bundle=bundle)
        return Kind("tensor", signature=sig, bundle=bundle) if sig else UNKNOWN

    # --- structural nodes -------------------------------------------
    from jacopy.core.hodge import HodgeStar
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.core.multi_eval import MultiEval
    from jacopy.core.pairing import Pairing
    from jacopy.core.symmetrize import Antisymmetrization, Symmetrization
    from jacopy.core.tensor_product import TensorProduct
    from jacopy.core.wedge import Wedge

    if isinstance(expr, (Pairing, MultiEval)):
        return FUNCTION
    if isinstance(expr, IndexedSum):
        return kind_of(expr.body, registry)  # Σ_i body grades like its body; the bound index does not grade
    if isinstance(expr, (Symmetrization, Antisymmetrization)):
        return kind_of(expr.arg, registry)
    if isinstance(expr, HodgeStar):
        inner = kind_of(expr.arg, registry)
        if inner.kind == "form" and not is_unknown(inner.degree):
            return Kind("form", expr.dim - inner.degree, bundle=inner.bundle)
        return UNKNOWN
    if isinstance(expr, Wedge):
        return _kind_of_wedge([kind_of(c, registry) for c in expr.children])
    if isinstance(expr, TensorProduct):
        parts = [kind_of(c, registry) for c in expr.children]
        sigs = [_as_signature(p) for p in parts]
        if all(s is not None for s in sigs):
            q = sum(s[0] for s in sigs)
            r = sum(s[1] for s in sigs)
            return Kind("tensor", signature=(q, r), bundle=_common_bundle(parts))
        return UNKNOWN

    # --- operator applications --------------------------------------
    if isinstance(expr, Act):
        return _kind_of_act(expr, registry)

    # --- brackets ---------------------------------------------------
    from jacopy.algebra.commutator import Commutator
    from jacopy.algebra.lie_bracket_vf import LieBracketVF
    from jacopy.brackets.base import BracketApply

    if isinstance(expr, LieBracketVF):
        return Kind("vector", bundle=kind_of(expr.X, registry).bundle)
    if isinstance(expr, BracketApply):
        return _kind_of_bracket(expr, registry)
    if isinstance(expr, Commutator):
        return UNKNOWN

    # --- derivation atoms: vector-valued ones carry a wedge lift ----
    if isinstance(expr, Derivation):
        lift = getattr(expr, "wedge_degree", None)
        if isinstance(lift, Degree):
            if lift == Degree.const(1):
                return Kind("vector", bundle=getattr(expr, "bundle", None))
            return Kind("operator", lift)  # an operator that also lives in the wedge algebra
        return Kind("operator")

    # --- other atoms: their own degree (0 = function, k = k-form) ---
    deg = getattr(expr, "degree", None)
    if isinstance(deg, Degree):
        if deg == Degree.const(0):
            return FUNCTION
        return Kind("form", deg, bundle=getattr(expr, "bundle", None))
    return UNKNOWN


def _registry_graded(expr, registry) -> Kind:
    """A symbol declared ``Graded(k)`` in the registry is an abstract
    graded element: a function for ``k = 0`` (already caught by the
    scalar test), otherwise graded like a ``k``-form — the convention
    the degree-carrying atoms follow (0 = function, k = k-form)."""
    if registry is None:
        return UNKNOWN
    from jacopy.core.properties import Graded

    graded = registry.get(expr, Graded)
    if graded is None:
        return UNKNOWN
    deg = graded.degree
    if deg == Degree.const(0):
        return FUNCTION
    return Kind("form", deg)


def _kind_of_wedge(parts) -> Kind:
    """``∧`` adds exterior degrees: scalar factors are transparent
    (``f ∧ X = f·X``), all forms make a form, all vectors /
    multivectors make a multivector; anything else is unknown."""
    if any(p.kind == "zero" for p in parts):
        return ZERO
    geometric = [p for p in parts if p.kind != "function"]
    if not geometric:
        return FUNCTION
    if any(not p.known for p in geometric):
        return MISMATCH if any(p.kind == "mismatch" for p in geometric) else UNKNOWN
    if any(is_unknown(p.degree) for p in geometric):
        return UNKNOWN
    total = Degree.const(0)
    for p in geometric:
        total = total + p.degree
    if all(p.kind == "form" for p in geometric):
        return Kind("form", total, bundle=_common_bundle(geometric))
    if all(p.kind in ("vector", "multivector") for p in geometric):
        return Kind("multivector", total, bundle=_common_bundle(geometric))
    return UNKNOWN


def _kind_of_bracket(expr, registry) -> Kind:
    """A bracket application: the bracket's own ``apply_degree`` hook
    when it has one (the Schouten–Nijenhuis bracket grades in the
    multivector grading), else the operand kinds — two forms under a
    bracket of degree ``k`` give a form of degree ``|a|+|b|+k`` (the
    Koszul bracket), two vectors under a degree-0 bracket a vector;
    anything else is unknown."""
    a, b = kind_of(expr.a, registry), kind_of(expr.b, registry)
    hook = getattr(expr.bracket, "apply_degree", None)
    if hook is not None:
        try:
            deg = hook(expr.a, expr.b, registry)
        except ValueError:
            return UNKNOWN
        if a.kind in ("vector", "multivector") and b.kind in ("vector", "multivector"):
            return Kind("multivector", deg, bundle=_common_bundle([a, b]))
        return UNKNOWN
    shift = getattr(expr.bracket, "degree", None)
    if not isinstance(shift, Degree):
        return UNKNOWN
    if a.kind == "form" and b.kind == "form" and not (is_unknown(a.degree) or is_unknown(b.degree)):
        return Kind("form", a.degree + b.degree + shift, bundle=_common_bundle([a, b]))
    if a.kind == "vector" and b.kind == "vector" and shift == Degree.const(0):
        return Kind("vector", bundle=_common_bundle([a, b]))
    return UNKNOWN


def _as_signature(k: Kind) -> Optional[Tuple[int, int]]:
    if k.kind == "tensor":
        return k.signature
    if k.kind == "vector":
        return (1, 0)
    if k.kind == "form" and k.concrete_degree is not None:
        return (0, k.concrete_degree)
    if k.kind == "multivector" and k.concrete_degree is not None:
        return (k.concrete_degree, 0)
    if k.kind == "function":
        return (0, 0)
    return None


def _common_bundle(parts):
    bundle = None
    for p in parts:
        if p.bundle is None:
            continue
        if bundle is None:
            bundle = p.bundle
        elif bundle != p.bundle:
            return None
    return bundle


def _form_or_lower(degree: Degree, bundle) -> Kind:
    """A form of the given degree after a lowering operator: concrete
    0 is a function, concrete negative is zero, otherwise a form (a
    symbolic degree stays symbolic)."""
    n = degree.as_int()
    if n is None:
        return Kind("form", degree, bundle=bundle)
    if n == 0:
        return FUNCTION
    if n < 0:
        return ZERO
    return Kind("form", degree, bundle=bundle)


def _kind_of_act(expr, registry) -> Kind:
    """``degree(Act(D, a)) = action_shift(D, domain(a)) + degree(a)`` —
    only when ``D``'s homogeneous shift on that domain is known:
    ``X(f)`` a function, ``d: +1``, ``ι_X: −1``, ``ι_P: −p``,
    ``ℒ, ∇: 0``, ``♭/♯`` between vectors and 1-forms, an endomorphism
    on a vector. A vector field's ``Act`` on a form is NOT a Lie
    derivative — unknown."""
    from jacopy.algebra.derivation import Derivation
    from jacopy.central.calculus.bracket_calculus import ExteriorDerivative, LieDerivative
    from jacopy.central.objects.connection import CovariantOp
    from jacopy.central.objects.endomorphism import EndoVF
    from jacopy.central.objects.interior import Interior
    from jacopy.central.objects.multivector_interior import MultivectorInterior
    from jacopy.central.objects.musical import Flat, Sharp

    op, arg = expr.op, expr.arg
    inner = kind_of(arg, registry)
    if inner.kind == "zero":
        return ZERO
    if isinstance(op, ExteriorDerivative):
        if inner.kind == "function":
            return Kind("form", 1)
        if inner.kind == "form" and not is_unknown(inner.degree):
            return Kind("form", inner.degree + 1, bundle=inner.bundle)
        return UNKNOWN
    if isinstance(op, Interior):
        if inner.kind == "function":
            return ZERO
        if inner.kind == "form" and not is_unknown(inner.degree):
            return _form_or_lower(inner.degree - 1, inner.bundle)
        return UNKNOWN
    if isinstance(op, MultivectorInterior):
        p = kind_of(op.multivector, registry)
        if inner.kind == "function":
            return ZERO
        if (
            inner.kind == "form"
            and p.kind in ("vector", "multivector")
            and not (is_unknown(inner.degree) or is_unknown(p.degree))
        ):
            return _form_or_lower(inner.degree - p.degree, inner.bundle)
        return UNKNOWN
    if isinstance(op, (LieDerivative, CovariantOp)):
        return inner  # type-preserving
    if isinstance(op, Flat):
        return Kind("form", 1) if inner.kind == "vector" else UNKNOWN
    if isinstance(op, Sharp):
        return Kind("vector") if inner.kind == "form" and inner.degree == Degree.const(1) else UNKNOWN
    if isinstance(op, EndoVF):
        return Kind("vector", bundle=inner.bundle) if inner.kind == "vector" else UNKNOWN
    if isinstance(op, Derivation):
        lift = getattr(op, "wedge_degree", None)
        if isinstance(lift, Degree) and lift == Degree.const(1):
            # a vector field acting on a function is a function
            return FUNCTION if inner.kind == "function" else UNKNOWN
    return UNKNOWN


__all__ = ["Kind", "kind_of", "UNKNOWN", "ZERO", "FUNCTION", "MISMATCH"]
