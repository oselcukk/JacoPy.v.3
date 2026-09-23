"""
The common TYPE query — kind, exterior degree and owning bundle of an
expression (Faz 8 step 1c/2a; audit rounds 2026-09-22/23: "degree is
not a type", "a foreign bundle is not unknown", "a known mixed sum is
not unknown").

:func:`kind_of` returns a :class:`Kind` with

* ``kind`` — ``"function"``, ``"vector"``, ``"form"``,
  ``"multivector"``, ``"tensor"``, ``"operator"``, ``"zero"``,
  ``"unknown"`` or ``"mismatch"`` (a sum of DIFFERENT known kinds or
  degrees — known to be ill-typed, distinct from unknown);
* ``degree`` — the exterior degree for forms / multivectors (``None``
  when symbolic);
* ``signature`` — ``(q, r)`` for tensors;
* ``bundle`` — the owning :class:`~jacopy.central.objects.bundle.Bundle`
  when the expression carries one (``None`` when not determinable).

Every answer is derived from the node's own metadata (a
:class:`~jacopy.central.objects.form.Form` is a form because it is a
Form — a bivector of degree 2 is NOT a 2-form); bilinear combinations
are looked through (sums, negations, scalar products), the calculus
operators shift degrees (``d``, ``ι``, ``ℒ``, ``∇``, ``ι_P``, the
musical maps), and anything else is ``unknown`` — never guessed.
Consumers (section slots, the projector provers) turn ``mismatch`` and
a known wrong kind/degree/bundle into an error, and ``unknown`` into an
explicit "unverified" mark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum, Symbol
from jacopy.core.symbolic_degree import Degree


@dataclass(frozen=True)
class Kind:
    kind: str
    degree: Optional[int] = None
    signature: Optional[Tuple[int, int]] = None
    bundle: object = None

    @property
    def known(self) -> bool:
        return self.kind not in ("unknown", "mismatch")

    def same_type(self, other: "Kind") -> bool:
        """Same kind, degree and signature; bundles agree when both known."""
        if self.kind != other.kind or self.degree != other.degree or self.signature != other.signature:
            return False
        if self.bundle is not None and other.bundle is not None and self.bundle != other.bundle:
            return False
        return True


UNKNOWN = Kind("unknown")
ZERO = Kind("zero")
FUNCTION = Kind("function")


def _int(deg) -> Optional[int]:
    if isinstance(deg, Degree):
        try:
            return deg.as_int()
        except ValueError:
            return None
    return deg if isinstance(deg, int) else None


def _merge(kinds) -> Kind:
    """Combine the kinds of the terms of a sum: zeros drop out, one
    unknown makes the sum unknown, two different known types make it
    a MISMATCH (a known error, not an unknown)."""
    kinds = [k for k in kinds if k.kind != "zero"]
    if not kinds:
        return ZERO
    if any(not k.known for k in kinds):
        if any(k.kind == "mismatch" for k in kinds):
            return Kind("mismatch")
        return UNKNOWN
    first = kinds[0]
    bundle = first.bundle
    for k in kinds[1:]:
        if not first.same_type(k):
            return Kind("mismatch")
        if bundle is None:
            bundle = k.bundle
    return Kind(first.kind, first.degree, first.signature, bundle)


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
        return FUNCTION if is_scalar_function(expr, registry) else UNKNOWN

    # --- bilinear combinations --------------------------------------
    if isinstance(expr, Neg):
        return kind_of(expr.arg, registry)
    if isinstance(expr, Sum):
        return _merge(kind_of(c, registry) for c in expr.children)
    if isinstance(expr, Product):
        kinds = [kind_of(c, registry) for c in expr.children]
        if any(k.kind == "zero" for k in kinds):
            return ZERO
        rest = [k for k in kinds if k.kind != "function"]
        if not rest:
            return FUNCTION
        if len(rest) == 1:
            return rest[0]
        return UNKNOWN  # composition / a product of two geometric factors

    # --- the central objects ----------------------------------------
    from jacopy.central.objects.form import Form
    from jacopy.central.objects.metric import InverseMetric, Metric
    from jacopy.central.objects.multivector import PVector
    from jacopy.central.objects.tensor import Tensor
    from jacopy.central.objects.vector_field import VectorField

    if isinstance(expr, VectorField):
        return Kind("vector", bundle=expr.bundle)
    if isinstance(expr, Form):
        return Kind("form", _int(expr.degree), bundle=expr.bundle)
    if isinstance(expr, PVector):
        return Kind("multivector", _int(expr.degree), bundle=expr.bundle)
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
        return kind_of(expr.body, registry)
    if isinstance(expr, (Symmetrization, Antisymmetrization)):
        return kind_of(expr.arg, registry)
    if isinstance(expr, HodgeStar):
        inner = kind_of(expr.arg, registry)
        n = _int(expr.dim)
        if inner.kind == "form" and inner.degree is not None and n is not None:
            return Kind("form", n - inner.degree, bundle=inner.bundle)
        return UNKNOWN
    if isinstance(expr, Wedge):
        parts = [kind_of(c, registry) for c in expr.children]
        if any(p.kind == "zero" for p in parts):
            return ZERO
        if all(p.kind == "form" and p.degree is not None for p in parts):
            return Kind("form", sum(p.degree for p in parts), bundle=_common_bundle(parts))
        if all(p.kind in ("vector", "multivector") for p in parts):
            degs = [1 if p.kind == "vector" else p.degree for p in parts]
            if all(d is not None for d in degs):
                return Kind("multivector", sum(degs), bundle=_common_bundle(parts))
        return UNKNOWN
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

    # --- brackets of vector fields ----------------------------------
    from jacopy.algebra.commutator import Commutator
    from jacopy.algebra.lie_bracket_vf import LieBracketVF

    if isinstance(expr, LieBracketVF):
        return _merge([kind_of(expr.X, registry), kind_of(expr.Y, registry)]) if False else Kind(
            "vector", bundle=kind_of(expr.X, registry).bundle
        )
    if isinstance(expr, Commutator):
        return UNKNOWN

    # --- derivation atoms: vector-valued ones carry a wedge lift ----
    if isinstance(expr, Derivation):
        lift = getattr(expr, "wedge_degree", None)
        if isinstance(lift, Degree) and lift == Degree.const(1):
            return Kind("vector", bundle=getattr(expr, "bundle", None))
        return Kind("operator")

    # --- other atoms: their own degree (0 = function, k = k-form) ---
    deg = getattr(expr, "degree", None)
    if isinstance(deg, Degree):
        k = _int(deg)
        if k is None:
            return UNKNOWN
        if k == 0:
            return FUNCTION
        return Kind("form", k, bundle=getattr(expr, "bundle", None))
    return UNKNOWN


def _as_signature(k: Kind) -> Optional[Tuple[int, int]]:
    if k.kind == "tensor":
        return k.signature
    if k.kind == "vector":
        return (1, 0)
    if k.kind == "form" and k.degree is not None:
        return (0, k.degree)
    if k.kind == "multivector" and k.degree is not None:
        return (k.degree, 0)
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


def _kind_of_act(expr, registry) -> Kind:
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
        if inner.kind == "form" and inner.degree is not None:
            return Kind("form", inner.degree + 1, bundle=inner.bundle)
        return UNKNOWN
    if isinstance(op, Interior):
        if inner.kind == "function":
            return ZERO
        if inner.kind == "form" and inner.degree is not None:
            return FUNCTION if inner.degree == 1 else Kind("form", inner.degree - 1, bundle=inner.bundle)
        return UNKNOWN
    if isinstance(op, MultivectorInterior):
        p = kind_of(op.multivector, registry)
        pdeg = 1 if p.kind == "vector" else p.degree
        if inner.kind == "form" and inner.degree is not None and pdeg is not None:
            left = inner.degree - pdeg
            return FUNCTION if left == 0 else (Kind("form", left, bundle=inner.bundle) if left > 0 else ZERO)
        return UNKNOWN
    if isinstance(op, (LieDerivative, CovariantOp)):
        return inner  # type-preserving
    if isinstance(op, Flat):
        return Kind("form", 1) if inner.kind == "vector" else UNKNOWN
    if isinstance(op, Sharp):
        return Kind("vector") if inner.kind == "form" and inner.degree == 1 else UNKNOWN
    if isinstance(op, EndoVF):
        return Kind("vector", bundle=inner.bundle) if inner.kind == "vector" else UNKNOWN
    if isinstance(op, Derivation):
        lift = getattr(op, "wedge_degree", None)
        if isinstance(lift, Degree) and lift == Degree.const(1):
            # a vector field acting on a function is a function
            return FUNCTION if inner.kind == "function" else UNKNOWN
    return UNKNOWN
