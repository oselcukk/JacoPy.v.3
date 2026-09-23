"""
The EXTERIOR-degree contract (Faz 8 step 2a).

Two gradings live in the package and they must not be confused:

* the OPERATOR grading of :func:`jacopy.algebra.derivation.degree_of`
  — ``|d| = 1``, ``|ι_X| = −1``, ``|X| = 0`` for a vector field seen as
  a derivation, ``|D₁∘D₂| = |D₁| + |D₂|``;
* the EXTERIOR grading of this module — the degree of an element of
  the exterior algebra ``Λ^•T*M ⊕ Λ^•TM``: a function is 0, a vector
  field is 1, a ``p``-form is ``p``, a ``p``-vector is ``p``, a wedge
  adds, a bilinear combination inherits, and a calculus operator
  shifts (``d: +1``, ``ι_X: −1``, ``ι_P: −p``, ``ℒ, ∇: 0``, ``♭/♯``
  between vectors and 1-forms).

The graded-commutativity of ``∧`` is governed by the exterior grading,
and three private helpers used to compute it independently
(``research.engine_assembly._wedge_degree``,
``central.objects.multivector_interior._multivector_degree`` and the
parity test of ``algorithms.normalize_alternating``); the 2026-09-09,
09-10 and 09-22 audits each found one of them grading an odd factor as
even. They now all read :func:`exterior_degree`, which itself reads
the common type query :func:`jacopy.central.objects.kind.kind_of` —
one dispatch table, one answer.

The contract:

* :func:`exterior_degree` returns a :class:`~jacopy.core.symbolic_degree.Degree`
  (possibly SYMBOLIC — a ``p``-form with ``p`` a degree variable
  keeps ``p``) or the single :data:`Unknown` object. Symbolic and
  unknown are different answers: ``p − 1`` is a known degree whose
  value is open, ``Unknown`` means the expression is not (known to
  be) a homogeneous element of the exterior algebra at all — an
  operator, a composition ``X∘Y``, an action nobody defined, an
  undeclared symbol, a known type mismatch.
* :func:`as_concrete` turns a ``Degree | Unknown`` into an ``int`` or
  ``None``; a rule that needs a concrete integer concretizes itself
  and stays inert otherwise (soundness over completeness).
* :func:`exterior_parity` gives the parity ``0 | 1 | None`` — the
  quantity the graded sign of ``∧`` actually depends on; a symbolic
  degree with an even symbolic part still has a parity.
* ``Unknown`` is shared with :class:`~jacopy.central.objects.kind.Kind`
  (``Kind.degree is Unknown``); there is no second sentinel and no
  ``None``.
"""

from __future__ import annotations

from typing import Optional, Union

from jacopy.core.expr import Expr
from jacopy.core.symbolic_degree import Degree


class UnknownDegree:
    """The one ``Unknown`` degree (a singleton; ``UnknownDegree() is
    Unknown``). It compares equal to nothing but itself — in
    particular not to ``0``, not to a :class:`Degree` and not to
    ``None`` — so an accidental ``degree == 0`` on an unknown is
    False, never True."""

    __slots__ = ()
    _instance: Optional["UnknownDegree"] = None

    def __new__(cls) -> "UnknownDegree":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Unknown"

    def __reduce__(self):
        return (UnknownDegree, ())


Unknown: UnknownDegree = UnknownDegree()

DegreeOrUnknown = Union[Degree, UnknownDegree]


def is_unknown(value: object) -> bool:
    """True exactly for :data:`Unknown`."""
    return value is Unknown


def as_concrete(value: DegreeOrUnknown) -> Optional[int]:
    """The integer value of a concrete :class:`Degree`; ``None`` for a
    symbolic degree and for :data:`Unknown`."""
    if isinstance(value, Degree):
        return value.as_int()
    return None


def exterior_degree(expr: Expr, registry=None) -> DegreeOrUnknown:
    """The exterior degree of ``expr`` (module docstring) — the
    ``degree`` field of :func:`~jacopy.central.objects.kind.kind_of`.

    ``registry`` grades declared symbols (a ``functions(...)`` symbol
    is a function only in its registry; without one it is
    :data:`Unknown`, never guessed)."""
    from jacopy.central.objects.kind import kind_of  # noqa: WPS433 - layering: the dispatch over the central objects lives there

    return kind_of(expr, registry).degree


def concrete_exterior_degree(expr: Expr, registry=None) -> Optional[int]:
    """:func:`exterior_degree` concretized: an ``int``, or ``None``
    when the degree is symbolic or unknown."""
    return as_concrete(exterior_degree(expr, registry))


def exterior_parity(expr: Expr, registry=None) -> Optional[int]:
    """Parity (``0`` even, ``1`` odd) of the exterior degree, or
    ``None`` when it is unknown or undecidable symbolically."""
    deg = exterior_degree(expr, registry)
    if isinstance(deg, Degree):
        return deg.parity()
    return None


__all__ = [
    "UnknownDegree",
    "Unknown",
    "DegreeOrUnknown",
    "is_unknown",
    "as_concrete",
    "exterior_degree",
    "concrete_exterior_degree",
    "exterior_parity",
]
