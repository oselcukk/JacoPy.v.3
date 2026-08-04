"""
Vector fields and their action on functions ``U(f)`` (PDF items 8e-f).

**Central-code view.** A vector field is really a *section* of a bundle
``E``. Taking ``E = TM`` (the usual case) it is the ordinary vector
field; for a general algebroid ``E`` a section acts on functions
through the anchor ``ρ_E``: ``u(f) := ρ_E(u)(f)``. On ``TM`` the
anchor is the identity, so ``u(f)`` is the directional derivative
itself.

Since the anchor is not carried at this skeleton layer (Phase 3), the
action is given directly as ``Act(U, f)`` for the ``TM`` case; the
general algebroid anchor will be wired in Phase 3 as
``u(f) = ρ_E(u)(f)``.

Implementation: :class:`VectorField` is a subclass of
:class:`~jacopy.algebra.derivation.Derivation`. Hence

* ``U(f)`` → ``Act(U, f)`` (Derivation's ``__call__``) yields the
  directional derivative; nested actions like ``U(V(f))`` come for
  free,
* the Leibniz rule over products ``U(f·g) = U(f)·g + f·U(g)`` is
  applied by :mod:`jacopy.algorithms.product_rule`,
* the Lie bracket ``[U, V] = U∘V − V∘U`` (Phase 2) is built cleanly
  as operator composition.

**Grading convention.** The degree is always 0 — the *operator*
grading of the Cartan calculus (``L_X`` degree 0, ``ι_X`` degree −1,
``d`` degree +1), under which ``U(f)`` is again degree 0. The
*multivector* grading (a p-vector has degree ``p``) lives on
:class:`~jacopy.central.objects.multivector.PVector`; to treat a
vector field as a 1-vector in that grading, declare it as
``PVector(name, degree=1)``. See the note in
:mod:`~jacopy.central.objects.multivector`.

If no bundle is given the default is
:data:`~jacopy.central.objects.bundle.TM`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr
from jacopy.core.symbolic_degree import DegreeLike
from jacopy.central.objects.bundle import Bundle, TM


class VectorField(Derivation):
    """A section of a bundle; on ``TM`` the usual vector field.

    Parameters
    ----------
    name
        Display name (e.g. ``"U"``).
    bundle
        Owning bundle; defaults to
        :data:`~jacopy.central.objects.bundle.TM`.
    degree
        Grading degree; 0 for vector fields (default).

    Notes
    -----
    Equality is structural over ``(name, degree, bundle)``: two
    same-named sections of different bundles are distinct. Since the
    degree is carried by :class:`Derivation`,
    :func:`~jacopy.algebra.derivation.degree_of` needs no registry.
    """

    __slots__ = ("_bundle",)

    def __init__(
        self,
        name: str,
        *,
        bundle: Optional[Bundle] = None,
        degree: DegreeLike = 0,
    ) -> None:
        super().__init__(name, degree=degree)
        if bundle is not None and not isinstance(bundle, Bundle):
            raise TypeError("bundle must be a Bundle instance")
        self._bundle = bundle if bundle is not None else TM

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def wedge_degree(self):
        """Multivector-grading contribution inside a Wedge: a vector
        field is a 1-vector (the Wedge lift of the grading convention;
        see :mod:`jacopy.central.objects.multivector`)."""
        from jacopy.core.symbolic_degree import Degree

        return Degree.const(1)

    def _key(self) -> Any:
        return (self._name, self._degree, self._bundle)


def vector_fields(
    names: str,
    *,
    bundle: Optional[Bundle] = None,
) -> Tuple[VectorField, ...]:
    """Create one or more vector fields (sections).

    ``vector_fields("U V W")`` returns three :class:`VectorField`'s,
    each on :data:`TM` unless a bundle is given. Even for a single name
    the return value is a tuple: ``(U,) = vector_fields("U")``.

    No registry is needed — the degree is carried on each section.
    """
    if not isinstance(names, str):
        raise TypeError("names must be a whitespace-separated str")
    parts = tuple(names.split())
    if not parts:
        raise ValueError("at least one name is required")
    return tuple(VectorField(n, bundle=bundle) for n in parts)


def apply_to_function(U: VectorField, f: Expr) -> Act:
    """The directional derivative ``U(f)``.

    On ``TM`` this is directly ``Act(U, f)``. The call syntax ``U(f)``
    (Derivation ``__call__``) gives the same result; this helper exists
    to make the intent readable. For a general algebroid (Phase 3) it
    will generalize to ``ρ_E(u)(f)``.
    """
    if not isinstance(U, VectorField):
        raise TypeError("U must be a VectorField")
    if not isinstance(f, Expr):
        raise TypeError("f must be an Expr")
    return Act(U, f)
