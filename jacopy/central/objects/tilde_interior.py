"""
Tilde interior product ``ι̃_ω`` — contraction of a p-vector field with
a 1-form (PDF item 8s).

The **covariant twin** of
:class:`~jacopy.central.objects.interior.Interior`: while ``ι_X``
contracts a p-form with a vector, ``ι̃_ω`` contracts a p-vector field
with a 1-form ``ω``:

    (ι̃_ω π)(ω₁, …, ω_{p-1}) := π(ω, ω₁, …, ω_{p-1}),

the result is a (p−1)-vector field. ``ι̃_ω`` is a degree ``−1`` graded
anti-derivation (on multivectors). In Poisson geometry (Phase 5) the
tilde exterior derivative ``d̃`` and the tilde Lie derivative ``L̃``
are built from the Koszul bracket through this ``ι̃``.

The implementation is symmetric with
:class:`~jacopy.central.objects.interior.Interior`: a
:class:`TildeInterior` is a
:class:`~jacopy.algebra.derivation.Derivation` (degree ``−1``) carrying
the contracting 1-form; ``ι̃_ω π`` is an
:class:`~jacopy.algebra.derivation.Act` of degree ``|π| − 1``.

**Grading contact point.** The multivector grading lives on
:class:`~jacopy.central.objects.multivector.PVector` (a p-vector has
degree ``p``), while a plain
:class:`~jacopy.central.objects.vector_field.VectorField` is a degree-0
operator. Contracting a plain vector field with a 1-form is exactly the
canonical pairing, ``ι̃_ω X = ⟨ω, X⟩``, so that case is routed to
:class:`~jacopy.core.pairing.Pairing` directly (a scalar, degree 0)
instead of building a degree ``−1`` ``Act`` node.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr
from jacopy.core.pairing import Pairing


class TildeInterior(Derivation):
    """``ι̃_ω`` — tilde interior product operator (degree ``−1``).

    Parameters
    ----------
    omega
        The contracting 1-form.
    name
        Optional display name; defaults to ``"ι̃_ω"``.
    """

    __slots__ = ("_form",)

    def __init__(self, omega: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("TildeInterior requires an Expr 1-form")
        display = name if name is not None else f"ι̃_{omega._repr_inner()}"
        super().__init__(display, degree=-1)
        self._form = omega

    @property
    def form(self) -> Expr:
        return self._form

    def _key(self) -> Any:
        return (self._name, self._degree, self._form)

    def __call__(self, arg: Expr) -> Union[Act, Pairing]:
        """``ι̃_ω(π)`` — route a plain vector field to the pairing.

        For a :class:`VectorField` argument the contraction is the
        canonical scalar ``⟨ω, X⟩``; for genuine multivectors it stays
        an inert ``Act`` node of degree ``|π| − 1``.
        """
        from jacopy.central.objects.vector_field import VectorField
        if isinstance(arg, VectorField):
            return Pairing(self._form, arg)
        return Act(self, arg)


def tilde_interior(omega: Expr, *, name: Optional[str] = None) -> TildeInterior:
    """Build the ``ι̃_ω`` operator (for the 1-form ``ω``)."""
    return TildeInterior(omega, name=name)


def tilde_contract(omega: Expr, pi: Expr) -> Union[Act, Pairing]:
    """``ι̃_ω π`` — contraction of the p-vector ``π`` with ``ω``.

    Degree ``|π| − 1`` for a genuine multivector
    (:class:`~jacopy.central.objects.multivector.PVector`); for a plain
    :class:`~jacopy.central.objects.vector_field.VectorField` the result
    is the canonical pairing ``⟨ω, X⟩`` (a scalar, degree 0).
    """
    if not isinstance(omega, Expr) or not isinstance(pi, Expr):
        raise TypeError("tilde_contract arguments must be Expr")
    return TildeInterior(omega)(pi)
