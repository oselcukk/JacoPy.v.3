"""
Differential forms and their pairing with vector fields (PDF items
8g-h, 8t).

* 1-forms (``ω, η, ξ``) and arbitrary **p-forms** — graded covariant
  objects.
* The action of a 1-form on a vector field ``ω(U) = ⟨ω, U⟩`` and its
  symmetric counterpart (the action of a vector on a 1-form) — both
  produce the same scalar, the canonical pairing
  (:class:`~jacopy.core.pairing.Pairing`).

**Central-code view.** A p-form is a section of the exterior power
``Λ^p E^*`` of a bundle ``E``. Taking ``E = TM`` (the usual case) these
are ordinary differential forms; for a general algebroid ``E`` they are
sections of ``Λ^p E^*``. A form carries a reference to its bundle
(default :data:`~jacopy.central.objects.bundle.TM`).

Implementation: :class:`Form` is an :class:`~jacopy.core.expr.Atom` —
symmetric with :class:`~jacopy.central.objects.vector_field.VectorField`,
it *carries its degree and bundle on the instance* (no registry needed).
The call syntax ``ω(U₁, …, U_k)`` performs a **full evaluation** and
enforces arity whenever the degree is a concrete integer; a partial
contraction is a different operation (``ι_X ω``, see
:mod:`~jacopy.central.objects.interior`).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.core.expr import Atom, Expr
from jacopy.core.pairing import Pairing
from jacopy.core.multi_eval import MultiEval
from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree
from jacopy.central.objects.bundle import Bundle, TM


class Form(Atom):
    """A p-form; on ``TM`` the usual differential form.

    Parameters
    ----------
    name
        Display name (e.g. ``"ω"``).
    degree
        Form degree ``p`` (1 for a 1-form). Symbolic degrees
        (``Degree.var("p")``) are also accepted.
    bundle
        Owning bundle; defaults to :data:`TM`.

    Notes
    -----
    Equality is structural over ``(name, degree, bundle)``. Since the
    degree is carried on the instance,
    :func:`~jacopy.algebra.derivation.degree_of` needs no registry.
    """

    __slots__ = ("_name", "_degree", "_bundle")

    def __init__(
        self,
        name: str,
        *,
        degree: DegreeLike,
        bundle: Optional[Bundle] = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Form name must be a non-empty str")
        if bundle is not None and not isinstance(bundle, Bundle):
            raise TypeError("bundle must be a Bundle instance")
        self._name = name
        self._degree = as_degree(degree)
        self._bundle = bundle if bundle is not None else TM

    @property
    def name(self) -> str:
        return self._name

    @property
    def degree(self) -> Degree:
        return self._degree

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    def _key(self) -> Any:
        return (self._name, self._degree, self._bundle)

    def _repr_inner(self) -> str:
        return self._name

    def __call__(self, *args: Expr):
        """``ω(X₁, …, X_k)`` — **full** evaluation on vector fields.

        When the degree ``p`` is a concrete integer, exactly ``p``
        arguments are required (this is what distinguishes evaluation
        from partial contraction):

        * ``p == 1``: ``ω(X)`` → the canonical pairing ``⟨ω, X⟩``
          (:class:`~jacopy.core.pairing.Pairing`), a scalar.
        * ``p ≥ 2``: ``ω(X₁, …, X_p)`` → an alternating
          :class:`~jacopy.core.multi_eval.MultiEval`
          (``ω(X₁, …, X_p) = det[ω_i(X_j)]``, expansion rule in the
          engine layer), a scalar.
        * Any other argument count raises :class:`TypeError` — a
          partial contraction with a single vector is ``ι_X ω``; see
          :func:`~jacopy.central.objects.interior.contract`.

        When the degree is symbolic, arity cannot be checked; the call
        builds an alternating :class:`MultiEval` and the caller asserts
        the arity (same policy as
        :func:`~jacopy.core.multi_eval.validate_arity`).
        """
        if not args:
            raise TypeError("Form call requires at least one argument")
        for a in args:
            if not isinstance(a, Expr):
                raise TypeError("Form call arguments must be Expr")
        p = self._degree.as_int()
        if p is not None:
            if len(args) != p:
                raise TypeError(
                    f"{self._name} is a {p}-form: full evaluation takes "
                    f"exactly {p} argument(s), got {len(args)}. For a "
                    "partial contraction use contract(X, omega) (ι_X ω)."
                )
            if p == 1:
                return Pairing(self, args[0])
            return MultiEval(self, *args, alternating=True, slot_kind="vector")
        return MultiEval(self, *args, alternating=True, slot_kind="vector")


def forms(
    names: str,
    *,
    degree: DegreeLike,
    bundle: Optional[Bundle] = None,
) -> Tuple[Form, ...]:
    """Create one or more p-forms.

    ``forms("ω η", degree=1)`` returns two 1-forms. Even for a single
    name the return value is a tuple: ``(ω,) = forms("ω", degree=1)``.
    No registry is needed.
    """
    if not isinstance(names, str):
        raise TypeError("names must be a whitespace-separated str")
    parts = tuple(names.split())
    if not parts:
        raise ValueError("at least one name is required")
    return tuple(Form(n, degree=degree, bundle=bundle) for n in parts)


def form_on_vector(omega: Expr, X: Expr) -> Pairing:
    """Action of a 1-form on a vector field ``ω(X) = ⟨ω, X⟩`` (PDF item 8t)."""
    if not isinstance(omega, Expr) or not isinstance(X, Expr):
        raise TypeError("form_on_vector arguments must be Expr")
    return Pairing(omega, X)


def vector_on_form(X: Expr, omega: Expr) -> Pairing:
    """Action of a vector field on a 1-form ``X ⌟ ω = ⟨ω, X⟩`` (PDF item 8t).

    Produces the same scalar as the action of the 1-form on the vector;
    the direction only marks the reading intent.
    """
    if not isinstance(omega, Expr) or not isinstance(X, Expr):
        raise TypeError("vector_on_form arguments must be Expr")
    return Pairing(omega, X)
