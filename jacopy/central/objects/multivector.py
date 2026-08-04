"""
p-vector fields (multivectors) — PDF item 8i.

A **p-vector field** is a section of the exterior power ``Λ^p E`` of a
bundle ``E``. Taking ``E = TM`` (the usual case) these are ordinary
multivectors (``Λ^p TM``). The Poisson bivector ``π`` is a 2-vector
(``p = 2``) and the core input of the Schouten-Nijenhuis bracket
(Phase 2).

:class:`PVector` is the contravariant twin of
:class:`~jacopy.central.objects.form.Form`: it carries its degree
(``p``) and bundle on the instance. It is antisymmetric in its covector
arguments (``π(α, β) = −π(β, α)``); the call syntax builds an
alternating covector-slot :class:`~jacopy.core.multi_eval.MultiEval`.

**Grading convention (two gradings, one tree).** The multivector
grading (a p-vector has degree ``p``) lives on :class:`PVector` only.
A :class:`~jacopy.central.objects.vector_field.VectorField` is a
degree-0 *operator* (a derivation acting on functions, so that
``U(f)`` is again degree 0); to treat a vector field as a 1-vector in
the multivector grading, declare it as ``PVector(name, degree=1)``.
Consequently ``Wedge(X, Y)`` of two :class:`VectorField`'s does *not*
carry the bivector degree 2 — build multivectors from
:class:`PVector`'s (or wait for the SN-bracket lift in Phase 2).
The one contact point that is resolved structurally: contracting a
plain vector field with a 1-form is the canonical pairing, and
:func:`~jacopy.central.objects.tilde_interior.tilde_contract` routes
that case to :class:`~jacopy.core.pairing.Pairing` directly.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.core.expr import Atom, Expr
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree
from jacopy.central.objects.bundle import Bundle, TM


class PVector(Atom):
    """A p-vector field; on ``TM`` the usual multivector.

    Parameters
    ----------
    name
        Display name (e.g. ``"π"``).
    degree
        p-vector degree ``p`` (2 for a bivector). Symbolic degrees are
        also accepted.
    bundle
        Owning bundle; defaults to :data:`TM`.
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
            raise ValueError("PVector name must be a non-empty str")
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
        """``π(α₁, …, α_k)`` — **full** evaluation on 1-forms.

        Mirrors :meth:`Form.__call__ <jacopy.central.objects.form.Form.__call__>`
        on the contravariant side. When the degree ``p`` is a concrete
        integer, exactly ``p`` arguments are required:

        * ``p == 1``: ``π(α)`` → the canonical pairing ``⟨α, π⟩``.
        * ``p ≥ 2``: an alternating covector-slot
          :class:`~jacopy.core.multi_eval.MultiEval` (scalar).
        * Any other count raises :class:`TypeError` — a partial
          contraction with a single 1-form is ``ι̃_ω π``; see
          :func:`~jacopy.central.objects.tilde_interior.tilde_contract`.

        For a symbolic degree, arity is unenforced and an alternating
        :class:`MultiEval` is built (the caller asserts the arity).
        """
        if not args:
            raise TypeError("PVector call requires at least one argument")
        for a in args:
            if not isinstance(a, Expr):
                raise TypeError("PVector call arguments must be Expr")
        p = self._degree.as_int()
        if p is not None:
            if len(args) != p:
                raise TypeError(
                    f"{self._name} is a {p}-vector: full evaluation takes "
                    f"exactly {p} argument(s), got {len(args)}. For a "
                    "partial contraction use tilde_contract(omega, pi) "
                    "(ι̃_ω π)."
                )
            if p == 1:
                return Pairing(args[0], self)
            return MultiEval(
                self, *args, alternating=True, slot_kind="covector"
            )
        return MultiEval(self, *args, alternating=True, slot_kind="covector")


def p_vectors(
    names: str,
    *,
    degree: DegreeLike,
    bundle: Optional[Bundle] = None,
) -> Tuple[PVector, ...]:
    """Create one or more p-vector fields.

    ``p_vectors("π ρ", degree=2)`` returns two bivectors. Even for a
    single name the return value is a tuple. No registry is needed.
    """
    if not isinstance(names, str):
        raise TypeError("names must be a whitespace-separated str")
    parts = tuple(names.split())
    if not parts:
        raise ValueError("at least one name is required")
    return tuple(PVector(n, degree=degree, bundle=bundle) for n in parts)


def bivector(name: str, *, bundle: Optional[Bundle] = None) -> PVector:
    """A single 2-vector (``p = 2``) — shortcut for the Poisson bivector."""
    pieces = name.split()
    if len(pieces) != 1:
        raise ValueError("bivector takes exactly one name")
    return PVector(pieces[0], degree=2, bundle=bundle)
