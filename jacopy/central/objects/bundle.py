"""
Bundle — the context that central-code objects live over (PDF item 8a).

**Central-code principle.** Every object is defined so that it works
relative to a vector bundle ``E`` (bundle) parameter; taking
``E = TM``, ``ρ_E = id_TM`` and ``[·,·]_E = [·,·]_Lie`` reduces the
algebroid version to the usual one.

This module builds the **skeleton** of the bundle abstraction:

* :class:`Bundle` — the identity of a bundle (name, dimension) and,
  later, slots for the anchor ``ρ_E: E → TM`` and the bracket
  ``[·,·]_E``. These slots will be filled in Phase 3
  (``jacopy.central.algebroid``).
* :class:`TangentBundle` — ``TM``; the "usual" case of the central
  code. The anchor is the identity and the bracket is the Lie bracket
  (these are wired up in Phases 2-3).
* :data:`TM` — the process-wide shared symbolic ``TangentBundle``
  singleton; the default bundle for objects declared without one.

A Bundle is **not** an :class:`~jacopy.core.expr.Expr` — it is not a
mathematical expression but a context that objects belong to. Its
equality is structural over ``(type, name, dimension)``; thus two
``tangent_bundle()`` calls yield equal bundles and sections with the
same name are considered to belong to the same bundle.
"""

from __future__ import annotations

from typing import Any, Optional


class Bundle:
    """A vector bundle ``E`` (central-code context).

    Parameters
    ----------
    name
        The display name of the bundle (e.g. ``"TM"``, ``"E"``).
    dim
        Optional dimension. If ``None`` the bundle has **symbolic**
        dimension (the mode in which abstract proofs run).

    Notes
    -----
    The anchor ``ρ_E`` and the bracket ``[·,·]_E`` are not carried yet
    in this skeleton layer; the general algebroid structure will be
    added in Phase 3 under :mod:`jacopy.central.algebroid`. For now a
    bundle carries only an identity + dimension and marks which
    context objects (sections, forms, tensors) belong to.
    """

    __slots__ = ("_name", "_dim")

    def __init__(self, name: str, *, dim: Optional[int] = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Bundle name must be a non-empty str")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise ValueError("Bundle dim must be None or a positive int")
        self._name = name
        self._dim = dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def is_tangent(self) -> bool:
        """Is this bundle ``TM`` (the usual case)?"""
        return isinstance(self, TangentBundle)

    def _key(self) -> Any:
        return (type(self).__name__, self._name, self._dim)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bundle) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        d = "" if self._dim is None else f", dim={self._dim}"
        return f"{type(self).__name__}({self._name!r}{d})"


class TangentBundle(Bundle):
    """``TM`` — the "usual" case of the central code.

    The anchor is the identity ``ρ = id_TM`` and the bracket is the Lie
    bracket of vector fields. These two structures will be wired up in
    Phase 2 (Lie bracket, d, L) and Phase 3 (general anchor); in this
    skeleton layer ``TangentBundle`` is a marker. ``name`` is always
    ``"TM"``.
    """

    __slots__ = ()

    def __init__(self, *, dim: Optional[int] = None) -> None:
        super().__init__("TM", dim=dim)


def tangent_bundle(*, dim: Optional[int] = None) -> TangentBundle:
    """Create the bundle ``TM`` (with an optional dimension)."""
    return TangentBundle(dim=dim)


#: Process-wide symbolic ``TM`` singleton; the default bundle of
#: central objects declared without one. Symbolic dimension
#: (``dim=None``) is the abstract-proof mode.
TM: TangentBundle = tangent_bundle()
