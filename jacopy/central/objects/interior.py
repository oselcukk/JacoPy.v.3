"""
Interior product ``ι_X`` — contraction of a p-form with a vector field
(PDF item 8r).

Definition (PDF): for a vector field ``X`` and a p-form ``ω``

    (ι_X ω)(X₁, …, X_{p-1}) := ω(X, X₁, …, X_{p-1}),

i.e. ``X`` is inserted into the first slot of ``ω``; the result is a
(p−1)-form. ``ι_X`` is a graded anti-derivation of degree ``−1``:
``ι_X(α ∧ β) = (ι_X α) ∧ β + (−1)^{|α|} α ∧ (ι_X β)`` and
``ι_X ∘ ι_X = 0``.

Implementation: :class:`Interior` is a subclass of
:class:`~jacopy.algebra.derivation.Derivation` (degree ``−1``) and
carries the vector to contract with. ``ι_X ω`` is an
:class:`~jacopy.algebra.derivation.Act` node; its degree is computed
automatically as ``|ι_X| + |ω| = |ω| − 1``.

**Central-code view.** On ``TM`` this is the usual interior product;
for a general bundle ``E``, ``ι`` is contraction on ``Λ^p E^*`` with a
section of ``E``. Insertion into the contraction slot (the MultiEval
expansion of the definition above) will be added as an engine rule
(Phase 2, together with d/L); this layer sets up the operator and its
degree law.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr


class Interior(Derivation):
    """``ι_X`` — the interior product operator (degree ``−1``).

    Parameters
    ----------
    X
        The vector field to contract with (usually a
        :class:`~jacopy.central.objects.vector_field.VectorField`).
    name
        Optional display name; defaults to ``"ι_X"``.
    """

    __slots__ = ("_vector",)

    def __init__(self, X: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(X, Expr):
            raise TypeError("Interior requires an Expr vector field")
        display = name if name is not None else f"ι_{X._repr_inner()}"
        super().__init__(display, degree=-1)
        self._vector = X

    @property
    def vector(self) -> Expr:
        return self._vector

    # Slot protocol (Phase 6.E): the vector is a rewritable slot, so
    # engine rules can normalize INSIDE it (e.g. a sharp whose form
    # slot carries a scalar factor: ι_{Π(f·η)} → ι_{f·Πη} → f·ι_{Πη}
    # via the sharp linearity + the interior vector linearity).
    @property
    def rewritable_slots(self):
        return (self._vector,)

    def with_slots(self, X: Expr) -> "Interior":
        return Interior(X)

    def _key(self) -> Any:
        return (self._name, self._degree, self._vector)


def interior(X: Expr, *, name: Optional[str] = None) -> Interior:
    """Build the operator ``ι_X`` (for the vector field ``X``)."""
    return Interior(X, name=name)


def contract_all(omega: Expr, *vectors: Expr) -> Expr:
    """Iterated contraction ``ι_{X_k} … ι_{X_1} ω`` — the alternating
    case of PDF item 8w's "generalization to arbitrary numbers".

    Fixing ``j`` slots of a k-linear alternating map yields the
    (k−j)-linear map ``ω(X_1, …, X_j, ·, …, ·)``; for alternating
    forms that partial evaluation IS the iterated interior product
    (proved for the 3-form instance in the Phase 2.F tests). The
    symmetric bilinear case is the musical ``♭``/``♯`` pair; general
    (q, r)-tensor slot views arrive with the component layer (Phase 4).
    """
    if not isinstance(omega, Expr):
        raise TypeError("contract_all expects an Expr form")
    if not vectors:
        raise ValueError("contract_all needs at least one vector")
    out: Expr = omega
    for X in vectors:
        if not isinstance(X, Expr):
            raise TypeError("contract_all vectors must be Expr")
        out = Act(Interior(X), out)
    return out


def contract(X: Expr, omega: Expr) -> Act:
    """``ι_X ω`` — contraction of the p-form ``ω`` with ``X`` (degree ``|ω|−1``).

    Yields the same result as ``interior(X)(omega)`` (Derivation
    ``__call__``); this function makes the intent readable.
    """
    if not isinstance(X, Expr) or not isinstance(omega, Expr):
        raise TypeError("contract arguments must be Expr")
    return Act(Interior(X), omega)
