"""
Connection ``∇`` and its action on a (q, r)-type tensor (PDF items
8m-n).

An affine connection ``∇`` yields, for a vector field ``X``, the
**covariant derivative** ``∇_X``. It maps a (q, r)-tensor to a tensor
of the same type — the type/degree is **preserved** (``∇_X`` has
degree 0). Special cases:

* ``∇_X f = X(f)`` (function; scalar → scalar),
* ``∇_X Y`` (vector → vector),
* ``∇_X ω`` (1-form → 1-form),
* Leibniz: ``∇_X(f·T) = X(f)·T + f·∇_X T``.

Implementation: :class:`CovariantOp` (``∇_X``) is a
:class:`~jacopy.algebra.derivation.Derivation` (degree 0), so the node
``∇_X T = Act(∇_X, T)`` keeps degree ``|T|``. :class:`Connection`
itself is a context (like a Bundle), not an Expr; it produces the
``∇_X`` operator and the ``∇_X T`` node. Type preservation on tensors
outside the form grading is tracked by
:func:`~jacopy.central.objects.tensor.signature_of`, which maps
``Act(∇_X, T)`` to the signature of ``T``.

**Only the degree/type algebra is encoded at this layer**
(``|∇_X T| = |T|``, signature preserved); NONE of the covariant
derivative's *computational* (rewrite) rules are active yet.
Specifically:

* ``∇_X f = X(f)`` (anchor / directional derivative) — the node is
  built but does not expand,
* Leibniz ``∇_X(f·T) = X(f)·T + f·∇_X T`` — since ``∇_X`` is a
  Derivation, :mod:`jacopy.algorithms.product_rule` CAN apply to it,
  but the purified :func:`~jacopy.proof.expansion.default_engine` does
  not yet *invoke* that rule (it contains only ``ActOverSumOp``),
* ``∇_X e_b = ω^a_b(X) e_a`` (connection 1-forms / Christoffel data)
  and the full action on a (q, r)-tensor
  ``(∇_X T)(…) = X(T(…)) − Σ T(…, ∇_X e_i, …)``.

These expansion rules (definitions) arrive in Phase 2 alongside d/L in
the engine layer; torsion/curvature/non-metricity and the total
covariant derivative ``∇T`` ((q, r) → (q, r+1)) come with the
metric-affine package (Phase 4).

**Central-code view.** On ``TM`` this is the ordinary affine
connection; for a general bundle ``E`` it is an ``E``-connection
(``∇_u: Γ(E) → Γ(E)``, with the anchor ``ρ_E`` supplying the function
action).
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr
from jacopy.central.objects.bundle import Bundle, TM


class CovariantOp(Derivation):
    """``∇_X`` — covariant-derivative operator in the direction ``X``
    (degree 0).

    Parameters
    ----------
    connection_name
        Name of the owning connection (display + identity).
    X
        Direction of differentiation (a vector field).
    bundle
        Bundle of the owning connection; part of the operator's
        identity so that same-named connections on different bundles
        yield distinct operators. Defaults to :data:`TM`.
    name
        Optional display name; defaults to ``"∇_X"``.
    """

    __slots__ = ("_connection_name", "_vector", "_bundle")

    def __init__(
        self,
        connection_name: str,
        X: Expr,
        *,
        bundle: Optional[Bundle] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(X, Expr):
            raise TypeError("CovariantOp requires an Expr direction vector")
        if bundle is not None and not isinstance(bundle, Bundle):
            raise TypeError("bundle must be a Bundle instance")
        display = (
            name
            if name is not None
            else f"{connection_name}_{X._repr_inner()}"
        )
        super().__init__(display, degree=0)
        self._connection_name = connection_name
        self._vector = X
        self._bundle = bundle if bundle is not None else TM

    @property
    def vector(self) -> Expr:
        return self._vector

    @property
    def connection_name(self) -> str:
        return self._connection_name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def rewritable_slots(self):
        """Slot protocol (Phase 3.C): rules fire on the direction —
        e.g. the bracket module structure inside ``∇_{[fX,Y]}``."""
        return (self._vector,)

    def with_slots(self, X: Expr) -> "CovariantOp":
        return CovariantOp(
            self._connection_name, X, bundle=self._bundle
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        """Slot-walking substitution: the direction lives in a slot,
        not in children (Phase 4.E.2 coframe recombination)."""
        if self == dummy:
            return target
        new_vector = self._vector.substitute_atom(dummy, target)
        if new_vector is self._vector:
            return self
        return CovariantOp(
            self._connection_name, new_vector, bundle=self._bundle
        )

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._connection_name,
            self._vector,
            self._bundle,
        )


class Connection:
    """``∇`` — an affine connection (item 8m).

    A context object (not an Expr). Produces the ``∇_X`` operator
    (:meth:`op`) and the covariant derivative ``∇_X T`` (call syntax).

    Parameters
    ----------
    name
        Display name; defaults to ``"∇"``.
    bundle
        Bundle the connection lives on; defaults to :data:`TM`.
    """

    __slots__ = ("_name", "_bundle")

    def __init__(self, name: str = "∇", *, bundle: Optional[Bundle] = None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Connection name must be a non-empty str")
        if bundle is not None and not isinstance(bundle, Bundle):
            raise TypeError("bundle must be a Bundle instance")
        self._name = name
        self._bundle = bundle if bundle is not None else TM

    @property
    def name(self) -> str:
        return self._name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    def op(self, X: Expr) -> CovariantOp:
        """Build the operator ``∇_X`` (direction ``X``)."""
        return CovariantOp(self._name, X, bundle=self._bundle)

    def __call__(self, X: Expr, T: Expr) -> Act:
        """``∇_X T`` — covariant derivative along ``X`` (degree ``|T|``)."""
        if not isinstance(X, Expr) or not isinstance(T, Expr):
            raise TypeError("∇(X, T) arguments must be Expr")
        return Act(self.op(X), T)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Connection)
            and self._name == other._name
            and self._bundle == other._bundle
        )

    def __hash__(self) -> int:
        return hash((self._name, self._bundle))

    def __repr__(self) -> str:
        return f"Connection({self._name!r}, bundle={self._bundle!r})"


def connection(name: str = "∇", *, bundle: Optional[Bundle] = None) -> Connection:
    """Create an affine connection ``∇``."""
    return Connection(name, bundle=bundle)


def covariant_derivative(conn: Connection, X: Expr, T: Expr) -> Act:
    """``∇_X T`` — readable call helper."""
    if not isinstance(conn, Connection):
        raise TypeError("covariant_derivative first argument must be a Connection")
    return conn(X, T)
