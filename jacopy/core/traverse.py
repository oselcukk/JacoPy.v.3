"""
The read-only walker (Faz 8 step 2b).

An expression tree has two kinds of edges:

* ``children`` — the structural edges every node exposes
  (:attr:`~jacopy.core.expr.Expr.children`, rebuilt with
  :meth:`~jacopy.core.expr.Expr._rebuild`);
* ``rewritable_slots`` — the expressions an operator-like ATOM carries
  in private slots (the vector of ``ι_X``, the multivector of ``ι_P``,
  the sections of an algebroid bracket, …), rebuilt with
  ``with_slots(*subs)``. An atom has no children, so without the slot
  edges a walk stops at the operator and never sees ``X`` inside
  ``ι_X`` — the "operator-atom index opacity" every targeted pre-pass
  used to work around.

:func:`iter_nodes` walks both (slots optional), pre-order, lazily, and
never rebuilds anything. It yields nodes only: a binder's bound dummy
(:class:`~jacopy.core.indexed_sum.IndexedSum`) is metadata, not a
node, exactly as in :meth:`~jacopy.core.expr.Expr.walk`.

Substitution is a SEPARATE operation and stays one:
:meth:`~jacopy.core.expr.Expr.substitute_atom` is capture-avoiding
(a binder shadows its dummy) and rebuilds; the walker only reads. The
conformance test (``tests/test_traverse.py``) checks every node class
by protocol: ``_rebuild(children)`` and ``with_slots(*slots)`` return
an equal node with the same metadata, and a renamed leaf is reached
through children and slots alike.
"""

from __future__ import annotations

from typing import Iterator, Tuple

from jacopy.core.expr import Expr


def slots_of(expr: Expr) -> Tuple[Expr, ...]:
    """The ``rewritable_slots`` of ``expr`` as a tuple (``()`` when the
    node exposes none)."""
    slots = getattr(expr, "rewritable_slots", None)
    return tuple(slots) if slots else ()


def iter_nodes(expr: Expr, *, slots: bool = True) -> Iterator[Expr]:
    """Pre-order, lazy, read-only traversal: ``expr`` itself, then
    every descendant through ``children`` and — with ``slots=True``
    (the default) — through the ``rewritable_slots`` of atoms."""
    if not isinstance(expr, Expr):
        raise TypeError("iter_nodes expects an Expr")
    stack = [expr]
    while stack:
        node = stack.pop()
        yield node
        edges = list(node.children)
        if slots and not edges:
            edges = list(slots_of(node))
        stack.extend(reversed(edges))


def iter_atoms(expr: Expr, *, slots: bool = True) -> Iterator[Expr]:
    """The leaves of :func:`iter_nodes`: nodes with no children and
    (with ``slots=True``) no slots."""
    for node in iter_nodes(expr, slots=slots):
        if node.is_atom and not (slots and slots_of(node)):
            yield node


def contains(expr: Expr, target: Expr, *, slots: bool = True) -> bool:
    """True when ``target`` occurs in ``expr`` (structural equality),
    through children and — by default — slots."""
    return any(node == target for node in iter_nodes(expr, slots=slots))


__all__ = ["iter_nodes", "iter_atoms", "contains", "slots_of"]
