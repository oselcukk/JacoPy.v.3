"""Faz 8 step 2b — the read-only walker ``iter_nodes`` and the
traversal CONFORMANCE of every ``Expr`` subclass, checked by protocol
(not by class list): nodes with children round-trip through
``_rebuild``, nodes with slots round-trip through ``with_slots``, the
walker reaches children and slots alike, a leaf renamed through
``substitute_atom`` reaches slot-hidden atoms too, and atom metadata
(degree, bundle, signature), binder dummy / range and the alternating
flag survive every round-trip. The construction recipes are the
display coverage test's, so a new node class without a recipe fails
there and is exercised here."""

from __future__ import annotations

import pytest
from test_display_coverage import _INTERNAL, _all_expr_subclasses, _recipes

from jacopy.central.objects import Form, PVector, VectorField, kind_of
from jacopy.central.objects.frame import FrameIndex
from jacopy.core.expr import Atom, Expr, Integer, Symbol
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.traverse import contains, iter_atoms, iter_nodes, slots_of


def _twin(atom: Atom):
    """A fresh atom of the same class and metadata with a primed name,
    for the leaf-renaming round-trip; ``None`` for atoms that carry no
    name to rename (literals, wildcards, indices of other shapes)."""
    if isinstance(atom, FrameIndex):
        return FrameIndex(atom.name + "p")
    if isinstance(atom, VectorField) and type(atom) is VectorField:
        return VectorField(atom.name + "′", bundle=atom.bundle)
    if isinstance(atom, Form) and type(atom) is Form:
        return Form(atom.name + "′", degree=atom.degree, bundle=atom.bundle)
    if isinstance(atom, PVector) and type(atom) is PVector:
        return PVector(atom.name + "′", degree=atom.degree, bundle=atom.bundle)
    if type(atom) is Symbol:
        return Symbol(atom.name + "′")
    return None


def _metadata(expr: Expr):
    """Everything a round-trip must preserve besides equality."""
    meta = {"type": type(expr), "repr": repr(expr), "hash": hash(expr), "kind": kind_of(expr)}
    for attr in ("degree", "bundle", "signature", "wedge_degree", "alternating", "slot_kind", "name", "leibniz"):
        try:
            meta[attr] = getattr(expr, attr)
        except AttributeError:
            pass
    if isinstance(expr, IndexedSum):
        meta["dummy"], meta["range"] = expr.dummy, expr.range_
    return meta


@pytest.mark.parametrize("name", sorted(_recipes()))
def test_conformance_by_protocol(name):
    e = _recipes()[name]
    meta = _metadata(e)

    # --- the walker: self first, then every child and every slot ---
    nodes = list(iter_nodes(e))
    assert nodes[0] is e
    for c in e.children:
        assert any(n is c for n in nodes)
    slots = slots_of(e)
    for s in slots:
        assert any(n is s for n in nodes), (name, s)
        assert contains(e, s)
    without = list(iter_nodes(e, slots=False))
    assert without[0] is e and len(without) <= len(nodes)
    if e.is_atom:
        assert without == [e]  # slots are the only way into an atom
    assert all(a.is_atom for a in iter_atoms(e))

    # --- _rebuild round-trip (children) ---
    if not e.is_atom:
        r = e._rebuild(e.children)
        assert r == e and _metadata(r) == meta, name

    # --- with_slots round-trip (slots) ---
    if slots:
        r = e.with_slots(*slots)
        assert r == e and _metadata(r) == meta, name
        assert tuple(slots_of(r)) == tuple(slots)

    # --- a renamed leaf reaches everywhere, and back ---
    renamed = 0
    for leaf in list(iter_atoms(e)):
        twin = _twin(leaf)
        if twin is None or leaf == e:
            continue
        e2 = e.substitute_atom(leaf, twin)
        assert e2 != e, (name, leaf)
        assert contains(e2, twin) and not contains(e2, leaf), (name, leaf)
        assert type(e2) is type(e)
        assert e2.substitute_atom(twin, leaf) == e, (name, leaf)
        renamed += 1
    if slots and any(_twin(a) is not None for a in iter_atoms(e)):
        assert renamed > 0, name  # every slot-bearing node with a nameable leaf was exercised


def test_every_slot_bearing_class_is_covered():
    # the protocol test above is only as good as the recipe list: a
    # slot-bearing class without a recipe would never be walked
    slotted = {c.__name__ for c in _all_expr_subclasses() if "rewritable_slots" in vars(c)} - _INTERNAL
    assert slotted, "no slot-bearing classes found — the protocol probe is broken"
    assert slotted <= set(_recipes()), sorted(slotted - set(_recipes()))


def test_binder_scope_is_read_only_metadata():
    from jacopy.central.objects import vector_fields

    (X,) = vector_fields("X")
    i, j = Symbol("i"), Symbol("j")
    S = IndexedSum(i, (0, 1), X)
    # the bound dummy is metadata, not a node
    assert i not in list(iter_nodes(S)) and list(iter_atoms(S)) == [X]
    # substitution respects the binder: the dummy from outside is shadowed
    assert S.substitute_atom(i, j) == S
    # and α-renaming is a different operation
    assert S.with_dummy(j) == S and S.with_dummy(j).dummy == j


def test_walker_reaches_atoms_hidden_in_operator_slots():
    from jacopy.central.objects import forms, vector_fields
    from jacopy.central.objects.interior import Interior
    from jacopy.central.objects.multivector_interior import MultivectorInterior
    from jacopy.core.wedge import Wedge

    X, Y = vector_fields("X Y")
    (om,) = forms("ω", degree=1)
    op = MultivectorInterior(Wedge(X, Y))
    assert contains(op, X) and not contains(op, X, slots=False)
    assert contains(Interior(X), X) and not contains(Interior(X), Y)
    # the deferred "operator-atom index opacity": substitution now goes
    # through the slot protocol by default
    assert Interior(X).substitute_atom(X, Y) == Interior(Y)
    assert op.substitute_atom(X, Y) == MultivectorInterior(Wedge(Y, Y))
    assert contains(Wedge(om, Interior(X)), X)


def test_walker_is_pre_order_and_lazy():
    from jacopy.core.expr import Neg, Sum

    a, b = Symbol("a"), Symbol("b")
    e = Sum(Neg(a), b)
    assert [type(n).__name__ for n in iter_nodes(e)] == ["Sum", "Neg", "Symbol", "Symbol"]
    it = iter_nodes(e)
    assert next(it) is e  # a generator, not a list
    with pytest.raises(TypeError):
        list(iter_nodes("a"))
    assert list(iter_nodes(Integer(1))) == [Integer(1)]
    assert list(iter_atoms(MultiEval(Form("α", degree=2), a, b))) == [Form("α", degree=2), a, b]
