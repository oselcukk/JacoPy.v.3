"""PDF item 8w, "generalization to arbitrary numbers" (deferred-items
ledger K2, 2026-09-22): partial evaluation of k-linear maps — the
musical view with any number of fixed slots — as a node with
mechanical semantics: collapse on full evaluation, multilinearity in
the fixed slots, identification with the iterated interior product
for alternating forms, and the bilinear musical atoms as the j = 1
case. Type bookkeeping (degree / signature) is derived."""

import pytest

from jacopy.algebra.derivation import Act, degree_of
from jacopy.central.objects import (
    Flat,
    Metric,
    PVector,
    PartialEval,
    Sharp,
    Tensor,
    forms,
    functions,
    musical_view,
    partial_eval,
    vector_fields,
)
from jacopy.central.objects.interior import contract_all
from jacopy.central.objects.metric import InverseMetric
from jacopy.central.objects.tensor import signature_of
from jacopy.central.tangent.engine import tangent_engine
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure


@pytest.fixture()
def cast():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    X, Y, Z, W = vector_fields("X Y Z W")
    al, be = forms("α β", degree=1)
    (s3,) = forms("σ", degree=3)
    return reg, f, g, X, Y, Z, W, al, be, s3


def _prove(lhs, rhs, reg):
    return ExpandAndSimplify().prove(lhs, rhs, registry=reg, engine=tangent_engine(registry=reg))


# ---- the node ---------------------------------------------------------- #


def test_construction_and_bookkeeping(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    T = Tensor("T", upper=1, lower=3)
    v = partial_eval(T, al, None, X, None)  # T(α, ·, X, ·)
    assert isinstance(v, PartialEval)
    assert v.arity == 4 and v.n_open == 2
    assert v.fixed_positions == (0, 2) and v.open_positions == (1, 3)
    assert v.slot_kind == "mixed" and not v.alternating
    assert v._repr_inner() == "T(α, ·, X, ·)"
    assert v(Y, Z) == MultiEval(T, al, Y, X, Z, alternating=False, slot_kind="mixed")
    with pytest.raises(TypeError):
        v(Y)
    with pytest.raises(ValueError):
        partial_eval(T, al, be, X, Y)  # nothing open
    with pytest.raises(ValueError):
        partial_eval(T, None, None, None, None)  # nothing fixed
    with pytest.raises(ValueError):
        PartialEval(T, 4, {7: X})


def test_musical_view_defaults_from_the_head(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    gm = Metric("g")
    assert musical_view(gm, X).arity == 2 and musical_view(gm, X).slot_kind == "vector"
    pi = PVector("π", degree=2)
    v = musical_view(pi, al)
    assert v.alternating and v.slot_kind == "covector"
    w = musical_view(s3, X, Y)
    assert w.alternating and w.n_open == 1 and w.arity == 3
    (k,) = functions("k", registry=reg)
    with pytest.raises(ValueError):
        musical_view(k, X)  # no signature: arity needed
    assert musical_view(k, X, arity=3).arity == 3


def test_derived_types(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    # a 3-form with two slots fixed is a 1-form
    assert degree_of(musical_view(s3, X, Y)) == Degree.const(1)
    assert degree_of(musical_view(s3, X)) == Degree.const(2)
    # signatures: g(X,·) covariant 1; g⁻¹(α,·) contravariant 1; T(α,·,X,·) → (1,1)
    assert signature_of(musical_view(Metric("g"), X)) == (0, 1)
    assert signature_of(musical_view(InverseMetric("g⁻¹"), al)) == (1, 0)
    T = Tensor("T", upper=2, lower=2)
    assert signature_of(partial_eval(T, al, None, X, None)) == (1, 1)
    assert signature_of(partial_eval(T, None, None, X, Y)) == (2, 0)
    # a bivector with one covector fixed is a vector (multivector degree 1)
    assert musical_view(PVector("π", degree=2), al).wedge_degree == Degree.const(1)


def test_rebuild_and_equality_are_structural(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    gm = Metric("g")
    a, b = musical_view(gm, X), musical_view(gm, X)
    assert a == b and hash(a) == hash(b)
    assert a != musical_view(gm, Y)
    assert a != partial_eval(gm, None, X)  # a different slot fixed
    assert a._rebuild((gm, Y)) == musical_view(gm, Y)


# ---- engine semantics ------------------------------------------------- #


def test_collapse_on_full_evaluation_and_pairing(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    gm = Metric("g")
    # g(X,·)(Y) → g(X,Y);   ⟨g(X,·), Y⟩ → g(X,Y);   ⟨β, g⁻¹(α,·)⟩ → g⁻¹(α,β)
    assert _prove(Pairing(musical_view(gm, X), Y), gm(X, Y), reg).steps
    inv = InverseMetric("h")
    assert _prove(Pairing(be, musical_view(inv, al)), inv(al, be), reg).steps
    T = Tensor("T", upper=1, lower=3)
    v = partial_eval(T, al, None, X, None)
    assert _prove(MultiEval(v, Y, Z, alternating=False, slot_kind="mixed"), T(al, Y, X, Z), reg).steps


def test_multilinearity_in_the_fixed_slots(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    gm = Metric("g")
    lhs = Pairing(musical_view(gm, Sum(Product(f, X), Y)), Z)
    rhs = Sum(Product(f, gm(X, Z)), gm(Y, Z))
    assert _prove(lhs, rhs, reg).steps
    assert _prove(Pairing(musical_view(gm, Neg(X)), Z), Neg(gm(X, Z)), reg).steps
    assert _prove(Pairing(musical_view(gm, Integer(0)), Z), Integer(0), reg).steps


def test_alternating_partial_map_is_the_iterated_interior(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    # σ(X, Y, ·) = ι_Y ι_X σ  and  ⟨σ(X,Y,·), Z⟩ = σ(X,Y,Z)
    assert _prove(musical_view(s3, X, Y), contract_all(s3, X, Y), reg).steps
    assert _prove(
        Pairing(musical_view(s3, X, Y), Z),
        MultiEval(s3, X, Y, Z, alternating=True, slot_kind="vector"),
        reg,
    ).steps
    # non-leading fixed slots carry the permutation sign: σ(X, ·, Y) at Z
    # is σ(X, Z, Y) = −σ(X, Y, Z) (odd), while σ(·, X, Y) at Z is
    # σ(Z, X, Y) = +σ(X, Y, Z) (cyclic, even) — the engine refuses the
    # wrong sign in both cases.
    assert _prove(
        Pairing(partial_eval(s3, X, None, Y), Z),
        Neg(MultiEval(s3, X, Y, Z, alternating=True, slot_kind="vector")),
        reg,
    ).steps
    assert _prove(
        Pairing(partial_eval(s3, None, X, Y), Z),
        MultiEval(s3, X, Y, Z, alternating=True, slot_kind="vector"),
        reg,
    ).steps
    with pytest.raises(ProofFailure):
        _prove(
            Pairing(partial_eval(s3, None, X, Y), Z),
            Neg(MultiEval(s3, X, Y, Z, alternating=True, slot_kind="vector")),
            reg,
        )


def test_musical_atoms_are_the_one_slot_case(cast):
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    gm, inv = Metric("g"), InverseMetric("h")
    # g♭(X) paired with Y is g(X,Y); ⟨β, h♯(α)⟩ is h(α,β)
    assert _prove(Pairing(Act(Flat(gm), X), Y), gm(X, Y), reg).steps
    assert _prove(Pairing(be, Act(Sharp(inv), al)), inv(al, be), reg).steps
    # a 2-form's flat is alternating: ω♭(X) = ι_X ω, so ⟨ω♭(X), Y⟩ = ω(X,Y) = −ω(Y,X)
    (om,) = forms("ω", degree=2)
    assert _prove(
        Pairing(Act(Flat(om), X), Y),
        Neg(MultiEval(om, Y, X, alternating=True, slot_kind="vector")),
        reg,
    ).steps


def test_symmetric_head_does_not_get_an_alternating_sign(cast):
    # honesty: g(·, X) paired with Y is g(Y, X), NOT −g(X, Y); the
    # interior identification is reserved for alternating heads.
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    gm = Metric("g")
    assert _prove(Pairing(partial_eval(gm, None, X), Y), gm(Y, X), reg).steps
    with pytest.raises(ProofFailure):
        _prove(Pairing(partial_eval(gm, None, X), Y), Neg(gm(X, Y)), reg)


def test_three_slots_fixed_of_a_four_linear_map(cast):
    # "arbitrary numbers": T(α, β, X, ·) of a (2,2)-tensor is a 1-form;
    # pairing it closes the last slot.
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    T = Tensor("T", upper=2, lower=2)
    v = partial_eval(T, al, be, X, None)
    assert signature_of(v) == (0, 1)
    assert _prove(Pairing(v, Y), T(al, be, X, Y), reg).steps
    # fixing further slots composes: T(α, ·, ·, ·) then (β at 1, X at 2)
    w = musical_view(T, al).with_fixed({1: be, 2: X})
    assert w == v


# ---- 2026-09-23 audit pins (4DD888E_PROGRESS_REVIEW) --------------------- #


def test_one_open_vector_slot_is_a_one_form_for_any_head(cast):
    # F1: g(X,·) is a 1-form (signature (0,1)) — degree 1 even though g
    # is symmetric; scaling it simplifies; a form's partial map answers
    # the research layer's exterior-degree query.
    from jacopy.algorithms.simplify import simplify
    from jacopy.research.engine_assembly import _wedge_degree

    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    v = musical_view(Metric("g"), X)
    assert signature_of(v) == (0, 1) and degree_of(v) == Degree.const(1)
    simplify(Product(Integer(2), v), reg)
    assert _wedge_degree(musical_view(s3, X), reg) == 2
    # a symmetric map with TWO open slots is a tensor, not a form
    T = Tensor("S", upper=0, lower=3)
    with pytest.raises(AttributeError):
        musical_view(T, X).degree
    with pytest.raises(ValueError):
        degree_of(musical_view(T, X))


def test_known_arity_cannot_be_contradicted(cast):
    # F2: a bilinear metric is not a trilinear map; a 3-form has 3 slots.
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    with pytest.raises(ValueError):
        partial_eval(Metric("g"), X, None, None)
    with pytest.raises(ValueError):
        partial_eval(s3, X, None)
    with pytest.raises(ValueError):
        musical_view(Metric("g"), X, arity=3)
    with pytest.raises(ValueError):
        PartialEval(Metric("g"), 3, {0: X})


def test_nested_partial_maps_flatten_and_keep_their_type(cast):
    # F3: σ(X,·,·)(Y,·) IS σ(X,Y,·) — one node, alternating kept.
    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    nested = musical_view(musical_view(s3, X), Y)
    assert nested == musical_view(s3, X, Y)
    assert nested.alternating and signature_of(nested) == (0, 1)
    assert degree_of(nested) == Degree.const(1)
    assert _prove(
        Pairing(nested, Z), MultiEval(s3, X, Y, Z, alternating=True, slot_kind="vector"), reg
    ).steps
    # nesting must not change the flags or claim more open slots
    with pytest.raises(ValueError):
        PartialEval(musical_view(s3, X), 2, {0: Y}, alternating=False, slot_kind="vector")
    with pytest.raises(ValueError):
        PartialEval(musical_view(s3, X), 3, {0: Y}, alternating=True, slot_kind="vector")


def test_degree_is_derived_from_the_open_slots_for_any_head(cast):
    # 2026-09-23 recheck (remaining F1): the open map's type decides —
    # (0,1) is a 1-form and (1,0) a vector for ANY head (mixed tensors
    # included), and an alternating vector-slot map with n open slots
    # is an n-form even for an opaque head declared through the node.
    from jacopy.core.expr import Symbol
    from jacopy.research.engine_assembly import _wedge_degree

    reg, f, g, X, Y, Z, W, al, be, s3 = cast
    p = partial_eval(Tensor("T", upper=1, lower=2), al, X, None)
    assert signature_of(p) == (0, 1) and degree_of(p) == Degree.const(1)
    q = partial_eval(Tensor("S", upper=2, lower=1), None, al, X)
    assert signature_of(q) == (1, 0) and _wedge_degree(q, reg) == 1
    assert degree_of(musical_view(Tensor("C", upper=0, lower=2), X)) == Degree.const(1)
    a = PartialEval(Symbol("A"), 3, {0: X}, alternating=True, slot_kind="vector")
    assert signature_of(a) == (0, 2) and degree_of(a) == Degree.const(2)
    # still no false form for a symmetric multi-open map
    assert getattr(musical_view(Tensor("R", upper=0, lower=3), X), "degree", None) is None
