"""Faz 8 step 2a — the single exterior-degree contract
``exterior_degree(expr, registry) -> Degree | Unknown`` built on
``kind_of``: the acceptance table of the roadmap, the one shared
``Unknown`` object, symbolic degrees kept apart from unknown ones, and
the three former copies (``_wedge_degree``, ``_multivector_degree``,
the wedge normalizer's parity test) delegating to it."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose
from jacopy.algebra.grading import (
    Unknown,
    UnknownDegree,
    as_concrete,
    concrete_exterior_degree,
    exterior_degree,
    exterior_parity,
    is_unknown,
)
from jacopy.central.objects import (
    Metric,
    PVector,
    Tensor,
    forms,
    functions,
    kind_of,
    musical_view,
    vector_fields,
)
from jacopy.central.objects.interior import Interior
from jacopy.central.objects.multivector_interior import MultivectorInterior
from jacopy.central.tangent.cartan import L
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge


@pytest.fixture()
def cast():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    X, Y, Z = vector_fields("X Y Z")
    om, et = forms("ω η", degree=1)
    (B,) = forms("B", degree=2)
    (om_p,) = forms("ω_p", degree=Degree.var("p"))
    (om_q,) = forms("ω_q", degree=Degree.var("q"))
    return reg, f, g, X, Y, Z, om, et, B, om_p, om_q


# ------------------------------------------------------------------ #
# the acceptance table                                               #
# ------------------------------------------------------------------ #


def test_acceptance_table(cast):
    reg, f, g, X, Y, Z, om, et, B, om_p, om_q = cast
    p, q = Degree.var("p"), Degree.var("q")
    Pi = PVector("Π", degree=2)
    Pi_r = PVector("Π_r", degree=Degree.var("r"))
    table = [
        (Act(X, f), 0),                                   # X(f): 0
        (d(f), 1),                                        # d(f): 1
        (d(om_p), p + 1),                                 # d ω_p: p+1
        (Act(Interior(X), B), 1),                         # ι_X ω_2: 1
        (Act(Interior(X), om), 0),                        # ι_X ω_1: a function
        (Act(Interior(X), om_p), p - 1),                  # ι_X ω_p: p−1 (symbolic)
        (Act(MultivectorInterior(Pi), om_q), q - 2),      # ι_P ω_q: q−p
        (Act(MultivectorInterior(Pi_r, registry=reg), om_q), q - Degree.var("r")),
        (L(X, B), 2),                                     # ℒ preserves
        (om_p, p),                                        # symbolic p kept
        (Sum(X, Y), 1),                                   # a vector sum is odd
        (Wedge(Z, Wedge(Sum(X, Y), Z)), 3),               # nested wedge
        (IndexedSum(Symbol("i"), (0, 1), X), 1),          # Σ_i X grades like its body
        (Product(f, X), 1),                               # f·X: 1
        (Product(f, om_p), p),                            # f·ω_p: p
        (Product(f, g), 0),                               # f·g: a function
        (Neg(B), 2),
        (Wedge(om, et), 2),
        (Wedge(X, Y), 2),                                 # a bivector
        (Wedge(f, X), 1),                                 # a scalar factor in a wedge
        (Integer(3), 0),
        (musical_view(Metric("g"), X), 1),                # g(X,·) is a 1-form
        (lie_bracket(X, Y), 1),
    ]
    for expr, want in table:
        got = exterior_degree(expr, reg)
        assert got == want, (expr, got, want)
        assert kind_of(expr, reg).degree == got
    # and the same table through the concrete view
    assert concrete_exterior_degree(Product(f, X), reg) == 1
    assert concrete_exterior_degree(om_p, reg) is None      # symbolic, not unknown
    assert exterior_parity(Wedge(Z, Wedge(Sum(X, Y), Z)), reg) == 1


def test_unknowns(cast):
    reg, f, g, X, Y, Z, om, et, B, om_p, om_q = cast
    unknown = [
        compose(X, Y),                 # X∘Y: Unknown (NOT 2)
        Product(X, Y),
        Act(X, om),                    # a vector's Act on a form is not a Lie derivative
        Act(Derivation("δ", 1), om),   # an undefined action
        Symbol("h"),                   # undeclared without a registry
        Product(Symbol("h"), X),
        Tensor("T", upper=1, lower=2), # not an exterior-algebra element
        Derivation("δ", 1),            # an operator
        Sum(om, B),                    # a known mismatch is not graded either
    ]
    for expr in unknown:
        got = exterior_degree(expr, reg)
        assert got is Unknown, (expr, got)
        assert is_unknown(got)
        assert as_concrete(got) is None
        assert concrete_exterior_degree(expr, reg) is None
        assert exterior_parity(expr, reg) is None
    # the symbol IS graded once a registry says so
    assert exterior_degree(Symbol("h"), reg) is Unknown
    (h,) = functions("h", registry=reg)
    assert exterior_degree(h, reg) == 0
    assert exterior_degree(Product(h, X), reg) == 1


def test_one_unknown_object_shared_with_kind_of(cast):
    reg, f, g, X, Y, Z, om, et, B, om_p, om_q = cast
    assert isinstance(Unknown, UnknownDegree) and UnknownDegree() is Unknown
    assert kind_of(compose(X, Y), reg).degree is Unknown
    assert kind_of(Sum(om, B)).kind == "mismatch" and kind_of(Sum(om, B)).degree is Unknown
    # symbolic is not unknown: two symbolic forms of different declared
    # degree are not KNOWN to mismatch — the sum is a form of unknown degree
    s = kind_of(Sum(om_p, om_q))
    assert s.kind == "form" and s.degree is Unknown
    assert kind_of(Sum(om_p, om_p)).degree == Degree.var("p")
    assert kind_of(Sum(om, B)).kind == "mismatch"                  # concrete, different
    assert kind_of(Sum(om_p, Neg(om_p))).degree == Degree.var("p")
    # Unknown never compares equal to a degree, and never to None
    assert Unknown != Degree.const(0) and Unknown != 0 and Unknown is not None
    assert repr(Unknown) == "Unknown"
    assert as_concrete(Degree.const(2)) == 2 and as_concrete(Degree.var("p")) is None


def test_kind_keeps_the_symbolic_degree(cast):
    reg, f, g, X, Y, Z, om, et, B, om_p, om_q = cast
    k = kind_of(om_p)
    assert k.kind == "form" and k.degree == Degree.var("p") and k.concrete_degree is None
    assert kind_of(B).concrete_degree == 2
    assert kind_of(f, reg).degree == 0 and kind_of(X).degree == 1
    assert kind_of(Metric("g")).degree is Unknown


# ------------------------------------------------------------------ #
# the three copies delegate                                          #
# ------------------------------------------------------------------ #


def test_former_copies_agree_with_the_contract(cast):
    from jacopy.algorithms.normalize_alternating import (
        _certainly_scalar,
        _odd_wedge_degree,
    )
    from jacopy.central.objects.multivector_interior import _multivector_degree
    from jacopy.research.engine_assembly import _wedge_degree

    reg, f, g, X, Y, Z, om, et, B, om_p, om_q = cast
    samples = [
        X, Sum(X, Y), Wedge(X, Y), Wedge(Z, Wedge(Sum(X, Y), Z)),
        IndexedSum(Symbol("i"), (0, 1), X), Product(f, X), Product(f, g),
        om, B, om_p, d(f), Act(Interior(X), B), lie_bracket(X, Y),
        compose(X, Y), Act(X, om), Symbol("h"), Integer(2), Neg(X),
        PVector("Π", degree=2), musical_view(Metric("g"), X),
    ]
    for e in samples:
        want = exterior_degree(e, reg)
        assert _wedge_degree(e, reg) == as_concrete(want), e
        assert _odd_wedge_degree(e, reg) == (exterior_parity(e, reg) == 1), e
        assert _certainly_scalar(e, reg) == (want == Degree.const(0)), e
        if is_unknown(want):
            with pytest.raises(ValueError):
                _multivector_degree(e, reg)
        else:
            assert _multivector_degree(e, reg) == want


def test_multivector_interior_needs_a_registry_for_a_symbol_coefficient(cast):
    # the old helper GUESSED that an ungradeable Product factor is a
    # scalar coefficient; the contract asks instead
    reg, f, g, X, Y, Z, om, et, B, om_p, om_q = cast
    Pi = PVector("Π", degree=2)
    with pytest.raises(ValueError, match="registry"):
        MultivectorInterior(Product(f, Pi))
    op = MultivectorInterior(Product(f, Pi), registry=reg)
    assert op.degree == Degree.const(-2)
    assert op.with_slots(Pi).degree == Degree.const(-2)
    assert MultivectorInterior(Product(Integer(3), Pi)).degree == Degree.const(-2)


def test_wedge_normalizer_sorts_vector_sums_and_indexed_sums(cast):
    # the parity test now grades through the contract, so factors that
    # carried no attribute (a vector sum, an indexed sum) are sorted
    from jacopy.algorithms.normalize_alternating import normalize_alternating

    reg, f, g, X, Y, Z, om, et, B, om_p, om_q = cast
    S = Sum(X, Y)
    assert normalize_alternating(Wedge(Z, S), reg) == Neg(Wedge(S, Z))
    assert normalize_alternating(Wedge(S, S), reg) == Integer(0)
    # never sorts a symbolic-parity factor: soundness over completeness
    assert normalize_alternating(Wedge(om_p, om), reg) == Wedge(om_p, om)
