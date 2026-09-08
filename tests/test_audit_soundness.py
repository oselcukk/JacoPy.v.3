"""Soundness regressions from the 2026-09-07/08 independent audit.

Each test is a counterexample that USED to close as a false equality
(or crash) — see ``audit_jacopy_v3/REVIEW_TR.md`` findings 1-7 and
``COMPLIANCE_TR.md`` finding 1. They pin the reliability pass: the
same rules must keep REJECTING these neighbours of true identities.
"""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.derived import DerivedBracket
from jacopy.central.objects import (
    Bundle,
    forms,
    hodge,
    metric,
    tensors,
    vector_fields,
)
from jacopy.central.tangent import tangent_engine
from jacopy.central.tangent.cartan import L
from jacopy.core.expr import Integer, Symbol
from jacopy.core.hodge import hodge_star
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Graded, NonCommuting, Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.strategies import AgreementOnGenerators, ProofFailure
from jacopy.proof.verifier import prove_jacobi, show_equal


class _CommutatorBracket(GradedBracket):
    def __init__(self):
        super().__init__("commutator", degree=0)

    def expand(self, a, b, registry=None):
        return a * b - b * a


class _ZeroBracket(GradedBracket):
    def __init__(self):
        super().__init__("zero-bracket", degree=0)

    def expand(self, a, b, registry=None):
        return Integer(0)


def test_operator_composition_is_not_commutative():
    # Audit finding 1: X = ∂x, Y = x∂x gives (X∘Y − Y∘X)(x) = 1.
    X, Y = vector_fields("X Y")
    with pytest.raises(ProofFailure):
        show_equal(X * Y, Y * X, registry=PropertyRegistry())


def test_derived_jacobi_rejects_even_shifted_generator():
    # Audit finding 2: with |Q| = |base| = 0 the condition [Q,Q] = 0
    # is vacuous (matrix counterexample Q = A = E₁₁).
    reg = PropertyRegistry()
    A, B, C, Q = map(Symbol, ("A", "B", "C", "Q"))
    for symbol in (A, B, C, Q):
        reg.declare(symbol, Graded(0))
        reg.declare(symbol, NonCommuting())
    derived = DerivedBracket(_CommutatorBracket(), Q, degree_Q=0)
    with pytest.raises((ProofFailure, ValueError)):
        prove_jacobi(derived, A, B, C, registry=reg)


def test_derived_degree_counts_base_degree_twice():
    # Audit finding 2 (degree): |[[a,Q],b]| = |a|+|b|+|Q|+2|base|.
    base = _CommutatorBracket()
    derived = DerivedBracket(base, Symbol("Q"), degree_Q=1)
    assert derived.degree == derived.degree_Q + base.degree + base.degree


def test_indexed_sum_keeps_bracket_identity_in_key():
    # Audit finding 3: Σᵢ [A,B] must differ from Σᵢ 0-bracket(A,B).
    i, A, B = map(Symbol, ("i", "A", "B"))
    lhs = IndexedSum(i, (0, 1), _CommutatorBracket()(A, B))
    rhs = IndexedSum(i, (0, 1), _ZeroBracket()(A, B))
    assert lhs.body != rhs.body
    assert lhs != rhs


def test_agreement_on_generators_rejects_compositions():
    # Audit finding 4: D² is not a derivation; D²(x) = 0 on the
    # generator does not give D² = 0 (D²(x·x) = 2 for D(x) = 1).
    D, x = Derivation("D", degree=0), Symbol("x")
    reg = PropertyRegistry()
    reg.declare(x, Scalar())
    reg.declare(D, NonCommuting())

    class DX(Definition):
        name = "D(x)=1"
        anchor = Act

        def matches(self, expr):
            return expr == Act(D, x)

        def rewrite(self, expr):
            return Integer(1)

    class PolynomialAlgebra:
        generators = (x,)

    engine = ExpansionEngine([DX()])
    show_equal(Act(D * D, x * x), Integer(2), registry=reg, engine=engine)
    with pytest.raises((ProofFailure, TypeError, ValueError)):
        AgreementOnGenerators(PolynomialAlgebra()).prove(
            D * D, Integer(0), registry=reg, engine=engine
        )


def test_hodge_metric_is_part_of_identity():
    # Audit finding 5: ⋆_g dx = 1 but ⋆_{4g} dx = ½ — different g,
    # different operation.
    bundle = Bundle("M", dim=1)
    (omega,) = forms("omega", degree=1, bundle=bundle)
    g, h = metric("g", bundle=bundle), metric("h", bundle=bundle)
    assert g != h
    assert hodge(omega, g) != hodge(omega, h)


def test_hodge_survives_generic_tree_rebuild():
    # Audit finding 6: simplify/flatten must not crash on ⋆.
    (omega,) = forms("omega", degree=1)
    star = hodge_star(omega, 3)
    assert simplify(star) == star


def test_derived_anchor_path_uses_v3_modules():
    # Audit finding 7: the acting_on path imported dead
    # ``jacopy.calculus.*`` modules.
    alpha, beta = forms("alpha beta", degree=1)
    derived = DerivedBracket(
        _CommutatorBracket(), Symbol("Q"), acting_on=Derivation("rho")
    )
    assert derived.expand(alpha, beta) is not None


def test_nested_sums_distinguish_outer_and_inner_indices():
    # Re-audit finding 1 (regression pin): Σᵢ Σⱼ i (= 6 on the given
    # ranges) must not equal Σᵢ Σⱼ j (= 14); an embedded inner sum's
    # own _key must not restart the binder-depth counter.
    i, j = Symbol("i"), Symbol("j")
    lhs = IndexedSum(i, (1, 2), IndexedSum(j, (3, 4), i))
    rhs = IndexedSum(i, (1, 2), IndexedSum(j, (3, 4), j))
    assert lhs != rhs
    with pytest.raises(ProofFailure):
        show_equal(lhs, rhs, registry=PropertyRegistry())


def test_derived_cyclic_jacobi_needs_skewness_not_just_qq_zero():
    # Re-audit finding 2: a genuine graded-Lie base (supercommutator
    # on 3×3 matrices, weights (1,1,0)) with an odd square-zero Q
    # still fails the CYCLIC Jacobi — [Q,Q] = 0 licenses only the
    # Loday form; the strategy must demand certified skewness.
    from jacopy.core.expr import Atom
    from jacopy.core.symbolic_degree import Degree

    ZERO3 = tuple(tuple(0 for _ in range(3)) for _ in range(3))

    def mul(a, b):
        return tuple(
            tuple(
                sum(a[r][k] * b[k][c] for k in range(3))
                for c in range(3)
            )
            for r in range(3)
        )

    class Matrix(Atom):
        def __init__(self, entries, degree):
            self.entries = entries
            self.degree = Degree.const(degree)

        def _key(self):
            return self.entries, self.degree

        def _repr_inner(self):
            return f"M{self.entries}"

    def unit(r, c):
        weights = (1, 1, 0)
        return Matrix(
            tuple(
                tuple(
                    int(rr == r and cc == c) for cc in range(3)
                )
                for rr in range(3)
            ),
            weights[r] - weights[c],
        )

    class SuperCommutator(GradedBracket):
        def __init__(self):
            super().__init__("supercommutator", degree=0)

        def expand(self, a, b, registry=None):
            if a == Integer(0) or b == Integer(0):
                return Integer(0)
            sign = (-1) ** (a.degree * b.degree).parity()
            ab, ba = mul(a.entries, b.entries), mul(b.entries, a.entries)
            entries = tuple(
                tuple(
                    ab[r][c] - sign * ba[r][c] for c in range(3)
                )
                for r in range(3)
            )
            if entries == ZERO3:
                return Integer(0)
            return Matrix(entries, (a.degree + b.degree).as_int())

    base = SuperCommutator()
    q, a, b, c = unit(0, 2), unit(0, 0), unit(1, 0), unit(2, 1)
    derived = DerivedBracket(base, q, degree_Q=1)
    assert base.expand(q, q) == Integer(0)
    with pytest.raises((ProofFailure, ValueError)):
        prove_jacobi(derived, a, b, c, registry=PropertyRegistry())


def test_graded_obstructions_use_bracket_degree_shifted_signs():
    # Second re-audit: for a degree-k bracket the Koszul signs in the
    # skew/Jacobi obstructions must use the k-SHIFTED degrees. With
    # unshifted signs, six skew checks passed while the certified
    # cyclic expression evaluated to 2·E21 ≠ 0 (weights (1,1,0),
    # Q = E13, A = E21, B = C = E31).
    class DegreeOneBracket(GradedBracket):
        def __init__(self):
            super().__init__("deg-one", degree=1)

        def expand(self, a, b, registry=None):
            return Integer(0)

    from jacopy.core.properties import Graded

    reg = PropertyRegistry()
    a, b = Symbol("a"), Symbol("b")
    reg.declare(a, Graded(0))
    reg.declare(b, Graded(1))
    br = DegreeOneBracket()
    # |a|+k = 1, |b|+k = 2 → parity 0 → PLUS sign; the unshifted
    # convention (|a||b| = 0 → plus as well here) differs on the
    # (a, a) diagonal: (|a|+1)² parity 1 → MINUS.
    obs = br.graded_antisymmetry_obstruction(a, a, reg)
    from jacopy.brackets.base import BracketApply
    from jacopy.core.expr import Neg as _Neg, Sum as _Sum

    assert obs == _Sum(
        BracketApply(br, a, a), _Neg(BracketApply(br, a, a))
    )


def test_agreement_on_generators_rejects_leibniz_false_operators():
    # Re-audit finding 3: ι_{X∧Y} subclasses Derivation for slot
    # plumbing but carries leibniz=False (it is a composition);
    # vanishing on the 1-form generators must not certify it zero.
    from jacopy.central.objects.multivector_interior import (
        MultivectorInterior,
    )
    from jacopy.core.wedge import Wedge
    from jacopy.proof.strategies import _is_derivation_expr

    X, Y = vector_fields("X Y")
    op = MultivectorInterior(Wedge(X, Y))
    assert op.leibniz is False
    assert not _is_derivation_expr(op, PropertyRegistry())

    class ExteriorAlgebra:
        alpha, beta = forms("alpha beta", degree=1)
        generators = (alpha, beta)

    with pytest.raises((ProofFailure, TypeError, ValueError)):
        AgreementOnGenerators(ExteriorAlgebra()).prove(
            op, Integer(0), registry=PropertyRegistry()
        )


def test_lie_derivative_respects_non_alternating_tensors():
    # Compliance finding 1: (L_X T)(Y,Y) ≠ 0 for a general
    # (non-alternating) (0,2) tensor — T = x dx⊗dx, X = Y = ∂x
    # gives 1. The intrinsic rules must inherit the MultiEval's
    # alternating/slot_kind flags, never overwrite them.
    X, Y = vector_fields("X Y")
    (T,) = tensors("T", upper=0, lower=2)
    reg = PropertyRegistry()
    lhs = MultiEval(L(X, T), Y, Y, alternating=False, slot_kind="mixed")
    with pytest.raises(ProofFailure):
        show_equal(
            lhs,
            Integer(0),
            registry=reg,
            engine=tangent_engine(registry=reg),
        )
