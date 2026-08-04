"""Engine strengthening interim phase (E.1-E.5): rule indexing,
symmetry-aware simplify, TheoremBook + citation, portfolio, search."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algorithms.simplify import simplify
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import (
    Definition,
    ExpandAndSimplify,
    ExpansionEngine,
    PortfolioStrategy,
    ProofFailure,
    SearchStrategy,
    Theorem,
    TheoremBook,
    TheoremDefinition,
    cite,
    prove,
)
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.tangent import lie_bracket, tangent_engine


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    X, Y, Z = vector_fields("X Y Z")
    f, g = functions("f g", registry=reg)
    return reg, X, Y, Z, f, g


# --------------------------------------------------------------------- #
# E.1 — anchored rule dispatch                                          #
# --------------------------------------------------------------------- #


class _SpyRule(Definition):
    """Counts how often the engine consults it; matches nothing."""

    name = "spy"
    anchor = Pairing

    def __init__(self):
        self.consulted = 0

    def matches(self, expr):
        self.consulted += 1
        return False

    def rewrite(self, expr):  # pragma: no cover
        raise AssertionError("never fires")


class TestRuleIndexing:
    def test_anchored_rule_not_consulted_off_anchor(self, setup):
        """A Pairing-anchored rule is never consulted on an Act tree."""
        reg, X, _, _, f, _ = setup
        spy = _SpyRule()
        eng = ExpansionEngine([spy])
        eng.expand(Act(X, f))  # tree contains Act + atoms, no Pairing
        assert spy.consulted == 0

    def test_anchored_rule_consulted_on_anchor(self, setup):
        _, X, *_ = setup
        (w,) = forms("ω", degree=1)
        spy = _SpyRule()
        eng = ExpansionEngine([spy])
        eng.expand(Pairing(w, X))
        assert spy.consulted >= 1

    def test_behavior_unchanged(self, setup):
        """Indexed dispatch still closes the 2.A proof. (The exact step
        count grew with the 2.F canonical-form rules — orientation
        normalization adds steps — so only closure is pinned.)"""
        from jacopy.central.tangent import prove_antisymmetry

        reg, X, Y, _, f, _ = setup
        chain = prove_antisymmetry(X, Y, f, registry=reg)
        assert chain.steps and chain.steps[-1].after == Integer(0)


# --------------------------------------------------------------------- #
# E.2 — symmetry-aware simplify + factoring                             #
# --------------------------------------------------------------------- #


class TestSimplifyUpgrades:
    def test_alternating_orientation_cancels(self, setup):
        _, X, Y, *_ = setup
        (w,) = forms("ω", degree=2)
        e = Sum(MultiEval(w, Y, X), MultiEval(w, X, Y))
        assert simplify(e) == Integer(0)

    def test_alternating_repeated_arg_zero(self, setup):
        _, X, *_ = setup
        (w,) = forms("ω", degree=2)
        assert simplify(MultiEval(w, X, X)) == Integer(0)

    def test_symmetric_untouched(self, setup):
        """alternating=False declares no symmetry — args stay put."""
        _, X, Y, *_ = setup
        (w,) = forms("g", degree=2)
        e = MultiEval(w, Y, X, alternating=False)
        assert simplify(e) == e

    def test_factor_common_lead(self):
        f, A, B = Symbol("f"), Symbol("A"), Symbol("B")
        out = simplify(Sum(Product(f, A), Product(f, B)))
        assert out == Product(f, Sum(A, B))

    def test_factor_does_not_break_cancellation(self):
        f, A = Symbol("f"), Symbol("A")
        assert simplify(Sum(Product(f, A), Neg(Product(f, A)))) == Integer(0)


# --------------------------------------------------------------------- #
# E.3 — TheoremBook + citation                                          #
# --------------------------------------------------------------------- #


def _antisym_theorem(reg, X, Y, f):
    from jacopy.central.tangent import prove_antisymmetry

    chain = prove_antisymmetry(X, Y, f, registry=reg)
    return Theorem(
        name="lie_antisymmetry",
        statement="[X, Y](f) = -[Y, X](f)",
        lhs=Act(lie_bracket(X, Y), f),
        rhs=Neg(Act(lie_bracket(Y, X), f)),
        proof=chain,
        generality="generic-function",
    )


class TestTheoremBook:
    def test_add_get(self, setup):
        reg, X, Y, _, f, _ = setup
        book = TheoremBook()
        book.add(_antisym_theorem(reg, X, Y, f))
        assert "lie_antisymmetry" in book
        assert book.get("lie_antisymmetry").generality == "generic-function"

    def test_duplicate_raises(self, setup):
        reg, X, Y, _, f, _ = setup
        book = TheoremBook()
        thm = _antisym_theorem(reg, X, Y, f)
        book.add(thm)
        with pytest.raises(KeyError):
            book.add(thm)

    def test_bad_generality_rejected(self, setup):
        reg, X, Y, _, f, _ = setup
        thm = _antisym_theorem(reg, X, Y, f)
        with pytest.raises(ValueError):
            Theorem(
                name="x",
                statement="s",
                lhs=thm.lhs,
                rhs=thm.rhs,
                proof=thm.proof,
                generality="nonsense",
            )

    def test_cited_theorem_fires_with_theorem_tag(self, setup):
        reg, X, Y, _, f, _ = setup
        book = TheoremBook()
        book.add(_antisym_theorem(reg, X, Y, f))
        eng = ExpansionEngine([])
        cite(eng, book, "lie_antisymmetry")
        expr = Act(lie_bracket(X, Y), f)
        out, steps = eng.expand(expr)
        assert out == Neg(Act(lie_bracket(Y, X), f))
        assert steps[0].provenance_tag == "theorem"

    def test_foundational_mode_inlines_subproof(self, setup):
        reg, X, Y, _, f, _ = setup
        book = TheoremBook()
        book.add(_antisym_theorem(reg, X, Y, f))
        eng = ExpansionEngine([], mode="foundational")
        cite(eng, book, "lie_antisymmetry")
        _, steps = eng.expand(Act(lie_bracket(X, Y), f))
        assert steps[0].children  # stored proof inlined under the step

    def test_reverse_citation(self, setup):
        reg, X, Y, _, f, _ = setup
        book = TheoremBook()
        book.add(_antisym_theorem(reg, X, Y, f))
        eng = ExpansionEngine([])
        cite(eng, book, "lie_antisymmetry", reverse=True)
        rhs = Neg(Act(lie_bracket(Y, X), f))
        out, steps = eng.expand(rhs)
        assert out == Act(lie_bracket(X, Y), f)
        assert steps

    def test_not_in_definitional_engines(self, setup):
        """Policy guard: the only theorem-classified rules in the
        definitional engine are the canonical-form normalizations
        (bracket orientation) — cited theorems never appear here."""
        from jacopy.central.tangent.lie_bracket import (
            BracketOrientationDefinition,
        )

        reg, *_ = setup
        theorem_rules = [
            d for d in tangent_engine(registry=reg).definitions if d.is_theorem
        ]
        assert all(
            isinstance(d, BracketOrientationDefinition) for d in theorem_rules
        )


# --------------------------------------------------------------------- #
# E.4 — portfolio                                                        #
# --------------------------------------------------------------------- #


class _AlwaysFails(ExpandAndSimplify):
    name = "AlwaysFails"

    def prove(self, lhs, rhs, *, registry=None, engine=None):
        raise ProofFailure("nope")


class TestPortfolio:
    def test_falls_through_to_success(self, setup):
        reg, X, Y, _, f, _ = setup
        pf = PortfolioStrategy([_AlwaysFails(), ExpandAndSimplify()])
        lhs = Act(lie_bracket(X, Y), f)
        rhs = Neg(Act(lie_bracket(Y, X), f))
        chain = pf.prove(lhs, rhs, registry=reg, engine=tangent_engine(registry=reg))
        assert chain.steps

    def test_all_fail_combined_message(self, setup):
        reg, X, Y, _, f, _ = setup
        pf = PortfolioStrategy([_AlwaysFails(), _AlwaysFails()])
        with pytest.raises(ProofFailure, match="all portfolio strategies"):
            pf.prove(Symbol("a"), Symbol("b"), registry=reg)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            PortfolioStrategy([])


# --------------------------------------------------------------------- #
# E.5 — search + prefer API                                              #
# --------------------------------------------------------------------- #


class TestSearch:
    def test_prefer_shortest_matches_deterministic(self, setup):
        reg, X, Y, _, f, _ = setup
        eng = tangent_engine(registry=reg)
        lhs = Act(lie_bracket(X, Y), f)
        rhs = Neg(Act(lie_bracket(Y, X), f))
        chain = prove(lhs, rhs, prefer="shortest", registry=reg, engine=eng)
        assert len(chain.steps) == 3

    def test_prefer_readable_closes(self, setup):
        reg, X, Y, _, f, _ = setup
        eng = tangent_engine(registry=reg)
        lhs = Act(lie_bracket(X, Y), f)
        rhs = Neg(Act(lie_bracket(Y, X), f))
        chain = prove(lhs, rhs, prefer="readable", registry=reg, engine=eng)
        assert chain.steps

    def test_prefer_none_is_deterministic_route(self, setup):
        reg, X, Y, _, f, _ = setup
        eng = tangent_engine(registry=reg)
        lhs = Act(lie_bracket(X, Y), f)
        rhs = Neg(Act(lie_bracket(Y, X), f))
        assert prove(lhs, rhs, registry=reg, engine=eng).steps

    def test_invalid_prefer_rejected(self):
        with pytest.raises(ValueError):
            SearchStrategy(prefer="fastest")

    def test_false_identity_fails(self, setup):
        reg, X, Y, _, f, _ = setup
        eng = tangent_engine(registry=reg)
        lhs = Act(lie_bracket(X, Y), f)
        rhs = Act(lie_bracket(Y, X), f)  # missing minus
        with pytest.raises(ProofFailure):
            prove(lhs, rhs, prefer="shortest", registry=reg, engine=eng,
                  max_depth=6, max_states=400)

    def test_beam_fallback_closes_on_budget_exhaustion(self, setup):
        """Tiny exhaustive budget forces the beam; proof still found."""
        reg, X, Y, Z, f, _ = setup
        eng = tangent_engine(registry=reg)
        jac = Sum(
            Act(lie_bracket(X, lie_bracket(Y, Z)), f),
            Act(lie_bracket(Y, lie_bracket(Z, X)), f),
            Act(lie_bracket(Z, lie_bracket(X, Y)), f),
        )
        chain = SearchStrategy(
            prefer="shortest", max_depth=14, max_states=300, beam_width=16
        ).prove(jac, Integer(0), registry=reg, engine=eng)
        assert chain.steps
        assert chain.steps[-1].after == Integer(0)

    def test_search_chain_replays_to_zero(self, setup):
        """The found chain is an honest witness: steps connect."""
        reg, X, Y, _, f, _ = setup
        eng = tangent_engine(registry=reg)
        lhs = Act(lie_bracket(X, Y), f)
        rhs = Neg(Act(lie_bracket(Y, X), f))
        chain = prove(lhs, rhs, prefer="shortest", registry=reg, engine=eng)
        for prev, nxt in zip(chain.steps, chain.steps[1:]):
            assert prev.after == nxt.before
        assert chain.steps[-1].after == Integer(0)
