"""Algebroid case — Phase 3.E.2: the E-metric layer.

The metric node g(u,v) with its definitional structure (symmetry,
C∞-bilinearity, inverse-metric evaluation) and the two declared metric
axioms C1 (metric invariance) / C2 (symmetric part), with the new
hierarchy levels [B Def 4.2-4.4]."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Rational, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle, functions
from jacopy.central.algebroid import (
    EMetric,
    MetricSharp,
    algebroid,
    algebroid_engine,
    tangent_algebroid,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("metric",))
    u, v, w = E.sections("u v w")
    return reg, f, E, u, v, w


# --------------------------------------------------------------------- #
# Levels and bookkeeping                                                 #
# --------------------------------------------------------------------- #


class TestMetricLevels:
    def test_level_contents(self, setup):
        assert algebroid("E", declare=("almost-courant",)).declarations == frozenset(
            {"right-leibniz", "symmetric-part"}
        )
        assert algebroid("E", declare=("metric",)).declarations == frozenset(
            {"symmetric-part", "metric-invariance"}
        )
        assert algebroid("E", declare=("pre-courant",)).declarations == frozenset(
            {"symmetric-part", "metric-invariance", "anchor-morphism"}
        )
        assert algebroid("E", declare=("courant",)).declarations == frozenset(
            {"symmetric-part", "metric-invariance", "jacobi"}
        )

    def test_derivable_axioms_absent(self, setup):
        """Honesty of the levels: metric/pre-courant/courant do NOT
        hand out right-Leibniz ([B 4.11] derives it); courant does NOT
        hand out anchor-morphism (4.11 + the 3.D theorem derive it)."""
        for level in ("metric", "pre-courant", "courant"):
            assert not algebroid("E", declare=(level,)).declares(
                "right-leibniz"
            )
        assert not algebroid("E", declare=("courant",)).declares(
            "anchor-morphism"
        )

    def test_tangent_has_no_metric(self, setup):
        TMalg = tangent_algebroid()
        X, Y = TMalg.sections("X Y")
        with pytest.raises(ValueError):
            TMalg.metric(X, Y)
        with pytest.raises(ValueError):
            TMalg.sharp(X)


# --------------------------------------------------------------------- #
# Definitional structure of g                                            #
# --------------------------------------------------------------------- #


class TestMetricDefinitional:
    def test_symmetry_canonical_sort(self, setup):
        reg, _, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(E.metric(v, u))
        assert out == E.metric(u, v)
        assert steps
        out, steps = eng.expand(E.metric(u, v))
        assert out == E.metric(u, v) and not steps

    def test_bilinearity(self, setup):
        reg, f, E, u, v, w = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.metric(Sum(u, Product(f, w)), v))
        assert out == Sum(
            E.metric(u, v), Product(f, E.metric(v, w))
        )
        out, _ = eng.expand(E.metric(u, Neg(v)))
        assert out == Neg(E.metric(u, v))
        out, _ = eng.expand(E.metric(u, Integer(0)))
        assert out == Integer(0)

    def test_metric_is_scalar(self, setup):
        """g(u,v) must count as a C∞ function (so it can be pulled
        out of C∞-linear slots)."""
        from jacopy.central.calculus.scalars import is_scalar_function

        reg, _, E, u, v, _ = setup
        assert is_scalar_function(E.metric(u, v), reg)

    def test_sharp_evaluation_both_slots(self, setup):
        reg, f, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.metric(E.sharp(E.D(f)), v))
        assert out == Act(E.anchor(v), f)
        out, _ = eng.expand(E.metric(u, E.sharp(E.D(f))))
        assert out == Act(E.anchor(u), f)

    def test_sharp_linearity(self, setup):
        reg, f, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        Df, sharp = E.D(f), E.sharp(E.D(f))
        out, _ = eng.expand(MetricSharp("E", Product(f, Df)))
        assert out == Product(f, sharp)
        out, _ = eng.expand(MetricSharp("E", Sum(Df, Neg(Df))))
        assert out == Sum(sharp, Neg(sharp))
        out, _ = eng.expand(MetricSharp("E", Integer(0)))
        assert out == Integer(0)


# --------------------------------------------------------------------- #
# C1 — metric invariance (declared)                                      #
# --------------------------------------------------------------------- #


class TestMetricInvariance:
    def test_rule_fires(self, setup):
        reg, _, E, u, v, w = setup
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(Act(E.anchor(u), E.metric(v, w)))
        # symmetry may canonicalize slot order afterwards
        assert any("metric invariance (E)" in s.rule for s in steps)
        chain = ExpandAndSimplify().prove(
            Act(E.anchor(u), E.metric(v, w)),
            Sum(E.metric(E.bracket(u, v), w), E.metric(v, E.bracket(u, w))),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_inert_without_declaration(self, setup):
        reg, _, E, u, v, w = setup
        E0 = algebroid("E", Bundle("E"), declare=("almost-courant",))
        eng = algebroid_engine(E0, registry=reg)
        node = Act(E0.anchor(u), E0.metric(v, w))
        out, steps = eng.expand(node)
        assert not any("metric invariance" in s.rule for s in steps)

    def test_scoped_to_algebroid(self, setup):
        """E's C1 must not fire on F's metric (the definitional
        symmetry sort MAY — it is global by design)."""
        reg, _, E, *_ = setup
        F = algebroid("F", Bundle("F"), declare=("metric",))
        uf, vf, wf = F.sections("u v w")
        eng = algebroid_engine(E, registry=reg)
        _, steps = eng.expand(Act(F.anchor(uf), F.metric(vf, wf)))
        assert not any("metric invariance" in s.rule for s in steps)


# --------------------------------------------------------------------- #
# C2 — symmetric part (declared)                                         #
# --------------------------------------------------------------------- #


class TestSymmetricPart:
    def test_pair_collects_to_correction(self, setup):
        reg, _, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(Sum(E.bracket(v, u), E.bracket(u, v)))
        assert out == E.sharp(E.D(E.metric(u, v)))
        assert any("symmetric part (E)" in s.rule for s in steps)

    def test_negated_pair(self, setup):
        reg, _, E, u, v, w = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(
            Sum(Neg(E.bracket(u, v)), E.bracket(u, w), Neg(E.bracket(v, u)))
        )
        assert out == Sum(
            E.bracket(u, w), Neg(E.sharp(E.D(E.metric(u, v))))
        )

    def test_lone_brackets_untouched(self, setup):
        """No single-bracket swap: with C1 it would form a rewrite
        cycle (documented in SymmetricPartDeclaration)."""
        reg, _, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        for node in (E.bracket(u, v), E.bracket(v, u), E.bracket(u, u)):
            out, steps = eng.expand(node)
            assert out == node and not steps

    def test_mixed_signs_not_collected(self, setup):
        """[u,v] − [v,u] is the ANTIsymmetrized bracket — not a C2
        shape; collecting it would be wrong."""
        reg, _, E, u, v, _ = setup
        eng = algebroid_engine(E, registry=reg)
        node = Sum(E.bracket(u, v), Neg(E.bracket(v, u)))
        out, steps = eng.expand(node)
        assert not any("symmetric part" in s.rule for s in steps)

    def test_equal_slot_pair(self, setup):
        reg, _, E, u, *_ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(Sum(E.bracket(u, u), E.bracket(u, u)))
        assert out == E.sharp(E.D(E.metric(u, u)))

    def test_swap_equation_form_closes(self, setup):
        """Dropping the swap rule loses NO equation: the obstruction
        of ``[v,u] = −[u,v] + g⁻¹Dg(u,v)`` contains the pair, which
        collects."""
        reg, _, E, u, v, _ = setup
        chain = ExpandAndSimplify().prove(
            E.bracket(v, u),
            Sum(Neg(E.bracket(u, v)), E.sharp(E.D(E.metric(u, v)))),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_cofactor_pair_collects(self, setup):
        """Scalar-coefficient pairs (the shape Leibniz splits leave
        behind): f[u,v] + f[v,u] = f·g⁻¹Dg(u,v)."""
        reg, f, E, u, v, _ = setup
        chain = ExpandAndSimplify().prove(
            Sum(
                Product(f, E.bracket(u, v)),
                Product(f, E.bracket(v, u)),
            ),
            Product(f, E.sharp(E.D(E.metric(u, v)))),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_mismatched_cofactors_not_collected(self, setup):
        """f[u,v] + [v,u] is NOT a C2 instance."""
        reg, f, E, u, v, _ = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Sum(Product(f, E.bracket(u, v)), E.bracket(v, u)),
                Product(f, E.sharp(E.D(E.metric(u, v)))),
                registry=reg,
                engine=algebroid_engine(E, registry=reg),
            )

    def test_equal_slot_singleton_forms(self, setup):
        """[u,u] = ½g⁻¹Dg(u,u) in all its coefficient variants (the
        sum-term singleton rule; safe against the C1 cycle because it
        never fires inside EMetric slots)."""
        reg, f, E, u, *_ = setup
        corr = E.sharp(E.D(E.metric(u, u)))
        eng = lambda: algebroid_engine(E, registry=reg)
        for lhs, rhs in (
            (E.bracket(u, u), Product(Rational(1, 2), corr)),
            (Product(Integer(2), E.bracket(u, u)), corr),
            (
                Product(f, E.bracket(u, u)),
                Product(Rational(1, 2), f, corr),
            ),
        ):
            chain = ExpandAndSimplify().prove(
                lhs, rhs, registry=reg, engine=eng()
            )
            assert chain.steps
        # the ½ is not optional
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                E.bracket(u, u), corr, registry=reg, engine=eng()
            )

    def test_equal_slot_with_invariance_terminates(self, setup):
        """ρ(u)(g(u,u)) = 2·g([u,u],u) — the exact shape that cycled
        under the bracket-anchored singleton rewrite; must both close
        and terminate under the sum-term rule."""
        reg, _, E, u, *_ = setup
        chain = ExpandAndSimplify().prove(
            Act(E.anchor(u), E.metric(u, u)),
            Product(Integer(2), E.metric(E.bracket(u, u), u)),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_symmetric_part_equation_closes(self, setup):
        reg, _, E, u, v, _ = setup
        chain = ExpandAndSimplify().prove(
            Sum(E.bracket(u, v), E.bracket(v, u)),
            E.sharp(E.D(E.metric(u, v))),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_paired_symmetric_part_evaluates(self, setup):
        """⟨sym-part correction against w⟩ — the C2 + sharp + pairing
        + coboundary chain composes: g([u,v]+[v,u], w) = ρ(w)(g(u,v))."""
        reg, _, E, u, v, w = setup
        chain = ExpandAndSimplify().prove(
            E.metric(Sum(E.bracket(u, v), E.bracket(v, u)), w),
            Act(E.anchor(w), E.metric(u, v)),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_inert_without_declaration(self, setup):
        reg, _, _, u, v, _ = setup
        E0 = algebroid("E", Bundle("E"), declare=("leibniz",))
        eng = algebroid_engine(E0, registry=reg)
        node = E0.bracket(v, u)
        out, steps = eng.expand(node)
        assert out == node and not steps


# --------------------------------------------------------------------- #
# Honesty: [B 4.11] is NOT free                                          #
# --------------------------------------------------------------------- #


class TestHonesty:
    def test_right_leibniz_open_under_metric(self, setup):
        """Right-Leibniz follows from C1 + non-degeneracy only through
        the 3.E.3 tactic — plain rewriting must NOT close it."""
        reg, f, E, u, v, _ = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                E.bracket(u, Product(f, v)),
                Sum(
                    Product(Act(E.anchor(u), f), v),
                    Product(f, E.bracket(u, v)),
                ),
                registry=reg,
                engine=algebroid_engine(E, registry=reg),
            )

    def test_false_metric_identity_fails(self, setup):
        """ρ(u)(g(v,w)) = 2·g([u,v],w) is simply wrong."""
        reg, _, E, u, v, w = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Act(E.anchor(u), E.metric(v, w)),
                Product(Integer(2), E.metric(E.bracket(u, v), w)),
                registry=reg,
                engine=algebroid_engine(E, registry=reg),
            )


# --------------------------------------------------------------------- #
# Phase 3.E.3 — the [B 4.11] tactics                                     #
# --------------------------------------------------------------------- #


from jacopy.proof.theorems import TheoremBook, cite
from jacopy.central.algebroid import (
    prove_left_leibniz_from_metric,
    prove_right_leibniz_from_metric,
)


def _walk(steps):
    for s in steps:
        yield s
        yield from _walk(s.children)


class TestRightLeibnizFromMetric:
    def test_closes_under_metric(self, setup):
        reg, f, E, u, v, _ = setup
        chain, thm = prove_right_leibniz_from_metric(
            E, u, v, f, registry=reg
        )
        assert chain.steps
        assert thm.lhs == E.bracket(u, Product(f, v))
        assert thm.rhs == Sum(
            Product(Act(E.anchor(u), f), v),
            Product(f, E.bracket(u, v)),
        )
        assert "non-degeneracy" in " ".join(thm.from_axioms)

    def test_uses_declared_c1(self, setup):
        reg, f, E, u, v, _ = setup
        chain, _ = prove_right_leibniz_from_metric(
            E, u, v, f, registry=reg
        )
        assert any(
            "metric invariance (E)" in s.rule for s in _walk(chain.steps)
        )
        assert any(
            "non-degeneracy" in s.rule for s in chain.steps
        )

    def test_fails_without_metric_invariance(self, setup):
        reg, f, *_ = setup
        E0 = algebroid("E", Bundle("E"), declare=("symmetric-part",))
        u, v = E0.sections("u v")
        with pytest.raises(ProofFailure):
            prove_right_leibniz_from_metric(E0, u, v, f, registry=reg)

    def test_declared_right_leibniz_is_axiom(self, setup):
        reg, f, *_ = setup
        E0 = algebroid(
            "E", Bundle("E"), declare=("almost-courant", "metric-invariance")
        )
        u, v = E0.sections("u v")
        with pytest.raises(ValueError):
            prove_right_leibniz_from_metric(E0, u, v, f, registry=reg)

    def test_tangent_rejected(self, setup):
        reg, f, *_ = setup
        TMalg = tangent_algebroid()
        X, Y = TMalg.sections("X Y")
        with pytest.raises(ValueError):
            prove_right_leibniz_from_metric(TMalg, X, Y, f, registry=reg)

    def test_citation_closes_the_open_statement(self, setup):
        """The statement test_right_leibniz_open_under_metric proves
        honest-open now closes WITH the derived theorem cited."""
        reg, f, E, u, v, _ = setup
        _, thm = prove_right_leibniz_from_metric(E, u, v, f, registry=reg)
        book = TheoremBook()
        book.add(thm)
        eng = algebroid_engine(E, registry=reg)
        cite(eng, book, thm.name)
        chain = ExpandAndSimplify().prove(
            E.bracket(u, Product(f, v)),
            Sum(
                Product(Act(E.anchor(u), f), v),
                Product(f, E.bracket(u, v)),
            ),
            registry=reg,
            engine=eng,
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_false_variant_fails_even_with_citation(self, setup):
        reg, f, E, u, v, _ = setup
        _, thm = prove_right_leibniz_from_metric(E, u, v, f, registry=reg)
        book = TheoremBook()
        book.add(thm)
        eng = algebroid_engine(E, registry=reg)
        cite(eng, book, thm.name)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                E.bracket(u, Product(f, v)),
                Sum(
                    Neg(Product(Act(E.anchor(u), f), v)),
                    Product(f, E.bracket(u, v)),
                ),
                registry=reg,
                engine=eng,
            )

    def test_arbitrary_names(self, setup):
        reg, f, *_ = setup
        M = algebroid("M", Bundle("M"), declare=("metric",))
        A, B = M.sections("A B")
        chain, thm = prove_right_leibniz_from_metric(
            M, A, B, f, registry=reg
        )
        assert thm.lhs == M.bracket(A, Product(f, B))


class TestLeftLeibnizFromMetric:
    def test_closes_with_concrete_locality_term(self, setup):
        reg, f, E, u, v, _ = setup
        chain, thm = prove_left_leibniz_from_metric(
            E, u, v, f, registry=reg
        )
        assert chain.steps
        assert thm.lhs == E.bracket(Product(f, u), v)
        # the concrete locality term g(u,v)·g⁻¹(Df) is in the rhs
        assert Product(
            E.metric(u, v), E.sharp(E.D(f))
        ) in thm.rhs.children

    def test_fails_without_symmetric_part(self, setup):
        reg, f, *_ = setup
        E0 = algebroid("E", Bundle("E"), declare=("metric-invariance",))
        u, v = E0.sections("u v")
        with pytest.raises(ProofFailure):
            prove_left_leibniz_from_metric(E0, u, v, f, registry=reg)

    def test_declared_left_leibniz_is_axiom(self, setup):
        reg, f, *_ = setup
        E0 = algebroid("E", Bundle("E"), declare=("local", "metric"))
        u, v = E0.sections("u v")
        with pytest.raises(ValueError):
            prove_left_leibniz_from_metric(E0, u, v, f, registry=reg)

    def test_cites_the_first_identity(self, setup):
        reg, f, E, u, v, _ = setup
        chain, _ = prove_left_leibniz_from_metric(E, u, v, f, registry=reg)
        assert any(
            s.provenance_tag == "theorem" for s in chain.steps
        )
        assert any(
            "symmetric part (E)" in s.rule for s in _walk(chain.steps)
        )
