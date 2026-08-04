"""Phase 4.D.2a: the IndexedSum/Wedge rule layer and the form-valued
first Cartan structure equation.

The core nodes existed since Phase 0/1 (with α-equivalence); this
phase supplies their engine semantics via the name-based index
substitution protocol (FrameIndex + `index_names`/`substitute_atom`
overrides on the indexed atoms)."""

import pytest

from jacopy.core.expr import Integer, Neg, One, Product, Sum
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import connection, functions
from jacopy.central.objects.frame import Frame, FrameIndex, kronecker_delta
from jacopy.packages.metric_affine import (
    ConnectionFormEvaluationDefinition,
    connection_coefficient,
    connection_form,
    metric_affine_engine,
)
from jacopy.packages.metric_affine.frame_components import (
    prove_cartan_first_structure,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    nabla = connection()
    fr = Frame("e")
    return reg, nabla, fr


class TestIndexSubstitution:
    def test_name_based_renaming(self, setup):
        _, nabla, fr = setup
        dummy = FrameIndex("s")
        gamma = connection_coefficient(nabla, fr, "a", "b", "s")
        out = gamma.substitute_atom(dummy, FrameIndex("c"))
        assert out == connection_coefficient(nabla, fr, "a", "b", "c")

    def test_delta_collapses_on_equal_names(self, setup):
        delta = kronecker_delta("s", "c")
        out = delta.substitute_atom(FrameIndex("s"), FrameIndex("c"))
        assert out == One

    def test_alpha_equivalence_of_sums(self, setup):
        """Two sums differing only in the bound name are EQUAL — the
        sentinel machinery renames through the string slots."""
        _, nabla, fr = setup
        body = lambda n: Wedge(
            connection_form(nabla, fr, "a", n), fr.dual().field(n)
        )
        s1 = IndexedSum(FrameIndex("s"), fr, body("s"))
        s2 = IndexedSum(FrameIndex("t"), fr, body("t"))
        assert s1 == s2

    def test_free_index_untouched(self, setup):
        _, nabla, fr = setup
        gamma = connection_coefficient(nabla, fr, "a", "b", "c")
        assert (
            gamma.substitute_atom(FrameIndex("s"), FrameIndex("d"))
            is gamma
        )


class TestIndexedSumRules:
    def test_linearity_split(self, setup):
        reg, nabla, fr = setup
        eng = metric_affine_engine(registry=reg)
        dummy = FrameIndex("s")
        g1 = connection_coefficient(nabla, fr, "a", "b", "s")
        g2 = connection_coefficient(nabla, fr, "a", "c", "s")
        out, _ = eng.expand(IndexedSum(dummy, fr, Sum(g1, g2)))
        assert out == Sum(
            IndexedSum(dummy, fr, g1), IndexedSum(dummy, fr, g2)
        )

    def test_free_factor_pulls_out(self, setup):
        reg, nabla, fr = setup
        (f,) = functions("f", registry=reg)
        eng = metric_affine_engine(registry=reg)
        dummy = FrameIndex("s")
        g = connection_coefficient(nabla, fr, "a", "b", "s")
        out, _ = eng.expand(IndexedSum(dummy, fr, Product(f, g)))
        assert out == Product(f, IndexedSum(dummy, fr, g))

    def test_fully_free_body_left_inert(self, setup):
        """Σ_s (body free of s) would need the symbolic dimension —
        left inert, honestly."""
        reg, nabla, fr = setup
        eng = metric_affine_engine(registry=reg)
        g = connection_coefficient(nabla, fr, "a", "b", "c")
        node = IndexedSum(FrameIndex("s"), fr, g)
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_kronecker_contraction(self, setup):
        reg, nabla, fr = setup
        eng = metric_affine_engine(registry=reg)
        dummy = FrameIndex("s")
        body = Product(
            kronecker_delta("s", "c"),
            connection_coefficient(nabla, fr, "a", "b", "s"),
        )
        out, steps = eng.expand(IndexedSum(dummy, fr, body))
        assert out == connection_coefficient(nabla, fr, "a", "b", "c")
        assert any("Kronecker contraction" in s.rule for s in steps)

    def test_bare_delta_sums_to_one(self, setup):
        reg, _, fr = setup
        eng = metric_affine_engine(registry=reg)
        out, _ = eng.expand(
            IndexedSum(FrameIndex("s"), fr, kronecker_delta("s", "c"))
        )
        assert out == One

    def test_eval_pushes_in(self, setup):
        reg, nabla, fr = setup
        eng = metric_affine_engine(registry=reg)
        dummy = FrameIndex("s")
        ws = IndexedSum(
            dummy,
            fr,
            Wedge(
                connection_form(nabla, fr, "a", "s"),
                fr.dual().field("s"),
            ),
        )
        node = MultiEval(
            ws, fr.field("b"), fr.field("c"),
            alternating=True, slot_kind="vector",
        )
        eng.register(ConnectionFormEvaluationDefinition(nabla, fr))
        out, _ = eng.expand(node)
        assert out == Sum(
            connection_coefficient(nabla, fr, "a", "b", "c"),
            Neg(connection_coefficient(nabla, fr, "a", "c", "b")),
        )


class TestWedgeEval:
    def test_determinant_convention(self, setup):
        reg, nabla, fr = setup
        eng = metric_affine_engine(registry=reg)
        alpha = fr.dual().field("a")
        beta = fr.dual().field("b")
        node = MultiEval(
            Wedge(alpha, beta), fr.field("b"), fr.field("c"),
            alternating=True, slot_kind="vector",
        )
        out, _ = eng.expand(node)
        # ⟨e^a,e_b⟩⟨e^b,e_c⟩ − ⟨e^a,e_c⟩⟨e^b,e_b⟩ → δ-values; the
        # engine keeps the literal Product(δ^a_c, 1) — folding is
        # simplify's job.
        assert out == Sum(
            Product(kronecker_delta("a", "b"), kronecker_delta("b", "c")),
            Neg(Product(kronecker_delta("a", "c"), One)),
        )


class TestCartanFirstStructure:
    def test_closes_with_genuine_indexed_sum(self, setup):
        """⟨e^a, T(e_b,e_c)⟩ = de^a(e_b,e_c) + (Σ_s ω^a_s ∧ e^s)(e_b,e_c)
        — the full 4.D.2 chain: push-in, wedge determinant, duality,
        Kronecker contraction, torsion components."""
        reg, nabla, fr = setup
        chain = prove_cartan_first_structure(
            nabla, fr, "a", "b", "c", registry=reg
        )
        assert chain.steps

    def test_bound_index_collision_guard(self, setup):
        reg, nabla, fr = setup
        with pytest.raises(ValueError):
            prove_cartan_first_structure(
                nabla, fr, "a", "b", "s", bound="s", registry=reg
            )

    def test_alpha_equivalent_bound_name(self, setup):
        """The proof is independent of the bound-index name."""
        reg, nabla, fr = setup
        chain = prove_cartan_first_structure(
            nabla, fr, "a", "b", "c", bound="t", registry=reg
        )
        assert chain.steps

    def test_wrong_sign_fails(self, setup):
        from jacopy.central.tangent.exterior import d
        from jacopy.packages.metric_affine.torsion_curvature import torsion

        reg, nabla, fr = setup
        e_up = fr.dual().field("a")
        eb, ec = fr.field("b"), fr.field("c")
        dummy = FrameIndex("s")
        ws = IndexedSum(
            dummy,
            fr,
            Wedge(
                connection_form(nabla, fr, "a", "s"),
                fr.dual().field("s"),
            ),
        )
        eng = metric_affine_engine(registry=reg)
        eng.register(ConnectionFormEvaluationDefinition(nabla, fr))
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Pairing(e_up, torsion(nabla, eb, ec)),
                Sum(
                    MultiEval(
                        d(e_up), eb, ec,
                        alternating=True, slot_kind="vector",
                    ),
                    Neg(
                        MultiEval(
                            ws, eb, ec,
                            alternating=True, slot_kind="vector",
                        )
                    ),
                ),
                registry=reg,
                engine=eng,
            )


# --------------------------------------------------------------------- #
# Phase 4.D.2b — decompositions, Cartan II, curvature/Ricci components  #
# --------------------------------------------------------------------- #


from jacopy.packages.metric_affine import (
    curvature_component_formula,
    prove_cartan_second_structure,
    prove_curvature_components,
    prove_ricci_components,
    ricci_component,
)
from jacopy.packages.metric_affine.decomposition import (
    BracketFrameDecompositionDefinition,
    ConnectionFrameDecompositionDefinition,
)


class TestDecompositions:
    def test_connection_decomposition(self, setup):
        reg, nabla, fr = setup
        eng = metric_affine_engine(
            registry=reg, decompositions=((nabla, fr),)
        )
        out, steps = eng.expand(nabla(fr.field("b"), fr.field("c")))
        assert isinstance(out, IndexedSum)
        assert any("frame decomposition" in s.rule for s in steps)

    def test_bracket_decomposition(self, setup):
        from jacopy.algebra.lie_bracket_vf import LieBracketVF

        reg, nabla, fr = setup
        eng = metric_affine_engine(
            registry=reg, decompositions=((nabla, fr),)
        )
        out, _ = eng.expand(LieBracketVF(fr.field("a"), fr.field("b")))
        assert isinstance(out, IndexedSum)

    def test_fresh_bound_name_avoids_free_indices(self, setup):
        """∇_{e_s} e_c must NOT pick 's' as its bound name."""
        reg, nabla, fr = setup
        eng = metric_affine_engine(
            registry=reg, decompositions=((nabla, fr),)
        )
        out, _ = eng.expand(nabla(fr.field("s"), fr.field("c")))
        assert isinstance(out, IndexedSum)
        assert out.dummy._repr_inner() != "s"

    def test_inert_without_declaration(self, setup):
        reg, nabla, fr = setup
        eng = metric_affine_engine(registry=reg)
        node = nabla(fr.field("b"), fr.field("c"))
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_consistent_with_extraction(self, setup):
        """⟨e^a, ∇_eb ec⟩ gives Γ^a_bc via EITHER path (extraction or
        decomposition + duality + contraction) — confluent."""
        reg, nabla, fr = setup
        eng = metric_affine_engine(
            registry=reg, decompositions=((nabla, fr),)
        )
        chain = ExpandAndSimplify().prove(
            Pairing(
                fr.dual().field("a"),
                nabla(fr.field("b"), fr.field("c")),
            ),
            connection_coefficient(nabla, fr, "a", "b", "c"),
            registry=reg,
            engine=eng,
        )
        assert chain.steps


class TestCurvatureComponents:
    def test_classical_formula(self, setup):
        """R^a_bcd = e_c(Γ^a_db) − e_d(Γ^a_cb) + ΣΓΓ − ΣΓΓ − Σγ Γ."""
        reg, nabla, fr = setup
        chain = prove_curvature_components(
            nabla, fr, "a", "b", "c", "d", registry=reg
        )
        assert chain.steps

    def test_wrong_gamma_sign_fails(self, setup):
        from jacopy.packages.metric_affine.torsion_curvature import (
            curvature,
        )
        from jacopy.central.tangent.anholonomy import (
            anholonomy_coefficient,
        )

        reg, nabla, fr = setup
        wrong = Sum(
            curvature_component_formula(nabla, fr, "a", "b", "c", "d"),
            IndexedSum(
                FrameIndex("s"),
                fr,
                Product(
                    connection_coefficient(nabla, fr, "a", "s", "b"),
                    anholonomy_coefficient(fr, "s", "c", "d"),
                ),
            ),
        )
        lhs = Pairing(
            fr.dual().field("a"),
            curvature(
                nabla, fr.field("c"), fr.field("d"), fr.field("b")
            ),
        )
        from jacopy.packages.metric_affine.frame_components import (
            _decomposed_engine,
        )

        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                wrong,
                registry=reg,
                engine=_decomposed_engine(nabla, fr, reg),
            )


class TestCartanSecondStructure:
    def test_closes(self, setup):
        """⟨e^a, R(e_c,e_d)e_b⟩ = (dω^a_b + Σ_s ω^a_s ∧ ω^s_b)(e_c,e_d)
        — nested sums α-convert on capture; shadowed binders resolve
        via the depth-aware α-hash."""
        reg, nabla, fr = setup
        chain = prove_cartan_second_structure(
            nabla, fr, "a", "b", "c", "d", registry=reg
        )
        assert chain.steps

    def test_bound_collision_guard(self, setup):
        reg, nabla, fr = setup
        with pytest.raises(ValueError):
            prove_cartan_second_structure(
                nabla, fr, "a", "b", "c", "s", bound="s", registry=reg
            )


class TestRicci:
    def test_component_formula(self, setup):
        """Ric_bd = Σ_s (trace of the component formula)."""
        reg, nabla, fr = setup
        chain = prove_ricci_components(nabla, fr, "b", "d", registry=reg)
        assert chain.steps

    def test_definition_shape(self, setup):
        _, nabla, fr = setup
        ric = ricci_component(nabla, fr, "b", "d")
        assert isinstance(ric, IndexedSum)

    def test_bound_collision_guard(self, setup):
        _, nabla, fr = setup
        with pytest.raises(ValueError):
            ricci_component(nabla, fr, "s", "d", bound="s")


# --------------------------------------------------------------------- #
# Phase 4.D.2c — inverse metric, Ricci scalar, Einstein                 #
# --------------------------------------------------------------------- #


from jacopy.packages.metric_affine import metric
from jacopy.packages.metric_affine.metric import (
    MetricValue,
    inverse_metric_component,
)
from jacopy.packages.metric_affine.frame_components import (
    einstein_component,
    prove_ricci_scalar_components,
    ricci_scalar,
)


class TestInverseMetric:
    def test_defining_contraction(self, setup):
        reg, nabla, fr = setup
        g = metric()
        eng = metric_affine_engine(registry=reg)
        node = IndexedSum(
            FrameIndex("s"),
            fr,
            Product(
                inverse_metric_component(g, "a", "s"),
                g(fr.field("s"), fr.field("c")),
            ),
        )
        out, steps = eng.expand(node)
        assert out == kronecker_delta("a", "c")
        assert any("inverse metric" in s.rule for s in steps)

    def test_symmetric_canonical_order(self, setup):
        g = metric()
        assert inverse_metric_component(
            g, "b", "a"
        ) == inverse_metric_component(g, "a", "b")

    def test_distinct_metrics_do_not_contract(self, setup):
        reg, nabla, fr = setup
        g, h = metric(), metric("h")
        eng = metric_affine_engine(registry=reg)
        node = IndexedSum(
            FrameIndex("s"),
            fr,
            Product(
                inverse_metric_component(h, "a", "s"),
                g(fr.field("s"), fr.field("c")),
            ),
        )
        out, steps = eng.expand(node)
        # only the (global, definitional) symmetry sort may fire —
        # never the contraction across DIFFERENT metrics
        assert not any("inverse metric" in s.rule for s in steps)
        assert isinstance(out, IndexedSum)

    def test_rest_capture_guard(self, setup):
        """A leftover factor still carrying the bound index blocks the
        contraction (removing the sum would be unsound)."""
        reg, nabla, fr = setup
        g = metric()
        eng = metric_affine_engine(registry=reg)
        node = IndexedSum(
            FrameIndex("s"),
            fr,
            Product(
                inverse_metric_component(g, "a", "s"),
                g(fr.field("s"), fr.field("c")),
                connection_coefficient(nabla, fr, "a", "b", "s"),
            ),
        )
        out, steps = eng.expand(node)
        assert not any("inverse metric" in s.rule for s in steps)

    def test_metric_value_slot_substitution(self, setup):
        _, _, fr = setup
        g = metric()
        gv = g(fr.field("s"), fr.field("c"))
        out = gv.substitute_atom(FrameIndex("s"), FrameIndex("d"))
        assert out == g(fr.field("d"), fr.field("c"))


class TestRicciScalarEinstein:
    def test_ricci_scalar_unfolds(self, setup):
        reg, nabla, fr = setup
        g = metric()
        chain = prove_ricci_scalar_components(nabla, fr, g, registry=reg)
        assert chain.steps

    def test_constructor_guards(self, setup):
        _, nabla, fr = setup
        g = metric()
        with pytest.raises(ValueError):
            ricci_scalar(nabla, fr, g, bounds=("s", "s", "t"))
        with pytest.raises(TypeError):
            einstein_component(nabla, fr, "not-a-metric", "b", "d")

    def test_einstein_shape(self, setup):
        _, nabla, fr = setup
        g = metric()
        G = einstein_component(nabla, fr, g, "b", "d")
        assert isinstance(G, Sum) and len(G.children) == 2
