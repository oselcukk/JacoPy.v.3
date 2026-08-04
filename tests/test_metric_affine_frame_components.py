"""Metric-affine package — Phase 4.D.1: frame components of the
connection (Γ^a_bc, the connection 1-form ω^a_b, torsion components,
and the first Cartan structure equation in frame-evaluated form)."""

import pytest

from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import connection
from jacopy.central.objects.frame import Frame
from jacopy.central.tangent.anholonomy import anholonomy_coefficient
from jacopy.packages.metric_affine import (
    connection_coefficient,
    connection_form,
    metric_affine_engine,
    prove_cartan_first_evaluated,
    prove_connection_form_on_frame,
    prove_torsion_components,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    nabla = connection()
    fr = Frame("e")
    return reg, nabla, fr


class TestCoefficientExtraction:
    def test_gamma_extraction(self, setup):
        reg, nabla, fr = setup
        eng = metric_affine_engine(registry=reg)
        node = Pairing(
            fr.dual().field("a"), nabla(fr.field("b"), fr.field("c"))
        )
        out, steps = eng.expand(node)
        assert out == connection_coefficient(nabla, fr, "a", "b", "c")
        assert any("connection coefficient" in s.rule for s in steps)

    def test_no_index_symmetry_imposed(self, setup):
        """Γ^a_bc and Γ^a_cb are distinct atoms (a general connection
        has no symmetry — unlike γ's canonical antisymmetry)."""
        _, nabla, fr = setup
        assert connection_coefficient(
            nabla, fr, "a", "b", "c"
        ) != connection_coefficient(nabla, fr, "a", "c", "b")

    def test_distinct_connections_and_frames(self, setup):
        _, nabla, fr = setup
        other = connection("D")
        fr2 = Frame("f")
        base = connection_coefficient(nabla, fr, "a", "b", "c")
        assert base != connection_coefficient(other, fr, "a", "b", "c")
        assert base != connection_coefficient(nabla, fr2, "a", "b", "c")

    def test_gamma_is_scalar(self, setup):
        from jacopy.central.calculus.scalars import is_scalar_function

        reg, nabla, fr = setup
        assert is_scalar_function(
            connection_coefficient(nabla, fr, "a", "b", "c"), reg
        )


class TestConnectionForm:
    def test_evaluation_on_frame(self, setup):
        """ω^a_b(e_c) = Γ^a_cb."""
        reg, nabla, fr = setup
        chain = prove_connection_form_on_frame(
            nabla, fr, "a", "b", "c", registry=reg
        )
        assert chain.steps

    def test_form_degree(self, setup):
        _, nabla, fr = setup
        assert str(connection_form(nabla, fr, "a", "b").degree) == "1"


class TestTorsionComponents:
    def test_components_theorem(self, setup):
        """⟨e^a, T(e_b,e_c)⟩ = Γ^a_bc − Γ^a_cb − γ^a_bc."""
        reg, nabla, fr = setup
        chain = prove_torsion_components(
            nabla, fr, "a", "b", "c", registry=reg
        )
        assert chain.steps

    def test_false_components_fail(self, setup):
        """Dropping the γ term breaks the identity."""
        reg, nabla, fr = setup
        from jacopy.packages.metric_affine.torsion_curvature import torsion

        lhs = Pairing(
            fr.dual().field("a"),
            torsion(nabla, fr.field("b"), fr.field("c")),
        )
        rhs = Sum(
            connection_coefficient(nabla, fr, "a", "b", "c"),
            Neg(connection_coefficient(nabla, fr, "a", "c", "b")),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                rhs,
                registry=reg,
                engine=metric_affine_engine(registry=reg),
            )


class TestCartanFirstEvaluated:
    def test_closes(self, setup):
        """T^a(e_b,e_c) = de^a(e_b,e_c) + ω^a_c(e_b) − ω^a_b(e_c) —
        Palais + coefficient extraction + γ-cancellation together."""
        reg, nabla, fr = setup
        chain = prove_cartan_first_evaluated(
            nabla, fr, "a", "b", "c", registry=reg
        )
        assert chain.steps

    def test_wrong_sign_fails(self, setup):
        reg, nabla, fr = setup
        from jacopy.core.multi_eval import MultiEval
        from jacopy.central.tangent.exterior import d
        from jacopy.packages.metric_affine.frame_components import (
            ConnectionFormEvaluationDefinition,
        )
        from jacopy.packages.metric_affine.torsion_curvature import torsion

        e_up = fr.dual().field("a")
        eb, ec = fr.field("b"), fr.field("c")
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
                    Pairing(connection_form(nabla, fr, "a", "b"), ec),
                    Neg(Pairing(connection_form(nabla, fr, "a", "c"), eb)),
                ),
                registry=reg,
                engine=eng,
            )

    def test_holonomic_frame_specialization(self, setup):
        """On a holonomic frame γ = 0: T^a components reduce to the
        Γ-antisymmetrization alone."""
        reg, nabla, _ = setup
        from jacopy.packages.metric_affine.torsion_curvature import torsion

        fr = Frame("p")
        eng = metric_affine_engine(registry=reg, holonomic_frames=(fr,))
        chain = ExpandAndSimplify().prove(
            Pairing(
                fr.dual().field("a"),
                torsion(nabla, fr.field("b"), fr.field("c")),
            ),
            Sum(
                connection_coefficient(nabla, fr, "a", "b", "c"),
                Neg(connection_coefficient(nabla, fr, "a", "c", "b")),
            ),
            registry=reg,
            engine=eng,
        )
        assert chain.steps
