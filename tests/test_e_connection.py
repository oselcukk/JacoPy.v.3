"""Metric-affine package — Phase 4.E.1: linear E-connections and the
pseudo E-torsion/E-curvature [MC Def 3.2-3.3, Prop 3.3].

The punchlines: the pseudo-torsion's first slot carries the locality
DEFECT (why the modified ᴸT exists), and the E-curvature's last-slot
tensoriality is exactly the anchor-morphism condition (MC Prop 3.3,
conditional + honest)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import TheoremBook, cite
from jacopy.central.objects import Bundle, functions
from jacopy.central.algebroid import (
    algebroid,
    locality_term,
    prove_anchor_morphism,
)
from jacopy.packages.metric_affine.e_connection import (
    e_connection,
    e_connection_engine,
    prove_e_connection_anchored_leibniz,
    prove_e_curvature_last_slot_conditional,
    prove_e_torsion_second_slot_tensorial,
    prove_pseudo_torsion_first_slot_defect,
)
from jacopy.packages.metric_affine.torsion_curvature import (
    curvature,
    torsion,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("local",))
    u, v, w = E.sections("u v w")
    nabla = e_connection(E)
    return reg, f, E, u, v, w, nabla


class TestEConnection:
    def test_anchored_scalar_action(self, setup):
        reg, f, E, u, *_ = setup
        nabla = e_connection(E)
        eng = e_connection_engine(E, registry=reg)
        out, _ = eng.expand(nabla(u, f))
        assert out == Act(E.anchor(u), f)

    def test_anchored_leibniz_inherited(self, setup):
        """∇_u(fv) = ρ(u)(f)v + f∇_u v — product_rule + the anchored
        scalar action; no new axiom."""
        reg, f, E, u, v, _, nabla = setup
        assert prove_e_connection_anchored_leibniz(
            E, nabla, u, v, f, registry=reg
        ).steps

    def test_direction_linearity(self, setup):
        reg, f, E, u, v, w, nabla = setup
        chain = ExpandAndSimplify().prove(
            nabla(Product(f, u), w),
            Product(f, nabla(u, w)),
            registry=reg,
            engine=e_connection_engine(E, registry=reg),
        )
        assert chain.steps


class TestPseudoTorsion:
    def test_expansion_uses_algebroid_bracket(self, setup):
        reg, _, E, u, v, _, nabla = setup
        eng = e_connection_engine(E, registry=reg)
        out, steps = eng.expand(torsion(nabla, u, v))
        assert out == Sum(
            nabla(u, v), Neg(nabla(v, u)), Neg(E.bracket(u, v))
        )
        assert any("E-torsion definition" in s.rule for s in steps)

    def test_second_slot_tensorial(self, setup):
        reg, f, E, u, v, _, nabla = setup
        assert prove_e_torsion_second_slot_tensorial(
            E, nabla, u, v, f, registry=reg
        ).steps

    def test_first_slot_defect(self, setup):
        """T⁰(fu,v) = f·T⁰(u,v) − L(Df,u,v) — the pseudo-tensor's
        honest defect; the reason ᴸT exists."""
        reg, f, E, u, v, _, nabla = setup
        assert prove_pseudo_torsion_first_slot_defect(
            E, nabla, u, v, f, registry=reg
        ).steps

    def test_first_slot_naive_tensoriality_fails(self, setup):
        """Claiming plain tensoriality (no L-term) must fail on a
        local algebroid."""
        reg, f, E, u, v, _, nabla = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                torsion(nabla, Product(f, u), v),
                Product(f, torsion(nabla, u, v)),
                registry=reg,
                engine=e_connection_engine(E, registry=reg),
            )


class TestECurvatureConditional:
    def test_last_slot_closes_under_morphism(self, setup):
        """MC Prop 3.3: the ΓΓ-free obstruction is exactly the anchor
        commutator; pre-Leibniz kills it."""
        reg, f, *_ = setup
        E = algebroid("E", Bundle("E"), declare=("pre-leibniz",))
        u, v, w = E.sections("u v w")
        nabla = e_connection(E)
        assert prove_e_curvature_last_slot_conditional(
            E, nabla, u, v, w, f, registry=reg
        ).steps

    def test_honest_without_morphism(self, setup):
        reg, f, *_ = setup
        E = algebroid("E", Bundle("E"), declare=("almost-leibniz",))
        u, v, w = E.sections("u v w")
        nabla = e_connection(E)
        with pytest.raises(ProofFailure):
            prove_e_curvature_last_slot_conditional(
                E, nabla, u, v, w, f, registry=reg
            )

    def test_closes_on_leibniz_via_cited_flagship(self, setup):
        """On a Leibniz algebroid the morphism is derived (3.D) and
        cited — MC Prop 3.3 meets the flagship theorem."""
        reg, f, *_ = setup
        E = algebroid("E", Bundle("E"), declare=("leibniz",))
        u, v, w = E.sections("u v w")
        nabla = e_connection(E)
        _, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        book = TheoremBook()
        book.add(thm)
        engine = e_connection_engine(E, registry=reg)
        cite(engine, book, thm.name)
        chain = ExpandAndSimplify().prove(
            curvature(nabla, u, v, Product(f, w)),
            Product(f, curvature(nabla, u, v, w)),
            registry=reg,
            engine=engine,
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)


class TestScoping:
    def test_tm_torsion_untouched_by_e_rules(self, setup):
        """A TM torsion node expands with the LIE bracket even when an
        E-engine is around (bundle guards on both expansions)."""
        from jacopy.algebra.lie_bracket_vf import LieBracketVF
        from jacopy.central.objects import connection, vector_fields
        from jacopy.packages.metric_affine import metric_affine_engine

        reg, *_ = setup
        X, Y = vector_fields("X Y")
        tm_conn = connection()
        eng = metric_affine_engine(registry=reg)
        out, _ = eng.expand(torsion(tm_conn, X, Y))
        assert out == Sum(
            tm_conn(X, Y),
            Neg(tm_conn(Y, X)),
            Neg(LieBracketVF(X, Y)),
        )

    def test_arbitrary_names(self, setup):
        reg, f, *_ = setup
        M = algebroid("M", Bundle("M"), declare=("local",))
        A, B = M.sections("ξ ζ")
        nabla = e_connection(M, "D")
        assert prove_pseudo_torsion_first_slot_defect(
            M, nabla, A, B, f, registry=reg
        ).steps
