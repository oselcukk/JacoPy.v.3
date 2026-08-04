"""Metric-affine package — Phase 4.E.2: the modified E-torsion ᴸT and
the projected E-curvature ᴸ̂R [MC Def 3.7/3.11, Prop 3.2-3.4, Cor 3.3]
— the main consumers of the Phase 3.E locality/projector machinery."""

import pytest

from jacopy.core.expr import Product
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle, functions
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid import algebroid, locality_projector
from jacopy.packages.metric_affine.e_connection import e_connection
from jacopy.packages.metric_affine.modified import (
    CoframeRecombinationDefinition,
    modified_torsion,
    projected_curvature,
    prove_modified_torsion_tensorial,
    prove_projected_curvature_first_slot,
    prove_projected_curvature_last_slot,
    _engine,
)
from jacopy.packages.metric_affine.torsion_curvature import torsion


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("local", "regular"))
    u, v, w = E.sections("u v w")
    nabla = e_connection(E)
    fr = Frame("e", bundle=E.bundle)
    return reg, f, E, u, v, w, nabla, fr


class TestCoframeRecombination:
    def test_recombines_locality_sum(self, setup):
        """Σ_s ρ(e_s)(f)·L(e^s,u,v) → L(Df,u,v) — coframe completeness
        in the collection direction."""
        from jacopy.algebra.derivation import Act
        from jacopy.central.algebroid import LocalityOperator, locality_term
        from jacopy.central.objects.frame import FrameIndex

        reg, f, E, u, v, _, nabla, fr = setup
        eng = _engine(E, fr, reg)
        node = IndexedSum(
            FrameIndex("s"),
            fr,
            Product(
                Act(E.anchor(fr.field("s")), f),
                LocalityOperator(
                    E.name, fr.dual().field("s"), u, v
                ),
            ),
        )
        out, steps = eng.expand(node)
        assert out == locality_term(E, f, u, v)
        assert any("coframe recombination" in s.rule for s in steps)

    def test_unsound_leftover_blocks(self, setup):
        """A leftover factor still carrying the bound index (outside
        the substituted coframe) blocks the collection."""
        from jacopy.algebra.derivation import Act
        from jacopy.central.algebroid import LocalityOperator
        from jacopy.central.objects.frame import FrameIndex

        reg, f, E, u, v, _, nabla, fr = setup
        eng = _engine(E, fr, reg)
        node = IndexedSum(
            FrameIndex("s"),
            fr,
            Product(
                Act(E.anchor(fr.field("s")), f),
                LocalityOperator(
                    E.name, fr.dual().field("s"), fr.field("s"), v
                ),
            ),
        )
        out, steps = eng.expand(node)
        assert not any(
            "coframe recombination" in s.rule for s in steps
        )


class TestModifiedTorsion:
    def test_tensorial_both_slots(self, setup):
        """ᴸT is a genuine tensor on a local algebroid — the
        recombined L(Df,u,v) cancels the 4.E.1 pseudo defect."""
        reg, f, E, u, v, _, nabla, fr = setup
        first, second = prove_modified_torsion_tensorial(
            E, nabla, fr, u, v, f, registry=reg
        )
        assert first.steps and second.steps

    def test_unmodified_pseudo_still_fails(self, setup):
        """The bare T⁰ (no correction) stays non-tensorial even with
        the recombination rule available — the defect is genuine."""
        reg, f, E, u, v, _, nabla, fr = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                torsion(nabla, Product(f, u), v),
                Product(f, torsion(nabla, u, v)),
                registry=reg,
                engine=_engine(E, fr, reg),
            )

    def test_frame_bundle_guard(self, setup):
        reg, f, E, u, v, _, nabla, _ = setup
        wrong_frame = Frame("e")  # TM frame, not E
        with pytest.raises(ValueError):
            modified_torsion(E, nabla, wrong_frame, u, v)


class TestProjectedCurvature:
    def test_first_slot_tensorial(self, setup):
        """ᴸ̂R(fu,v,w) = f·ᴸ̂R — recombination meets the projector's
        ABSORPTION rule (L̂ ∈ [L̃] on coboundary values)."""
        reg, f, E, u, v, w, nabla, fr = setup
        P = locality_projector(E)
        chain = prove_projected_curvature_first_slot(
            E, nabla, fr, P, u, v, w, f, registry=reg
        )
        assert chain.steps

    def test_last_slot_under_morphism(self, setup):
        """ᴸ̂R(u,v,fw) = f·ᴸ̂R — the R⁰ part needs the anchor morphism
        and the correction's Leibniz term dies by the projector's
        KERNEL rule (im L̂ ⊂ ker ρ)."""
        reg, f, *_ = setup
        E = algebroid(
            "E", Bundle("E"), declare=("local", "pre-leibniz", "regular")
        )
        u, v, w = E.sections("u v w")
        nabla = e_connection(E)
        fr = Frame("e", bundle=E.bundle)
        P = locality_projector(E)
        chain = prove_projected_curvature_last_slot(
            E, nabla, fr, P, u, v, w, f, registry=reg
        )
        assert chain.steps

    def test_last_slot_honest_without_morphism(self, setup):
        reg, f, E, u, v, w, nabla, fr = setup
        P = locality_projector(E)
        with pytest.raises(ProofFailure):
            prove_projected_curvature_last_slot(
                E, nabla, fr, P, u, v, w, f, registry=reg
            )

    def test_requires_projector_type(self, setup):
        reg, f, E, u, v, w, nabla, fr = setup
        with pytest.raises(TypeError):
            projected_curvature(
                E, nabla, fr, "not-a-projector", u, v, w
            )


class TestGeneralizedRicciIdentity:
    """Phase 4.F.4 [ADM Prop 3.8]: the generalized Ricci identity —
    definitional, arbitrary connection."""

    def test_closes_without_declarations(self, setup):
        from jacopy.central.algebroid import locality_projector
        from jacopy.packages.metric_affine.modified import (
            prove_generalized_ricci_identity,
        )

        reg, f, E, u, v, w, nabla, fr = setup
        P = locality_projector(E)
        assert prove_generalized_ricci_identity(
            E, nabla, fr, P, u, v, w, registry=reg
        ).steps

    def test_anomaly_needed(self, setup):
        """Dropping the (1−𝒫) anomaly must break the identity —
        the gap between ᴸT's L and ᴸ̂R's 𝒫L is real."""
        from jacopy.algebra.derivation import Act
        from jacopy.core.expr import Neg, Sum
        from jacopy.central.objects.connection import CovariantOp
        from jacopy.central.algebroid import locality_projector
        from jacopy.packages.metric_affine.modified import (
            modified_torsion,
            projected_curvature,
        )
        from jacopy.proof.strategies import (
            ExpandAndSimplify,
            ProofFailure,
        )

        reg, f, E, u, v, w, nabla, fr = setup
        P = locality_projector(E)

        def cov(direction, arg):
            return Act(
                CovariantOp(
                    nabla.name, direction, bundle=nabla.bundle
                ),
                arg,
            )

        lhs = Sum(
            nabla(u, nabla(v, w)),
            Neg(nabla(v, nabla(u, w))),
            Neg(cov(nabla(u, v), w)),
            cov(nabla(v, u), w),
        )
        rhs_no_anomaly = Sum(
            projected_curvature(E, nabla, fr, P, u, v, w),
            Neg(cov(modified_torsion(E, nabla, fr, u, v), w)),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                rhs_no_anomaly,
                registry=reg,
                engine=_engine(E, fr, reg, projectors=(P,)),
            )
