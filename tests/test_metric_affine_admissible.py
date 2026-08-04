"""Metric-affine package — Phase 4.F.1: admissible E-connections
(ADM = Dereli-Doğan JGP 186 (2023)) + the projector-definition audit
(admissible Def 3.8 vs MC Def 3.10)."""

import pytest

from jacopy.core.expr import Integer, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle, functions
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid import algebroid, locality_projector
from jacopy.packages.metric_affine.e_connection import e_connection
from jacopy.packages.metric_affine.modified import (
    _engine,
    modified_torsion,
)
from jacopy.packages.metric_affine.admissible import (
    AdmissibilityDeclaration,
    ModifiedTorsionFreeDeclaration,
    modified_bracket,
    prove_modified_bracket_left_leibniz,
    prove_modified_torsion_symmetric_part_vanishes,
    prove_torsion_free_implies_admissible,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("local", "regular"))
    u, v, w = E.sections("u v w")
    nabla = e_connection(E)
    fr = Frame("e", bundle=E.bundle)
    return reg, f, E, u, v, w, nabla, fr


class TestTorsionFreeImpliesAdmissible:
    def test_closes(self, setup):
        """[ADM Prop 3.3]: declared ᴸT = 0 ⟹ admissibility
        equation closes."""
        reg, _, E, u, v, _, nabla, fr = setup
        assert prove_torsion_free_implies_admissible(
            E, nabla, fr, u, v, registry=reg
        ).steps

    def test_honest_without_declaration(self, setup):
        """Without the ᴸT-free declaration the admissibility
        equation must NOT close (it is not a general truth)."""
        from jacopy.packages.metric_affine.admissible import (
            _locality_sum,
        )

        reg, _, E, u, v, _, nabla, fr = setup
        lhs = Sum(E.bracket(u, v), E.bracket(v, u))
        rhs = Sum(
            _locality_sum(E, nabla, fr, u, v, "s"),
            _locality_sum(E, nabla, fr, v, u, "s"),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                rhs,
                registry=reg,
                engine=_engine(E, fr, reg),
            )


class TestModifiedTorsionAntisymmetry:
    def test_closes_with_declaration(self, setup):
        """[ADM Cor 3.2]: ᴸT(u,v) + ᴸT(v,u) = 0 for an admissible
        connection."""
        reg, _, E, u, v, _, nabla, fr = setup
        assert prove_modified_torsion_symmetric_part_vanishes(
            E, nabla, fr, u, v, registry=reg
        ).steps

    def test_honest_without_declaration(self, setup):
        reg, _, E, u, v, _, nabla, fr = setup
        lhs = Sum(
            modified_torsion(E, nabla, fr, u, v),
            modified_torsion(E, nabla, fr, v, u),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                Integer(0),
                registry=reg,
                engine=_engine(E, fr, reg),
            )


class TestModifiedBracketDull:
    def test_left_leibniz_exact(self, setup):
        """[ADM Prop 3.2, dull part]: [fu,v]^∇ = f[u,v]^∇ − ρ(v)(f)u
        for ANY connection — no declaration needed."""
        reg, f, E, u, v, _, nabla, fr = setup
        assert prove_modified_bracket_left_leibniz(
            E, nabla, fr, u, v, f, registry=reg
        ).steps


class TestProjectorDefinitionAudit:
    """Literature audit 2026-07-31: MC Def 3.10 (absorption+kernel)
    vs admissible Def 3.8 (+ 𝒫|_ker ρ = id ⇒ idempotence on
    projected values)."""

    def test_default_is_mc_definition(self, setup):
        """The default projector has NO idempotence rule — the extra
        condition is not derivable from MC Def 3.10 and must not be
        smuggled in."""
        from jacopy.central.algebroid import LocalityOperator

        reg, _, E, u, v, _, _, fr = setup
        P = locality_projector(E)
        assert len(P.rules()) == 2
        node = P(
            P(LocalityOperator(E.name, fr.dual().field("s"), u, v))
        )
        from jacopy.central.algebroid.engine import algebroid_engine

        eng = algebroid_engine(E, registry=reg, projectors=(P,))
        out, _steps = eng.expand(node)
        assert out == node  # stays inert

    def test_admissible_flag_adds_idempotence(self, setup):
        """With admissible=True, 𝒫(𝒫(L)) → 𝒫(L)."""
        from jacopy.central.algebroid import LocalityOperator
        from jacopy.central.algebroid.engine import algebroid_engine

        reg, _, E, u, v, _, _, fr = setup
        P = locality_projector(E, admissible=True)
        assert len(P.rules()) == 3
        inner = P(
            LocalityOperator(E.name, fr.dual().field("s"), u, v)
        )
        eng = algebroid_engine(E, registry=reg, projectors=(P,))
        out, steps = eng.expand(P(inner))
        assert out == inner
        assert any("P|_ker" in s.rule for s in steps)


class TestCourantLink:
    """Phase 4.F.2 [MC Prop 4.1, ADM Prop 3.5 ⟸]."""

    def test_courant_locality_generator(self, setup):
        """S(fu,v) = f·S(u,v) + ᵍL(Df,u,v) — the almost-Courant
        symmetric part is generated by ᵍL [MC Prop 4.1 / B eq 4.8]."""
        from jacopy.packages.metric_affine.admissible import (
            prove_courant_locality_generator,
        )

        reg, f, E, u, v, _, _, fr = setup
        assert prove_courant_locality_generator(
            E, fr, u, v, f, registry=reg
        ).steps

    def test_metric_compatible_implies_admissible(self, setup):
        """[ADM Prop 3.5 ⟸]: declared Q(∇,g) = 0 ⟹ the
        admissibility equation, via ᵍL + framed compatibility
        collection + coframe recombination with E = g⁻¹."""
        from jacopy.packages.metric_affine.admissible import (
            prove_metric_compatible_implies_admissible,
        )

        reg, _, E, u, v, _, nabla, fr = setup
        assert prove_metric_compatible_implies_admissible(
            E, nabla, fr, u, v, registry=reg
        ).steps

    def test_honest_without_compatibility(self, setup):
        """Without the compatibility declaration the admissibility
        equation must not close (ᵍL alone is not enough)."""
        from jacopy.packages.metric_affine.admissible import (
            CourantLocalityDefinition,
            _locality_sum,
        )

        reg, _, E, u, v, _, nabla, fr = setup
        engine = _engine(E, fr, reg)
        engine.register(CourantLocalityDefinition(E))
        lhs = Sum(
            _locality_sum(E, nabla, fr, u, v, "s"),
            _locality_sum(E, nabla, fr, v, u, "s"),
        )
        rhs = E.sharp(E.D(E.metric(u, v)))
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, rhs, registry=reg, engine=engine
            )

    def test_sharp_slot_substitution(self, setup):
        """MetricSharp now walks its form slot (the 4.F.2 infra fix
        enabling recombination through g⁻¹)."""
        reg, f, E, u, v, _, _, fr = setup
        node = E.sharp(fr.dual().field("s"))
        out = node.substitute_atom(
            fr.dual().field("s"), E.D(f)
        )
        assert out == E.sharp(E.D(f))


class TestAdmissibleImpliesCompatible:
    """Phase 4.F.2b [ADM Prop 3.5 ⟹] — completes the equivalence
    admissible ⟺ metric-compatible on almost-Courant algebroids."""

    def test_closes(self, setup):
        from jacopy.packages.metric_affine.admissible import (
            prove_admissible_implies_metric_compatible,
        )

        reg, _, E, u, v, _, nabla, fr = setup
        chain = prove_admissible_implies_metric_compatible(
            E, nabla, fr, u, v, registry=reg
        )
        assert len(chain.steps) == 7
        mech = [s for s in chain.steps if s.children]
        synth = [s for s in chain.steps if not s.children]
        assert len(mech) == 4 and len(synth) == 3

    def test_emetric_slot_substitution(self, setup):
        """EMetric now walks its slots (4.F.2b infra fix — Kronecker
        contraction reaches ∇_{e_s} inside metric slots)."""
        reg, _, E, u, v, _, nabla, fr = setup
        node = E.metric(nabla(fr.field("s"), u), v)
        out = node.substitute_atom(fr.field("s"), fr.field("a"))
        assert out == E.metric(nabla(fr.field("a"), u), v)

    def test_metric_indexed_sum_push(self, setup):
        """g(x, Σ_s b) → Σ_s g(x, b) with the capture guard
        (4.F.2b bilinearity extension)."""
        from jacopy.core.indexed_sum import IndexedSum
        from jacopy.central.objects.frame import FrameIndex
        from jacopy.central.algebroid.engine import algebroid_engine

        reg, _, E, u, v, _, _, fr = setup
        node = E.metric(
            u, IndexedSum(FrameIndex("s"), fr, fr.field("s"))
        )
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(node)
        assert isinstance(out, IndexedSum)
