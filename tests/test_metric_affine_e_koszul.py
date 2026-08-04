"""Metric-affine package — Phase 4.F.3: the E-Koszul formula family
[MC Prop 3.7-3.8, ADM Prop 3.12 one direction]."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid import algebroid
from jacopy.packages.metric_affine.e_connection import e_connection
from jacopy.packages.metric_affine.modified import _engine
from jacopy.packages.metric_affine.e_koszul import (
    _koszul_lhs,
    _koszul_rhs_clean,
    prove_e_generalized_koszul,
    prove_e_koszul_formula,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    E = algebroid("E", Bundle("E"), declare=("local", "regular"))
    u, v, w = E.sections("u v w")
    nabla = e_connection(E)
    fr = Frame("e", bundle=E.bundle)
    return reg, E, u, v, w, nabla, fr


class TestEGeneralizedKoszul:
    def test_identity_closes_without_declarations(self, setup):
        """The defect-block identity holds for an ARBITRARY
        E-connection, purely definitionally."""
        reg, E, u, v, w, nabla, fr = setup
        assert prove_e_generalized_koszul(
            E, nabla, fr, u, v, w, registry=reg
        ).steps


class TestEKoszulFormula:
    def test_closes_with_declarations(self, setup):
        """ᴸT = 0 + Q = 0 declared ⟹ the clean modified Koszul
        formula — E-LC ⟹ E-Koszul (with 4.F.1's TF ⟹ admissible,
        one direction of ADM Prop 3.12)."""
        reg, E, u, v, w, nabla, fr = setup
        assert prove_e_koszul_formula(
            E, nabla, fr, u, v, w, registry=reg
        ).steps

    def test_honest_without_declarations(self, setup):
        """The clean formula is FALSE for a general connection —
        must not close without the declarations."""
        reg, E, u, v, w, nabla, fr = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                _koszul_lhs(E, nabla, fr, u, v, w),
                _koszul_rhs_clean(E, u, v, w),
                registry=reg,
                engine=_engine(E, fr, reg),
            )


class TestKoszulAdmissibleImpliesLC:
    """Phase 4.F.3b [ADM Prop 3.12 ⟸] — completes the equivalence
    E-LC ⟺ admissible + E-Koszul."""

    def test_compatible_closes(self, setup):
        """Q(∇,g) = 0 — fully mechanical, NO non-degeneracy: the
        Koszul(u,v,w)+Koszul(u,w,v) combination."""
        from jacopy.packages.metric_affine.e_koszul import (
            prove_e_koszul_admissible_implies_compatible,
        )

        reg, E, u, v, w, nabla, fr = setup
        chain = prove_e_koszul_admissible_implies_compatible(
            E, nabla, fr, u, v, w, registry=reg
        )
        assert chain.steps

    def test_torsion_free_closes(self, setup):
        """g(ᴸT,w) = 0 mechanical + one labeled non-degeneracy
        step."""
        from jacopy.packages.metric_affine.e_koszul import (
            prove_e_koszul_admissible_implies_torsion_free,
        )

        reg, E, u, v, w, nabla, fr = setup
        chain = prove_e_koszul_admissible_implies_torsion_free(
            E, nabla, fr, u, v, w, registry=reg
        )
        assert len(chain.steps) == 2
        assert chain.steps[0].children  # mechanical leg
        assert not chain.steps[1].children  # labeled stripping

    def test_honest_without_admissibility(self, setup):
        """E-Koszul alone (without admissibility) must NOT give
        Q = 0."""
        from jacopy.core.expr import Integer
        from jacopy.packages.metric_affine.e_koszul import (
            EKoszulDeclaration,
            e_nonmetricity,
        )

        reg, E, u, v, w, nabla, fr = setup
        engine = _engine(E, fr, reg)
        engine.register(EKoszulDeclaration(E, nabla, fr))
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                e_nonmetricity(E, nabla, u, v, w),
                Integer(0),
                registry=reg,
                engine=engine,
            )


class TestMetricLieDerivative:
    """Phase 4.F.5 [ADM Prop 3.14]."""

    def test_closes_for_e_lc(self, setup):
        from jacopy.packages.metric_affine.e_koszul import (
            prove_metric_lie_derivative_identity,
        )

        reg, E, u, v, w, nabla, fr = setup
        assert prove_metric_lie_derivative_identity(
            E, nabla, fr, u, v, w, registry=reg
        ).steps

    def test_honest_without_declarations(self, setup):
        """The identity is FALSE for a general connection."""
        from jacopy.algebra.derivation import Act
        from jacopy.core.expr import Neg, Sum
        from jacopy.packages.metric_affine.admissible import (
            modified_bracket,
        )

        reg, E, u, v, w, nabla, fr = setup
        g = E.metric
        lhs = Sum(
            Act(E.anchor(v), g(u, w)),
            Neg(g(modified_bracket(E, nabla, fr, v, u), w)),
            Neg(g(u, modified_bracket(E, nabla, fr, v, w))),
        )
        rhs = Sum(g(nabla(u, v), w), g(u, nabla(w, v)))
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, rhs, registry=reg, engine=_engine(E, fr, reg)
            )


class TestKoszulReferenceDecomposition:
    """Phase 4.F.5 [ADM Prop 3.13, identity form]."""

    def test_closes(self, setup):
        from jacopy.packages.metric_affine.e_connection import (
            e_connection,
        )
        from jacopy.packages.metric_affine.e_koszul import (
            prove_koszul_reference_decomposition,
        )

        reg, E, u, v, w, nabla, fr = setup
        nbar = e_connection(E, "∇̄")
        assert prove_koszul_reference_decomposition(
            E, nabla, nbar, fr, u, v, w, registry=reg
        ).steps
