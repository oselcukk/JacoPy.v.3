"""Metric-affine package — Phase 4.F.4b: E-Bianchi I
[ADM Prop 3.7]."""

import pytest

from jacopy.core.expr import Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid import algebroid, locality_projector
from jacopy.packages.metric_affine.e_connection import e_connection
from jacopy.packages.metric_affine.modified import _engine
from jacopy.packages.metric_affine.e_bianchi import (
    _cyc,
    covariant_modified_torsion_derivative,
    prove_e_bianchi_first_identity,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    E = algebroid("E", Bundle("E"), declare=("local", "regular"))
    u, v, w = E.sections("u v w")
    nabla = e_connection(E)
    fr = Frame("e", bundle=E.bundle)
    P = locality_projector(E)
    return reg, E, u, v, w, nabla, fr, P


class TestEBianchiFirst:
    def test_identity_form_no_declarations(self, setup):
        """The full identity form closes for an ARBITRARY
        connection, purely definitionally."""
        reg, E, u, v, w, nabla, fr, P = setup
        assert prove_e_bianchi_first_identity(
            E, nabla, fr, P, u, v, w, registry=reg
        ).steps

    def test_admissible_specialization(self, setup):
        """[ADM Prop 3.7 proper]: under the admissibility
        declaration the anti-symmetry-defect block drops."""
        reg, E, u, v, w, nabla, fr, P = setup
        assert prove_e_bianchi_first_identity(
            E, nabla, fr, P, u, v, w, registry=reg, admissible=True
        ).steps

    def test_admissible_form_honest_without_declaration(self, setup):
        """The shortened (admissible) right-hand side must NOT close
        for a general connection."""
        from jacopy.packages.metric_affine.admissible import (
            modified_bracket,
        )
        from jacopy.packages.metric_affine.modified import (
            generalized_ricci_anomaly,
            modified_torsion,
            projected_curvature,
        )

        reg, E, u, v, w, nabla, fr, P = setup

        def LhatR(a, b_, c):
            return projected_curvature(E, nabla, fr, P, a, b_, c)

        def nabla_LT(a, b_, c):
            return covariant_modified_torsion_derivative(
                E, nabla, fr, a, b_, c
            )

        def LT_of_LT(a, b_, c):
            return modified_torsion(
                E, nabla, fr, modified_torsion(E, nabla, fr, a, b_), c
            )

        def anomaly(a, b_, c):
            return generalized_ricci_anomaly(
                E, nabla, fr, P, a, b_, c
            )

        def b_of_b(a, b_, c):
            return modified_bracket(
                E, nabla, fr, modified_bracket(E, nabla, fr, a, b_), c
            )

        lhs = _cyc(LhatR, u, v, w)
        rhs = Sum(
            _cyc(nabla_LT, u, v, w),
            _cyc(LT_of_LT, u, v, w),
            Neg(_cyc(anomaly, u, v, w)),
            Neg(_cyc(b_of_b, u, v, w)),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                rhs,
                registry=reg,
                engine=_engine(E, fr, reg, projectors=(P,)),
            )


class TestEBianchiSecond:
    """Phase 4.F.4c: E-Bianchi II identity form."""

    def test_identity_closes_without_declarations(self, setup):
        from jacopy.packages.metric_affine.e_bianchi import (
            prove_e_bianchi_second_identity,
        )

        reg, E, u, v, w, nabla, fr, P = setup
        (x,) = E.sections("x")
        assert prove_e_bianchi_second_identity(
            E, nabla, fr, P, u, v, w, x, registry=reg
        ).steps

    def test_anomaly_blocks_needed(self, setup):
        """Dropping the (1−𝒫)Λ composition blocks must break the
        identity."""
        from jacopy.algebra.derivation import Act
        from jacopy.central.objects.connection import CovariantOp
        from jacopy.packages.metric_affine.e_bianchi import (
            _cyc,
            _lam,
            _lam_hat,
            projected_modified_bracket,
        )
        from jacopy.packages.metric_affine.modified import (
            modified_torsion,
            projected_curvature,
        )

        reg, E, u, v, w, nabla, fr, P = setup
        (x,) = E.sections("x")

        def cov(direction, arg):
            return Act(
                CovariantOp(
                    nabla.name, direction, bundle=nabla.bundle
                ),
                arg,
            )

        def R(a, b, c):
            return projected_curvature(E, nabla, fr, P, a, b, c)

        def LT(a, b):
            return modified_torsion(E, nabla, fr, a, b)

        def cov_R_deriv(a, b, c, d):
            return Sum(
                nabla(a, R(b, c, d)),
                Neg(R(nabla(a, b), c, d)),
                Neg(R(b, nabla(a, c), d)),
                Neg(R(b, c, nabla(a, d))),
            )

        def piece(a, b, c):
            return Sum(cov_R_deriv(a, b, c, x), R(LT(a, b), c, x))

        def bhat(p, q):
            return projected_modified_bracket(E, nabla, fr, P, p, q)

        lhs = _cyc(piece, u, v, w)
        rhs_truncated = Sum(
            _cyc(
                lambda a, b, c: cov(bhat(E.bracket(a, b), c), x),
                u,
                v,
                w,
            ),
            _cyc(
                lambda a, b, c: cov(bhat(a, nabla(c, b)), x), u, v, w
            ),
            _cyc(
                lambda a, b, c: cov(bhat(nabla(a, c), b), x), u, v, w
            ),
            Neg(
                _cyc(
                    lambda a, b, c: cov(
                        E.bracket(_lam(E, nabla, fr, a, b), c), x
                    ),
                    u,
                    v,
                    w,
                )
            ),
            _cyc(
                lambda a, b, c: cov(
                    _lam_hat(
                        E, nabla, fr, P, _lam(E, nabla, fr, a, b), c
                    ),
                    x,
                ),
                u,
                v,
                w,
            ),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                rhs_truncated,
                registry=reg,
                engine=_engine(E, fr, reg, projectors=(P,)),
            )
