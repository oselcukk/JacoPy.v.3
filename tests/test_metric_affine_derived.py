"""Metric-affine package — Phase 4.F.6: the derived algebroid and
the two exterior derivatives d(∇)/d̂(∇) [ADM Defs 3.14-3.16,
eq (3.43)]."""

import pytest

from jacopy.core.expr import Integer
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle, functions
from jacopy.central.objects.frame import Frame
from jacopy.central.algebroid import algebroid, locality_projector
from jacopy.packages.metric_affine.e_connection import e_connection
from jacopy.packages.metric_affine.derived import (
    d_derived,
    derived_algebroid,
    derived_engine,
    prove_derived_magic_on_functions,
    prove_projected_d_squared_on_functions,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid(
        "E",
        Bundle("E"),
        declare=("local", "regular", "anchor-morphism"),
    )
    u, v = E.sections("u v")
    nabla = e_connection(E)
    fr = Frame("e", bundle=E.bundle)
    P = locality_projector(E)
    return reg, f, E, u, v, nabla, fr, P


class TestDerivedMagic:
    def test_unprojected(self, setup):
        """ℒ^∇_u f = (d(∇)ι_u + ι_u d(∇))f — declaration-free
        (3.F generic magic through the bridges)."""
        reg, f, E, u, _, nabla, fr, _ = setup
        assert prove_derived_magic_on_functions(
            E, nabla, fr, u, f, registry=reg
        ).steps

    def test_projected(self, setup):
        reg, f, E, u, _, nabla, fr, P = setup
        assert prove_derived_magic_on_functions(
            E, nabla, fr, u, f, registry=reg, projector=P
        ).steps


class TestProjectedDSquared:
    def test_closes(self, setup):
        """[ADM eq (3.43)]: d̂(∇)²f = 0 under anchor-morphism +
        the projector kernel rule."""
        reg, f, E, u, v, nabla, fr, P = setup
        assert prove_projected_d_squared_on_functions(
            E, nabla, fr, P, f, u, v, registry=reg
        ).steps

    def test_honest_without_anchor_morphism(self, setup):
        """Without the base anchor-morphism declaration the
        function-level d̂² must not close."""
        reg, f, _, _, _, _, _, _ = setup
        E2 = algebroid("F", Bundle("F"), declare=("local", "regular"))
        u2, v2 = E2.sections("u v")
        nabla2 = e_connection(E2)
        fr2 = Frame("c", bundle=E2.bundle)
        P2 = locality_projector(E2)
        with pytest.raises(ProofFailure):
            prove_projected_d_squared_on_functions(
                E2, nabla2, fr2, P2, f, u2, v2, registry=reg
            )

    def test_unprojected_d_squared_honestly_fails(self, setup):
        """d(∇)²f ≠ 0 in general — ρ(Λ) does not vanish; exactly why
        ADM introduces the PROJECTED d̂."""
        reg, f, E, u, v, nabla, fr, _ = setup
        derived = derived_algebroid(E, projected=False)
        node = MultiEval(
            d_derived(derived, d_derived(derived, f)),
            u,
            v,
            alternating=True,
            slot_kind="vector",
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=reg,
                engine=derived_engine(
                    E, derived, nabla, fr, registry=reg
                ),
            )


class TestDerivedMagicHigherForms:
    """User audit 2026-07-31: operators accepting general p-forms
    must be exercised beyond p = 1 — magic at p = 1, 2 for both d's,
    and the associator honest-fail at the 1-form level."""

    def test_magic_one_form_projected(self, setup):
        from jacopy.central.objects import forms
        from jacopy.packages.metric_affine.derived import (
            prove_derived_magic_on_forms,
        )

        reg, _, E, u, v, nabla, fr, P = setup
        (alpha,) = forms("α", degree=1, bundle=E.bundle)
        assert prove_derived_magic_on_forms(
            E, nabla, fr, u, alpha, (v,), registry=reg, projector=P
        ).steps

    def test_magic_two_form_both_variants(self, setup):
        from jacopy.central.objects import forms
        from jacopy.packages.metric_affine.derived import (
            prove_derived_magic_on_forms,
        )

        reg, _, E, u, v, nabla, fr, P = setup
        (w,) = E.sections("w")
        (omega,) = forms("ω", degree=2, bundle=E.bundle)
        assert prove_derived_magic_on_forms(
            E, nabla, fr, u, omega, (v, w), registry=reg
        ).steps
        assert prove_derived_magic_on_forms(
            E, nabla, fr, u, omega, (v, w), registry=reg, projector=P
        ).steps

    def test_base_magic_two_form(self, setup):
        """Legacy-coverage gap: the BASE 3.F magic also holds at
        p = 2 (was only tested on functions and 1-forms)."""
        from jacopy.core.expr import Sum
        from jacopy.core.multi_eval import MultiEval
        from jacopy.algebra.derivation import Act
        from jacopy.central.objects import forms
        from jacopy.central.objects.interior import Interior
        from jacopy.central.algebroid.calculus import (
            _engine as calc_engine,
            algebroid_calculus,
            d_E,
            lie_E,
        )

        reg, _, E, u, v, nabla, fr, _ = setup
        (w,) = E.sections("w")
        (omega,) = forms("ω", degree=2, bundle=E.bundle)
        iota = Interior(u)
        calc = algebroid_calculus(E)
        lhs = MultiEval(
            Act(lie_E(E, u), omega),
            v,
            w,
            alternating=True,
            slot_kind="vector",
        )
        rhs = Sum(
            MultiEval(
                Act(calc.d, Act(iota, omega)),
                v,
                w,
                alternating=True,
                slot_kind="vector",
            ),
            MultiEval(
                Act(iota, d_E(E, omega)),
                v,
                w,
                alternating=True,
                slot_kind="vector",
            ),
        )
        assert (
            ExpandAndSimplify()
            .prove(lhs, rhs, registry=reg, engine=calc_engine(E, reg))
            .steps
        )

    def test_d_hat_squared_on_one_form_honestly_fails(self, setup):
        """[ADM 3.44]: d̂² on a 1-form carries the ASSOCIATOR
        obstruction of the projected modified bracket — must not
        close."""
        from jacopy.central.objects import forms
        from jacopy.packages.metric_affine.derived import (
            d_derived,
            derived_algebroid,
            derived_engine,
        )

        reg, _, E, u, v, nabla, fr, P = setup
        (w,) = E.sections("w")
        (alpha,) = forms("α", degree=1, bundle=E.bundle)
        D = derived_algebroid(E, projected=True)
        node = MultiEval(
            d_derived(D, d_derived(D, alpha)),
            u,
            v,
            w,
            alternating=True,
            slot_kind="vector",
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=reg,
                engine=derived_engine(
                    E, D, nabla, fr, registry=reg, projector=P
                ),
            )


class TestAssociatorIdentity:
    """Phase 4.F.6b [ADM 3.44-45]: d̂² on 1-forms = the associator
    of b̂, as a positive identity."""

    def test_closes(self, setup):
        from jacopy.central.objects import forms
        from jacopy.packages.metric_affine.derived import (
            prove_projected_d_squared_associator,
        )

        reg, _, E, u, v, nabla, fr, P = setup
        (w,) = E.sections("w")
        (alpha,) = forms("α", degree=1, bundle=E.bundle)
        assert prove_projected_d_squared_associator(
            E, nabla, fr, P, alpha, u, v, w, registry=reg
        ).steps


class TestFundamentalTheoremAbstract:
    """Phase 4.F.6b [MC Thm 3.2 abstract half + MC 4.2 fragment]."""

    def test_koszul_unique_under_null_locality(self, setup):
        from jacopy.packages.metric_affine.e_koszul import (
            prove_e_koszul_unique_under_null_locality,
        )

        reg, _, E, u, v, nabla, fr, _ = setup
        (w,) = E.sections("w")
        n2 = e_connection(E, "∇²")
        chain = prove_e_koszul_unique_under_null_locality(
            E, nabla, n2, fr, u, v, w, registry=reg
        )
        assert chain.steps[0].children  # mechanical leg
        assert not chain.steps[1].children  # labeled stripping

    def test_projectors_agree_on_coboundaries(self, setup):
        from jacopy.packages.metric_affine.e_koszul import (
            prove_projectors_agree_on_coboundary_values,
        )

        reg, f, E, u, v, _, fr, _ = setup
        assert prove_projectors_agree_on_coboundary_values(
            E, fr, f, u, v, registry=reg
        ).steps
