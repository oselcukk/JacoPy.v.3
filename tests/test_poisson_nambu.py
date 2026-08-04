"""Poisson package — Phase 5.E.3: Nambu-Poisson structures and the
higher Koszul bracket on p-forms (PDF 12r first half; drinfeld paper
§6 eq (6.8))."""

import pytest

from jacopy.core.expr import Integer, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.algebra.derivation import Act
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.tangent.exterior import d
from jacopy.packages.poisson import poisson_structure
from jacopy.packages.poisson.nambu import (
    NambuPoissonStructure,
    nambu_engine,
    nambu_koszul_bracket,
    nambu_structure,
    prove_dorfman_form_reduces_to_koszul_at_p1,
    prove_nambu_right_leibniz,
    prove_nambu_symmetric_part_exact,
    _ev,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g, h = functions("f g h", registry=reg)
    Y = vector_fields("Y1 Y2 Y3")
    return reg, f, g, h, Y


class TestStructure:
    def test_order_guard(self):
        from jacopy.central.objects.multivector import PVector

        with pytest.raises(ValueError):
            NambuPoissonStructure(PVector("v", degree=1))

    def test_sharp_action_on_decomposables(self, setup):
        """(Π(df∧dg))(h) → Π(df, dg, dh) — the Nambu bracket."""
        reg, f, g, h, _ = setup
        N = nambu_structure(p=2)
        node = Act(N.sharp_vf(Wedge(d(f), d(g))), h)
        out, steps = nambu_engine(N, registry=reg).expand(node)
        assert out == MultiEval(
            N.pi,
            d(f),
            d(g),
            d(h),
            alternating=True,
            slot_kind="covector",
        )
        assert steps

    def test_sharp_inert_on_opaque_forms(self, setup):
        """An atomic p-form cannot decompose — (Πω)(f) stays inert
        (honest)."""
        reg, f, _, _, _ = setup
        N = nambu_structure(p=2)
        (om,) = forms("ω", degree=2)
        node = Act(N.sharp_vf(om), f)
        out, _ = nambu_engine(N, registry=reg).expand(node)
        assert out == node


class TestBracketTheorems:
    @pytest.mark.parametrize("p", [2, 3])
    def test_right_leibniz(self, setup, p):
        reg, f, _, _, Y = setup
        N = nambu_structure(p=p)
        (om,) = forms(f"ω{p}", degree=p)
        (et,) = forms(f"η{p}", degree=p)
        assert prove_nambu_right_leibniz(
            N, om, et, f, Y[:p], registry=reg
        ).steps

    @pytest.mark.parametrize("p", [2, 3])
    def test_symmetric_part_exact(self, setup, p):
        """[ω,η] + [η,ω] = d(ι_{Πω}η + ι_{Πη}ω) — Leibniz, not Lie;
        the Dorfman pattern."""
        reg, _, _, _, Y = setup
        N = nambu_structure(p=p)
        (om,) = forms(f"ω{p}", degree=p)
        (et,) = forms(f"η{p}", degree=p)
        assert prove_nambu_symmetric_part_exact(
            N, om, et, Y[:p], registry=reg
        ).steps

    @pytest.mark.parametrize("p", [2, 3])
    def test_not_antisymmetric(self, setup, p):
        """[ω,η] + [η,ω] = 0 must FAIL for p ≥ 2 (spec 12r: the
        bracket is a LEIBNIZ algebroid bracket)."""
        reg, _, _, _, Y = setup
        N = nambu_structure(p=p)
        (om,) = forms(f"ω{p}", degree=p)
        (et,) = forms(f"η{p}", degree=p)
        lhs = _ev(
            Sum(
                nambu_koszul_bracket(N, om, et),
                nambu_koszul_bracket(N, et, om),
            ),
            Y[:p],
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                Integer(0),
                registry=reg,
                engine=nambu_engine(N, registry=reg),
            )

    def test_p1_reduces_to_koszul(self, setup):
        """p = 1 consistency: the Dorfman-type formula IS the 5.B
        Koszul bracket (via the magic formula)."""
        reg, _, _, _, Y = setup
        al, be = forms("α β", degree=1)
        P = poisson_structure()
        assert prove_dorfman_form_reduces_to_koszul_at_p1(
            P, al, be, Y[0], registry=reg
        ).steps


class TestFundamentalIdentity:
    """Phase 5.E.4: the DECLARED fundamental identity and the
    Leibniz-Jacobi of the higher Koszul bracket on exact
    generators."""

    def test_leibniz_jacobi_closes_under_fi(self, setup):
        """[ω,[η,μ]] = [[ω,η],μ] + [η,[ω,μ]] on exact wedges at
        p = 2, under the declared FI (1082 steps; p = 3 also closes
        — 5683 steps, kept out of the default suite for runtime)."""
        from jacopy.packages.poisson.nambu import (
            prove_nambu_leibniz_jacobi_on_exacts,
        )

        reg, _, _, _, Y = setup
        fs = functions(
            "F1 F2 G1 G2 H1 H2", registry=reg
        )
        N = nambu_structure(p=2)
        chain = prove_nambu_leibniz_jacobi_on_exacts(
            N,
            fs[:2],
            fs[2:4],
            fs[4:6],
            Y[:2],
            registry=reg,
        )
        assert chain.steps

    def test_honest_without_fi(self, setup):
        """Without the FI declaration the Jacobi must NOT close —
        the identity is the Nambu-Poisson hypothesis itself."""
        from jacopy.packages.poisson.nambu import (
            prove_nambu_leibniz_jacobi_on_exacts,
        )

        reg, _, _, _, Y = setup
        fs = functions("F1 F2 G1 G2 H1 H2", registry=reg)
        N = nambu_structure(p=2)
        with pytest.raises(ProofFailure):
            prove_nambu_leibniz_jacobi_on_exacts(
                N,
                fs[:2],
                fs[2:4],
                fs[4:6],
                Y[:2],
                registry=reg,
                declare_fi=False,
            )

    def test_wedge_eval_n_factor(self, setup):
        """The n-factor wedge evaluation (the closed deferral):
        3-factor determinant expands to 6 signed products."""
        from jacopy.core.expr import Neg
        from jacopy.central.calculus.indexed_rules import (
            WedgeEvalDefinition,
        )
        from jacopy.central.objects import forms

        reg, _, _, _, Y = setup
        a1, a2, a3 = forms("a1 a2 a3", degree=1)
        node = MultiEval(
            Wedge(a1, a2, a3),
            *Y[:3],
            alternating=True,
            slot_kind="vector",
        )
        rule = WedgeEvalDefinition(reg)
        assert rule.matches(node)
        out = rule.rewrite(node)
        assert len(out.children) == 6
