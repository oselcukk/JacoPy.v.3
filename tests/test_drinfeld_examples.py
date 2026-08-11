"""Drinfeld package — Phase 6.H: example/regression suites from
[2409.11973 §8]: B_n-generalized geometry, the exceptional Courant
bracket, and the Atiyah bridge to the Phase 4 Bianchi identity."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.poisson.nambu import nambu_structure
import jacopy.packages.drinfeld.examples as ex


@pytest.fixture()
def bn_setup():
    reg = PropertyRegistry()
    f, g, s = functions("fb gb sb", registry=reg)
    U, V, Y = vector_fields("U V Y")
    om, et = forms("ωb ηb", degree=1)
    return reg, f, g, s, U, V, Y, om, et


class TestBnGeneralizedGeometry:
    """(8.15)-(8.16): TM ⊕ C∞M ⊕ T*M with the SO(n+1,n) pairing."""

    def test_symmetric_part_is_d_pairing(self, bn_setup):
        reg, f, g, s, U, V, Y, om, et = bn_setup
        c1, c2, c3 = ex.prove_bn_symmetric_part(
            U, f, om, V, g, et, Y, registry=reg
        )
        assert c1.steps and c2.steps and c3.steps

    def test_right_leibniz(self, bn_setup):
        reg, f, g, s, U, V, Y, om, et = bn_setup
        c1, c2, c3 = ex.prove_bn_right_leibniz(
            U, f, om, V, g, et, s, Y, registry=reg
        )
        assert c1.steps and c2.steps and c3.steps


@pytest.fixture()
def exc_setup():
    reg = PropertyRegistry()
    (f,) = functions("fe", registry=reg)
    U, V = vector_fields("U V")
    om2, et2 = forms("Ωe2 He2", degree=2)
    om5, et5 = forms("Ωe5 He5", degree=5)
    slots2 = list(vector_fields("Ye1 Ye2"))
    slots5 = list(vector_fields("Ze1 Ze2 Ze3 Ze4 Ze5"))
    return reg, f, U, V, om2, et2, om5, et5, slots2, slots5


class TestExceptionalCourant:
    """(8.1): TM ⊕ Λ² ⊕ Λ⁵ with the M-theory cross-term η₂∧dω₂ —
    the user's X+α+β question, now a package theorem suite (5-slot
    mixed-degree wedge evaluations)."""

    def test_symmetric_part_closes_on_exceptional_pairing(
        self, exc_setup
    ):
        reg, f, U, V, om2, et2, om5, et5, s2, s5 = exc_setup
        N = nambu_structure(p=2)
        c2, c5 = ex.prove_exceptional_symmetric_part(
            N, U, om2, om5, V, et2, et5, s2, s5, registry=reg
        )
        assert c2.steps and c5.steps

    def test_right_leibniz(self, exc_setup):
        reg, f, U, V, om2, et2, om5, et5, s2, s5 = exc_setup
        N = nambu_structure(p=2)
        c2, c5 = ex.prove_exceptional_right_leibniz(
            N, U, om2, om5, V, et2, et5, f, s2, s5, registry=reg
        )
        assert c2.steps and c5.steps


class TestAtiyahBridge:
    """(8.10)-(8.14): for an Atiyah algebroid the (5.13) H-closure
    condition IS the Bianchi identity d^∇F = 0 — the concrete
    mechanical content lives in the Phase 4 metric-affine package
    (curvature F of ∇, cited VF-Jacobi instances)."""

    def test_h_closure_is_bianchi_second(self):
        from jacopy.central.objects import connection, vector_fields
        from jacopy.packages.metric_affine import (
            prove_bianchi_second,
        )

        reg = PropertyRegistry()
        (f,) = functions("fa", registry=reg)
        X, Y, Z, W = vector_fields("Xa Ya Za Wa")
        nabla = connection("∇")
        chain, used = prove_bianchi_second(
            nabla, X, Y, Z, W, f, registry=reg
        )
        assert chain.steps and used


class TestMultivectorInterior:
    """6.H.2: the partial-contraction primitive ι_P: Λ^q → Λ^{q−p}
    — the missing piece of the exceptional Ψ_Π twist scenario."""

    def test_degree_law(self):
        from jacopy.algebra.derivation import Act, degree_of
        from jacopy.core.symbolic_degree import Degree
        from jacopy.central.objects.multivector_interior import (
            MultivectorInterior,
        )

        (om5,) = forms("Ωm5", degree=5)
        N3 = nambu_structure(p=2)
        node = Act(MultivectorInterior(N3.pi), om5)
        assert degree_of(node, None) == Degree.const(2)

    def test_linearity_rule(self):
        from jacopy.algebra.derivation import Act
        from jacopy.core.expr import Product, Sum
        from jacopy.central.objects.multivector_interior import (
            MultivectorInterior,
            MultivectorInteriorLinearityDefinition,
        )

        reg = PropertyRegistry()
        (f,) = functions("fm", registry=reg)
        (om5,) = forms("Ωm5", degree=5)
        N3 = nambu_structure(p=2)
        rule = MultivectorInteriorLinearityDefinition(reg)
        node = Act(
            MultivectorInterior(Product(f, N3.pi)), om5
        )
        assert rule.matches(node)
        assert rule.rewrite(node) == Product(
            f, Act(MultivectorInterior(N3.pi), om5)
        )

    def test_decomposable_eval_consistency(self):
        """ι_{X∧Y∧Z}ω₅ evaluated on 2 slots == ω₅(X,Y,Z,·,·) —
        the (4.13) convention closes mechanically."""
        from jacopy.algebra.derivation import Act
        from jacopy.core.expr import Integer, Neg, Sum
        from jacopy.core.multi_eval import MultiEval
        from jacopy.core.wedge import Wedge
        from jacopy.proof.strategies import ExpandAndSimplify
        from jacopy.central.objects.multivector_interior import (
            MultivectorInterior,
            MultivectorInteriorDecomposableDefinition,
            MultivectorInteriorLinearityDefinition,
        )
        from jacopy.packages.drinfeld.tilde_calculus import (
            _tilde_engine,
        )

        reg = PropertyRegistry()
        X, Y, Z, Y1, Y2 = vector_fields("Xm Ym Zm Ym1 Ym2")
        (om5,) = forms("Ωm5", degree=5)
        N3 = nambu_structure(p=2)
        eng = _tilde_engine(N3, reg, declare_fi=False)
        eng.register(MultivectorInteriorLinearityDefinition(reg))
        eng.register(MultivectorInteriorDecomposableDefinition())
        lhs = MultiEval(
            Act(MultivectorInterior(Wedge(X, Y, Z)), om5),
            Y1, Y2, alternating=True, slot_kind="vector",
        )
        rhs = MultiEval(
            om5, X, Y, Z, Y1, Y2,
            alternating=True, slot_kind="vector",
        )
        chain = ExpandAndSimplify().prove(
            Sum(lhs, Neg(rhs)), Integer(0),
            registry=reg, engine=eng, max_steps=20000,
        )
        assert chain.steps

    def test_boxtimes_buildable_and_typed(self):
        """(Π₃⊛Π₃)(ω₅) = ½Π₃(ι_{Π₃}ω₅) — the (4.14) bundle map now
        exists as an expression (atomic Π₃: inert, honest)."""
        from jacopy.core.expr import Product, Rational

        (om5,) = forms("Ωm5", degree=5)
        N3 = nambu_structure(p=2)
        bx = ex.boxtimes(N3, om5)
        assert isinstance(bx, Product)
        assert bx.children[0] == Rational(1, 2)


class TestExceptionalDecompositionReadings:
    """(8.2)/(8.3): the SAME cross-term reads as an H-twist or an
    R-twist depending on the decomposition; the twisted linearity
    structure (5.6)-(5.7) is mechanical."""

    def test_h_second_entry_linear(self):
        reg = PropertyRegistry()
        (f,) = functions("fx", registry=reg)
        U, V = vector_fields("Ux Vx")
        om2, et2 = forms("Ωx Hx", degree=2)
        slots5 = list(vector_fields("Zx1 Zx2 Zx3 Zx4 Zx5"))
        chain = ex.prove_exceptional_h_second_entry_linear(
            U, om2, V, et2, f, slots5, registry=reg
        )
        assert chain.steps

    def test_h_first_entry_symbol(self):
        """H(f·e₁,e₂) − f·H(e₁,e₂) = −η₂ ∧ df ∧ ω₂ — the exact
        (5.7) symbol."""
        reg = PropertyRegistry()
        (f,) = functions("fx", registry=reg)
        U, V = vector_fields("Ux Vx")
        om2, et2 = forms("Ωx Hx", degree=2)
        slots5 = list(vector_fields("Zx1 Zx2 Zx3 Zx4 Zx5"))
        chain = ex.prove_exceptional_h_first_entry_symbol(
            U, om2, V, et2, f, slots5, registry=reg
        )
        assert chain.steps

    def test_r_reading_is_same_cross_term(self):
        """(8.3): for A = TM⊕Λ⁵, Z = Λ² the cross-term IS the
        R-twist — structural identification."""
        from jacopy.core.expr import Neg
        from jacopy.core.wedge import Wedge
        from jacopy.central.tangent.exterior import d

        om2, et2 = forms("Ωx Hx", degree=2)
        assert ex.exceptional_h_reading(
            None, om2, None, et2
        ) == Neg(Wedge(et2, d(om2)))
