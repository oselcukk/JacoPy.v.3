"""Poisson package — Phase 5.B: the Koszul bracket on 1-forms.

[α,β]_π := L_{π♯α}β − L_{π♯β}α − d(π(α,β)); antisymmetry and the
exact-forms identity [df,dg]_π = d{f,g} close mechanically."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.tangent.exterior import d
from jacopy.packages.poisson.core import poisson_structure
from jacopy.packages.poisson.koszul import (
    KoszulBracket,
    KoszulBracketDefinition,
    SharpVFLinearityDefinition,
    _engine,
    koszul_bracket,
    prove_koszul_antisymmetry,
    prove_koszul_on_exact_forms,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    (Y,) = vector_fields("Y")
    alpha, beta = forms("α β", degree=1)
    P = poisson_structure()
    return reg, f, g, Y, alpha, beta, P


class TestKoszulNode:
    def test_expansion(self, setup):
        reg, _, _, _, alpha, beta, P = setup
        eng = _engine((P,), reg)
        out, steps = eng.expand(koszul_bracket(P, alpha, beta))
        assert any("Koszul bracket" in s.rule for s in steps)

    def test_degree_is_one(self, setup):
        _, _, _, _, alpha, beta, P = setup
        assert str(koszul_bracket(P, alpha, beta).degree) == "1"

    def test_scoped_to_structure(self, setup):
        reg, _, _, _, alpha, beta, P = setup
        Q = poisson_structure("σ")
        eng = _engine((P,), reg)
        node = koszul_bracket(Q, alpha, beta)
        out, steps = eng.expand(node)
        assert out == node and not steps


class TestSharpLinearity:
    def test_scalar_and_sum(self, setup):
        reg, f, _, _, alpha, beta, P = setup
        eng = _engine((P,), reg)
        out, _ = eng.expand(
            P.sharp_vf(Sum(Product(f, alpha), beta))
        )
        assert out == Sum(
            Product(f, P.sharp_vf(alpha)), P.sharp_vf(beta)
        )


class TestKoszulTheorems:
    def test_antisymmetry(self, setup):
        reg, _, _, _, alpha, beta, P = setup
        assert prove_koszul_antisymmetry(
            P, alpha, beta, registry=reg
        ).steps

    def test_self_bracket_vanishes(self, setup):
        reg, _, _, _, alpha, _, P = setup
        chain = ExpandAndSimplify().prove(
            koszul_bracket(P, alpha, alpha),
            Integer(0),
            registry=reg,
            engine=_engine((P,), reg),
        )
        assert chain.steps

    def test_exact_forms_identity(self, setup):
        """[df, dg]_π = d{f,g} — the Koszul bracket on exact forms IS
        the Poisson bracket's differential."""
        reg, f, g, Y, _, _, P = setup
        assert prove_koszul_on_exact_forms(
            P, f, g, Y, registry=reg
        ).steps

    def test_wrong_sign_exact_identity_fails(self, setup):
        reg, f, g, Y, _, _, P = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Pairing(koszul_bracket(P, d(f), d(g)), Y),
                Neg(Pairing(d(P.bracket(f, g)), Y)),
                registry=reg,
                engine=_engine((P,), reg),
            )

    def test_arbitrary_names(self, setup):
        reg, *_ = setup
        f, g = functions("H K", registry=reg)
        (Z,) = vector_fields("ζ")
        P = poisson_structure("Π")
        assert prove_koszul_on_exact_forms(
            P, f, g, Z, registry=reg
        ).steps


class TestDegreeGuard:
    """The 5.B bracket is the DEGREE-1 (cotangent algebroid) bracket;
    higher degrees are Phase 5.E (Koszul-Brylinski / Nambu-Poisson) —
    a 2-form must be refused, not silently mis-expanded."""

    def test_two_form_refused(self, setup):
        reg, _, _, _, alpha, _, P = setup
        (omega,) = forms("ω", degree=2)
        with pytest.raises(ValueError):
            koszul_bracket(P, omega, alpha)
        with pytest.raises(ValueError):
            koszul_bracket(P, alpha, omega)

    def test_raw_higher_degree_node_stays_inert(self, setup):
        from jacopy.packages.poisson.koszul import KoszulBracket

        reg, _, _, _, alpha, _, P = setup
        (omega,) = forms("ω", degree=2)
        node = KoszulBracket(P.pi, omega, alpha)  # raw ctor, no guard
        eng = _engine((P,), reg)
        out, steps = eng.expand(node)
        assert out == node and not steps


class TestTwoScalarLeibniz:
    """The full two-scalar Leibniz identity (5.D.2 gap, closed):

        [gα, fβ]_π = fg[α,β]_π + g(π♯α)(f)β − f(π♯β)(g)α.

    Blocked before the product_rule scalar-coefficient fix: the
    composition unfold Leibniz-split scalar layers, doubling four
    terms on the left-hand side."""

    def test_identity_closes(self, setup):
        from jacopy.packages.poisson.core import SharpVF
        from jacopy.packages.poisson.showcase import showcase_engine

        reg, f, g, Y, alpha, beta, P = setup
        lhs = Pairing(
            koszul_bracket(
                P, Product(g, alpha), Product(f, beta)
            ),
            Y,
        )
        rhs = Pairing(
            Sum(
                Product(f, g, koszul_bracket(P, alpha, beta)),
                Product(g, Act(SharpVF(P.pi, alpha), f), beta),
                Neg(Product(f, Act(SharpVF(P.pi, beta), g), alpha)),
            ),
            Y,
        )
        chain = ExpandAndSimplify().prove(
            lhs,
            rhs,
            registry=reg,
            engine=showcase_engine(P, registry=reg),
        )
        assert chain.steps

    def test_wrong_sign_fails(self, setup):
        """Honesty check: flipping the correction-term sign must NOT
        prove."""
        from jacopy.packages.poisson.core import SharpVF
        from jacopy.packages.poisson.showcase import showcase_engine

        reg, f, g, Y, alpha, beta, P = setup
        lhs = Pairing(
            koszul_bracket(
                P, Product(g, alpha), Product(f, beta)
            ),
            Y,
        )
        bad = Pairing(
            Sum(
                Product(f, g, koszul_bracket(P, alpha, beta)),
                Product(g, Act(SharpVF(P.pi, alpha), f), beta),
                Product(f, Act(SharpVF(P.pi, beta), g), alpha),
            ),
            Y,
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                bad,
                registry=reg,
                engine=showcase_engine(P, registry=reg),
            )
