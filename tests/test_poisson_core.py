"""Poisson package — Phase 5.A: the Poisson core.

{f,g} := π(df,dg), X_f(g) := {f,g}, ⟨β,π♯α⟩ := π(α,β); [π,π]_SN = 0
is the DECLARED structure axiom (opt-in); the C∞-structure of the
bracket is proved, never assumed."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import functions
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.schouten import sn_bracket
from jacopy.packages.poisson.core import (
    poisson_engine,
    poisson_structure,
    prove_hamiltonian_is_sharp,
    prove_poisson_antisymmetry,
    prove_poisson_constant,
    prove_poisson_leibniz,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g, h = functions("f g h", registry=reg)
    P = poisson_structure()
    return reg, f, g, h, P


class TestBracketStructure:
    def test_antisymmetry(self, setup):
        reg, f, g, _, P = setup
        assert prove_poisson_antisymmetry(P, f, g, registry=reg).steps

    def test_leibniz(self, setup):
        reg, f, g, h, P = setup
        assert prove_poisson_leibniz(P, f, g, h, registry=reg).steps

    def test_constant(self, setup):
        reg, f, _, _, P = setup
        assert prove_poisson_constant(
            P, f, Integer(5), registry=reg
        ).steps

    def test_self_bracket_vanishes(self, setup):
        """{f, f} = 0 — the alternating evaluation kills the
        diagonal."""
        reg, f, _, _, P = setup
        chain = ExpandAndSimplify().prove(
            P.bracket(f, f),
            Integer(0),
            registry=reg,
            engine=poisson_engine(registry=reg, structures=(P,)),
        )
        assert chain.steps

    def test_false_leibniz_fails(self, setup):
        """Dropping one Leibniz term must fail."""
        reg, f, g, h, P = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                P.bracket(f, Product(g, h)),
                Product(g, P.bracket(f, h)),
                registry=reg,
                engine=poisson_engine(registry=reg, structures=(P,)),
            )


class TestHamiltonianAndSharp:
    def test_hamiltonian_action(self, setup):
        reg, f, g, _, P = setup
        from jacopy.core.multi_eval import MultiEval

        eng = poisson_engine(registry=reg, structures=(P,))
        out, steps = eng.expand(Act(P.hamiltonian(f), g))
        # the fixpoint carries through the bracket definition too
        assert out == MultiEval(
            P.pi, d(f), d(g), alternating=True, slot_kind="covector"
        )
        assert any("Hamiltonian action" in s.rule for s in steps)
        assert any("Poisson bracket" in s.rule for s in steps)

    def test_sharp_meets_hamiltonian(self, setup):
        reg, f, g, _, P = setup
        assert prove_hamiltonian_is_sharp(P, f, g, registry=reg).steps

    def test_sharp_is_a_derivation_not_a_scalar(self, setup):
        """The 5.A hazard: π♯(α) has form-degree 0 but is a VECTOR —
        the SharpVF atom keeps the scalar-action heuristics away."""
        from jacopy.central.calculus.scalars import is_scalar_function

        reg, f, _, _, P = setup
        assert not is_scalar_function(P.sharp_vf(d(f)), reg)

    def test_hamiltonian_antisymmetry_via_action(self, setup):
        """X_f(g) = −X_g(f) — the bracket antisymmetry seen through
        the Hamiltonian actions."""
        reg, f, g, _, P = setup
        chain = ExpandAndSimplify().prove(
            Act(P.hamiltonian(f), g),
            Neg(Act(P.hamiltonian(g), f)),
            registry=reg,
            engine=poisson_engine(registry=reg, structures=(P,)),
        )
        assert chain.steps


class TestPoissonDeclaration:
    def test_sn_vanishes_when_declared(self, setup):
        reg, _, _, _, P = setup
        eng = poisson_engine(registry=reg, structures=(P,))
        chain = ExpandAndSimplify().prove(
            sn_bracket(P.pi, P.pi),
            Integer(0),
            registry=reg,
            engine=eng,
        )
        assert any("Poisson (" in s.rule for s in chain.steps)

    def test_honest_without_declaration(self, setup):
        reg, _, _, _, P = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                sn_bracket(P.pi, P.pi),
                Integer(0),
                registry=reg,
                engine=poisson_engine(registry=reg),
            )

    def test_scoped_to_structure(self, setup):
        """P's declaration must not kill another bivector's [σ,σ]."""
        reg, _, _, _, P = setup
        Q = poisson_structure("σ")
        eng = poisson_engine(registry=reg, structures=(P,))
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                sn_bracket(Q.pi, Q.pi),
                Integer(0),
                registry=reg,
                engine=eng,
            )


class TestConstruction:
    def test_requires_bivector(self, setup):
        from jacopy.central.objects.multivector import PVector
        from jacopy.packages.poisson.core import PoissonStructure

        with pytest.raises(ValueError):
            PoissonStructure(PVector("τ", degree=3))

    def test_arbitrary_names(self, setup):
        reg, *_ = setup
        f, g = functions("Kedi Fare", registry=reg)
        P = poisson_structure("Π")
        assert prove_poisson_antisymmetry(P, f, g, registry=reg).steps
