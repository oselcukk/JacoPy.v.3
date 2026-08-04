"""Poisson package — Phase 5.E.1: the symplectic special case.

Midterm Q2 (a)-(b): ``ι_{X_f}ω = df`` canonical, ``dω = 0`` declared,
``π = ω⁻¹`` a declared bridge; flow invariance + the bracket chain."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.packages.poisson import poisson_structure
from jacopy.packages.poisson.symplectic import (
    SymplecticStructure,
    prove_bracket_via_double_iota,
    prove_bracket_via_iota,
    prove_bracket_via_omega,
    prove_hamiltonian_flow_invariance,
    prove_sharp_flat_identity,
    symplectic_engine,
    symplectic_structure,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    X, Y, Z = vector_fields("X Y Z")
    P = poisson_structure()
    S = symplectic_structure()
    return reg, f, g, X, Y, Z, P, S


class TestStructure:
    def test_two_form_guard(self, setup):
        (alpha,) = forms("α", degree=1)
        with pytest.raises(ValueError):
            SymplecticStructure(alpha)

    def test_defining_relation_node_form(self, setup):
        """ι_{X_f}ω → df (the canonical definition)."""
        reg, f, _, _, _, _, _, S = setup
        eng = symplectic_engine(S, registry=reg)
        node = Act(Interior(S.hamiltonian(f)), S.omega)
        out, steps = eng.expand(node)
        assert out == d(f)
        assert steps

    def test_defining_relation_evaluated(self, setup):
        """ω(X_f, Y) → ⟨df, Y⟩ → Y(f); second slot picks up the
        alternating sign."""
        from jacopy.algorithms.simplify import simplify
        from jacopy.core.expr import Neg

        reg, f, _, _, Y, _, _, S = setup
        eng = symplectic_engine(S, registry=reg)
        n1 = MultiEval(
            S.omega,
            S.hamiltonian(f),
            Y,
            alternating=True,
            slot_kind="vector",
        )
        o1, _ = eng.expand(n1)
        n2 = MultiEval(
            S.omega,
            Y,
            S.hamiltonian(f),
            alternating=True,
            slot_kind="vector",
        )
        o2, _ = eng.expand(n2)
        assert simplify(o1, reg) == simplify(
            Neg(simplify(o2, reg)), reg
        )

    def test_arbitrary_names(self, setup):
        """Arbitrary-named objects work (published-package check)."""
        reg = PropertyRegistry()
        (H,) = functions("H", registry=reg)
        A, B = vector_fields("A B")
        S2 = symplectic_structure("Ω")
        assert prove_hamiltonian_flow_invariance(
            S2, H, A, B, registry=reg
        ).steps


class TestFlowInvariance:
    def test_closes_with_declaration(self, setup):
        """Q2(a): (L_{X_f}ω)(Y,Z) = 0, theorem-tagged closedness
        citation inside."""
        reg, f, _, _, Y, Z, _, S = setup
        chain = prove_hamiltonian_flow_invariance(
            S, f, Y, Z, registry=reg
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_honest_without_declaration(self, setup):
        reg, f, _, _, Y, Z, _, S = setup
        lhs = MultiEval(
            Act(CARTAN_TM.lie(S.hamiltonian(f)), S.omega),
            Y,
            Z,
            alternating=True,
            slot_kind="vector",
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                Integer(0),
                registry=reg,
                engine=symplectic_engine(
                    S, registry=reg, declare_closed=False
                ),
            )


class TestBracketChain:
    """Q2(b) under the first-slot interior convention:
    {f,g} = ω(X_g, X_f) = X_f(g) = ι_{X_f}dg = ι_{X_f}ι_{X_g}ω."""

    def test_via_omega(self, setup):
        reg, f, g, _, _, _, P, S = setup
        assert prove_bracket_via_omega(P, S, f, g, registry=reg).steps

    def test_via_iota(self, setup):
        reg, f, g, _, _, _, P, S = setup
        assert prove_bracket_via_iota(P, S, f, g, registry=reg).steps

    def test_via_double_iota(self, setup):
        reg, f, g, _, _, _, P, S = setup
        assert prove_bracket_via_double_iota(
            P, S, f, g, registry=reg
        ).steps

    def test_wrong_slot_order_refused(self, setup):
        """{f,g} = ω(X_f, X_g) belongs to the OPPOSITE interior
        convention — must not close."""
        reg, f, g, _, _, _, P, S = setup
        rhs_bad = MultiEval(
            S.omega,
            S.hamiltonian(f),
            S.hamiltonian(g),
            alternating=True,
            slot_kind="vector",
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                P.bracket(f, g),
                rhs_bad,
                registry=reg,
                engine=symplectic_engine(
                    S, registry=reg, poisson_bridge=P
                ),
            )

    def test_bridge_requires_opt_in(self, setup):
        """Without poisson_bridge the π-ω connection must not exist:
        {f,g} = ω(X_g,X_f) has no route."""
        reg, f, g, _, _, _, P, S = setup
        rhs = MultiEval(
            S.omega,
            S.hamiltonian(g),
            S.hamiltonian(f),
            alternating=True,
            slot_kind="vector",
        )
        from jacopy.packages.poisson.core import poisson_engine

        eng = poisson_engine(registry=reg, structures=(P,))
        from jacopy.packages.poisson.symplectic import (
            SymplecticHamiltonianDefinition,
        )

        eng.register(SymplecticHamiltonianDefinition(S))
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                P.bracket(f, g), rhs, registry=reg, engine=eng
            )


class TestBridge:
    def test_sharp_flat_identity(self, setup):
        reg, _, _, X, _, _, P, S = setup
        (beta,) = forms("β", degree=1)
        assert prove_sharp_flat_identity(
            P, S, X, beta, registry=reg
        ).steps

    def test_structure_scoping(self, setup):
        """A second symplectic form's Hamiltonians must not trigger
        the first structure's defining relation."""
        reg, f, _, _, Y, _, _, S = setup
        S2 = symplectic_structure("Ω")
        eng = symplectic_engine(S, registry=reg)
        node = MultiEval(
            S.omega,
            S2.hamiltonian(f),
            Y,
            alternating=True,
            slot_kind="vector",
        )
        out, _ = eng.expand(node)
        assert out == node
