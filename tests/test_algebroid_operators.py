"""Algebroid case — defect operators: Derivator, Predator (+ the
SectionMap machinery and the slot protocol) — Phase 3.C.
Jacobiator itself landed with 3.B; its declaration tests live in
test_algebroid_declarations.py."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import TM, Bundle, functions
from jacopy.central.algebroid import (
    MappedSection,
    algebroid,
    algebroid_engine,
    anchor_predator,
    derivator,
    predator,
    section_map,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("right-leibniz",))
    u, v, w = E.sections("u v w")
    phi = section_map("Φ", E.bundle, E.bundle)
    return reg, E, u, v, w, f, phi


# --------------------------------------------------------------------- #
# SectionMap Φ                                                           #
# --------------------------------------------------------------------- #


class TestSectionMap:
    def test_mapped_section_is_section(self, setup):
        _, E, u, *_ = setup
        phi = section_map("Φ", E.bundle, E.bundle)
        assert isinstance(phi(u), MappedSection)
        assert phi(u)._repr_inner() == "Φ(u)"

    def test_linearity(self, setup):
        reg, E, u, v, _, f, phi = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(phi(Sum(Product(f, u), v)))
        assert out == Sum(Product(f, phi(u)), phi(v))
        out, _ = eng.expand(phi(Neg(u)))
        assert out == Neg(phi(u))
        out, _ = eng.expand(phi(Integer(0)))
        assert out == Integer(0)


# --------------------------------------------------------------------- #
# Derivator D_Φ                                                          #
# --------------------------------------------------------------------- #


class TestDerivator:
    def test_expansion(self, setup):
        reg, E, u, v, _, _, phi = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(derivator(E, phi, u, v))
        assert out == Sum(
            phi(E.bracket(u, v)),
            Neg(E.bracket(phi(u), v)),
            Neg(E.bracket(u, phi(v))),
        )

    def test_requires_endomorphism(self, setup):
        _, E, u, v, *_ = setup
        bad = section_map("ψ", E.bundle, TM)  # E → TM, not E → E
        with pytest.raises(ValueError):
            derivator(E, bad, u, v)

    def test_module_identity(self, setup):
        """D_Φ(u, fv) = f·D_Φ(u,v) − ρ(Φu)(f)·v  (right-Leibniz
        declared) — the C∞-module analysis of MC §3, and the first
        theorem that needed rules firing INSIDE atom slots."""
        reg, E, u, v, _, f, phi = setup
        eng = algebroid_engine(E, registry=reg)
        lhs = derivator(E, phi, u, Product(f, v))
        rhs = Sum(
            Product(f, derivator(E, phi, u, v)),
            Neg(Product(Act(E.anchor(phi(u)), f), v)),
        )
        chain = ExpandAndSimplify().prove(lhs, rhs, registry=reg, engine=eng)
        assert chain.steps

    def test_module_identity_needs_right_leibniz(self, setup):
        reg, _, _, _, _, f, _ = setup
        E0 = algebroid("E", Bundle("E"))  # no declarations
        u, v = E0.sections("u v")
        phi = section_map("Φ", E0.bundle, E0.bundle)
        lhs = derivator(E0, phi, u, Product(f, v))
        rhs = Sum(
            Product(f, derivator(E0, phi, u, v)),
            Neg(Product(Act(E0.anchor(phi(u)), f), v)),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs, rhs, registry=reg,
                engine=algebroid_engine(E0, registry=reg),
            )


# --------------------------------------------------------------------- #
# Predator P_Φ                                                           #
# --------------------------------------------------------------------- #


class TestPredator:
    def test_expansion_generic_map(self, setup):
        reg, E, u, v, *_ = setup
        psi = section_map("ψ", E.bundle, TM)
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(predator(E, psi, u, v))
        assert out == Sum(
            psi(E.bracket(u, v)),
            Neg(LieBracketVF(psi(u), psi(v))),
        )

    def test_requires_tangent_target(self, setup):
        _, E, u, v, _, _, phi = setup  # phi: E → E
        with pytest.raises(ValueError):
            predator(E, phi, u, v)

    def test_anchor_predator_expansion(self, setup):
        reg, E, u, v, *_ = setup
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(anchor_predator(E, u, v))
        assert out == Sum(
            E.anchor(E.bracket(u, v)),
            Neg(LieBracketVF(E.anchor(u), E.anchor(v))),
        )

    def test_predator_vanishes_iff_morphism(self, setup):
        """P_ρ = 0 closes exactly when anchor-morphism is declared —
        the defect-operator characterization of the 'pre' property."""
        reg, *_ = setup
        E_pre = algebroid("E", Bundle("E"), declare=("pre-leibniz",))
        u, v = E_pre.sections("u v")
        chain = ExpandAndSimplify().prove(
            anchor_predator(E_pre, u, v), Integer(0),
            registry=reg, engine=algebroid_engine(E_pre, registry=reg),
        )
        assert chain.steps
        assert any("anchor morphism (E)" in s.rule for s in chain.steps)

        E_no = algebroid("E", Bundle("E"), declare=("leibniz",))
        u, v = E_no.sections("u v")
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                anchor_predator(E_no, u, v), Integer(0),
                registry=reg, engine=algebroid_engine(E_no, registry=reg),
            )

    def test_scoped_to_algebroid(self, setup):
        reg, E, u, v, *_ = setup
        other = algebroid("F", Bundle("F"))
        node = anchor_predator(other, u, v)
        out, steps = algebroid_engine(E, registry=reg).expand(node)
        assert out == node and not steps


# --------------------------------------------------------------------- #
# The slot protocol (the structural fix for v2's atom opacity)           #
# --------------------------------------------------------------------- #


class TestSlotProtocol:
    def test_rule_fires_inside_atom_slot(self, setup):
        """Φ([u, fv]) — the bracket lives in the MappedSection's slot;
        the engine must reach it, expand right-Leibniz there, then
        split Φ by linearity."""
        reg, E, u, v, _, f, phi = setup
        eng = algebroid_engine(E, registry=reg)
        node = phi(E.bracket(u, Product(f, v)))
        out, steps = eng.expand(node)
        assert steps
        assert out == Sum(
            Product(Act(E.anchor(u), f), phi(v)),
            Product(f, phi(E.bracket(u, v))),
        )

    def test_nested_bracket_slot(self, setup):
        """[u, [v, fw]] — right-Leibniz fires inside the outer
        bracket's second slot."""
        reg, E, u, v, w, f, _ = setup
        eng = algebroid_engine(E, registry=reg)
        node = E.bracket(u, E.bracket(v, Product(f, w)))
        out, steps = eng.expand(node)
        assert steps
        # The inner bracket expanded; the outer one then split by
        # R-bilinearity over the resulting sum.
        assert node not in list(out.walk())
