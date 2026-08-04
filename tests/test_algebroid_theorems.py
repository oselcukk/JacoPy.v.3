"""Algebroid case — the flagship theorem (Phase 3.D):
right-Leibniz + Leibniz-Jacobi ⇒ anchor morphism
("every Leibniz algebroid is a pre-Leibniz algebroid", Bourbaki
Thm 4.1)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Integer
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import TheoremBook, cite
from jacopy.central.objects import Bundle, functions
from jacopy.central.tangent.lie_bracket import LieBracketActionDefinition
from jacopy.central.algebroid import (
    algebroid,
    algebroid_engine,
    anchor_predator,
    prove_anchor_morphism,
    prove_anchor_predator_vanishes,
    tangent_algebroid,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("leibniz",))
    u, v = E.sections("u v")
    return reg, f, E, u, v


def _walk(steps):
    """All steps of a chain including nested children."""
    for s in steps:
        yield s
        yield from _walk(s.children)


def _cited_engine(E, thm, reg, *, mode="efficient"):
    book = TheoremBook()
    book.add(thm)
    eng = algebroid_engine(E, registry=reg, mode=mode)
    eng.register(LieBracketActionDefinition())
    cite(eng, book, thm.name)
    return eng


# --------------------------------------------------------------------- #
# The theorem closes — and its record is honest                          #
# --------------------------------------------------------------------- #


class TestFlagshipTheorem:
    def test_closes_under_leibniz(self, setup):
        reg, f, E, u, v = setup
        chain, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        assert chain.steps
        assert thm.lhs == E.anchor(E.bracket(u, v))
        assert thm.rhs == LieBracketVF(E.anchor(u), E.anchor(v))
        assert thm.generality == "generic-function"

    def test_uses_exactly_the_two_declared_axioms(self, setup):
        """The nested step record must show BOTH inputs — and no other
        declared axiom (there is none to use)."""
        reg, f, E, u, v = setup
        chain, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        rules = {s.rule for s in _walk(chain.steps)}
        assert any("right-Leibniz (E)" in r for r in rules)
        assert any("Leibniz-Jacobi (E)" in r for r in rules)
        assert not any("anchor morphism (E)" in r for r in rules)
        assert thm.from_axioms == (
            "right-Leibniz (E)",
            "Leibniz-Jacobi (E)",
        )

    def test_generator_agreement_steps_are_explicit(self, setup):
        """The two generator-agreement inferences are named steps, not
        silent jumps."""
        reg, f, E, u, v = setup
        chain, _ = prove_anchor_morphism(E, u, v, f, registry=reg)
        rules = [s.rule for s in chain.steps]
        assert "agreement on generators (w)" in rules
        assert "agreement on generators (f)" in rules

    def test_explicit_auxiliary_section(self, setup):
        reg, f, E, u, v = setup
        (z,) = E.sections("z")
        chain, thm = prove_anchor_morphism(E, u, v, f, w=z, registry=reg)
        assert chain.steps and thm.lhs == E.anchor(E.bracket(u, v))

    def test_fresh_w_avoids_name_collision(self, setup):
        """Sections named w must not collide with the auxiliary."""
        reg, f, E, *_ = setup
        a, b = E.sections("w z")
        chain, thm = prove_anchor_morphism(E, a, b, f, registry=reg)
        assert thm.lhs == E.anchor(E.bracket(a, b))


# --------------------------------------------------------------------- #
# Honest failure without the inputs                                      #
# --------------------------------------------------------------------- #


class TestHonesty:
    @pytest.mark.parametrize(
        "decl", [("right-leibniz",), ("jacobi",), ("local",)]
    )
    def test_missing_declaration_fails(self, setup, decl):
        reg, f, *_ = setup
        E0 = algebroid("E", Bundle("E"), declare=decl)
        u, v = E0.sections("u v")
        with pytest.raises(ProofFailure):
            prove_anchor_morphism(E0, u, v, f, registry=reg)

    def test_declared_morphism_is_an_axiom_not_a_theorem(self, setup):
        reg, f, *_ = setup
        E0 = algebroid("E", Bundle("E"), declare=("pre-leibniz", "jacobi"))
        u, v = E0.sections("u v")
        with pytest.raises(ValueError):
            prove_anchor_morphism(E0, u, v, f, registry=reg)

    def test_tangent_case_rejected(self, setup):
        reg, f, *_ = setup
        TMalg = tangent_algebroid()
        X, Y = TMalg.sections("X Y")
        with pytest.raises(ValueError):
            prove_anchor_morphism(TMalg, X, Y, f, registry=reg)


# --------------------------------------------------------------------- #
# Citation: the proven morphism enters proofs as a theorem               #
# --------------------------------------------------------------------- #


class TestCitation:
    def test_act_level_identity_closes_via_citation(self, setup):
        """ρ([u,v])(f) = [ρ(u), ρ(v)](f) — demo 3a's honest failure,
        now closed by citing the derived theorem."""
        reg, f, E, u, v = setup
        _, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        chain = ExpandAndSimplify().prove(
            Act(E.anchor(E.bracket(u, v)), f),
            Act(LieBracketVF(E.anchor(u), E.anchor(v)), f),
            registry=reg,
            engine=_cited_engine(E, thm, reg),
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_false_identity_still_fails(self, setup):
        """Swapped commutator (wrong sign) must NOT close even with
        the theorem cited."""
        reg, f, E, u, v = setup
        _, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Act(E.anchor(E.bracket(u, v)), f),
                Act(LieBracketVF(E.anchor(v), E.anchor(u)), f),
                registry=reg,
                engine=_cited_engine(E, thm, reg),
            )

    def test_scoped_to_algebroid(self, setup):
        """E's theorem cited into F's engine must not close F's
        statement."""
        reg, f, E, u, v = setup
        _, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        F = algebroid("F", Bundle("F"), declare=("leibniz",))
        uf, vf = F.sections("u v")
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Act(F.anchor(F.bracket(uf, vf)), f),
                Act(LieBracketVF(F.anchor(uf), F.anchor(vf)), f),
                registry=reg,
                engine=_cited_engine(F, thm, reg),
            )

    def test_foundational_mode_inlines_derivation(self, setup):
        reg, f, E, u, v = setup
        _, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        chain = ExpandAndSimplify().prove(
            Act(E.anchor(E.bracket(u, v)), f),
            Act(LieBracketVF(E.anchor(u), E.anchor(v)), f),
            registry=reg,
            engine=_cited_engine(E, thm, reg, mode="foundational"),
        )
        tstep = next(
            s for s in chain.steps if s.provenance_tag == "theorem"
        )
        assert len(tstep.children) == len(thm.proof.steps)


# --------------------------------------------------------------------- #
# The predator face: P_ρ = 0 on a Leibniz algebroid                      #
# --------------------------------------------------------------------- #


class TestPredatorFace:
    def test_predator_vanishes_under_leibniz(self, setup):
        """3.C left this as the honest gap; 3.D closes it."""
        reg, f, E, u, v = setup
        chain = prove_anchor_predator_vanishes(E, u, v, f, registry=reg)
        assert chain.final == Integer(0)
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_predator_still_fails_without_plain_engine(self, setup):
        """Without the citation the 3.C behaviour is unchanged: the
        plain engine leaves the defect standing."""
        reg, f, E, u, v = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                anchor_predator(E, u, v),
                Integer(0),
                registry=reg,
                engine=algebroid_engine(E, registry=reg),
            )
