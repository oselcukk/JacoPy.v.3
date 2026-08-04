"""Algebroid case — Phase 3.E.4: the regularity assumption and the
locality projector [MC Def 3.9-3.10], plus the annihilation theorem
ρ(L(Df,u,v)) = 0 on local pre-Leibniz algebroids [MC §3]."""

import pytest

from jacopy.core.expr import Integer, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import TheoremBook, cite
from jacopy.central.objects import Bundle, forms, functions
from jacopy.central.algebroid import (
    LocalityOperator,
    algebroid,
    algebroid_engine,
    locality_projector,
    locality_term,
    prove_locality_anchor_annihilation,
    tangent_algebroid,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid(
        "E", Bundle("E"), declare=("local", "anchor-morphism", "regular")
    )
    u, v = E.sections("u v")
    P = locality_projector(E)
    return reg, f, E, u, v, P


# --------------------------------------------------------------------- #
# Regularity + projector construction                                    #
# --------------------------------------------------------------------- #


class TestProjectorConstruction:
    def test_regular_is_declarable(self, setup):
        E = algebroid("E", declare=("regular",))
        assert E.declares("regular")

    def test_requires_regular(self, setup):
        E0 = algebroid("F", Bundle("F"), declare=("local",))
        with pytest.raises(ValueError):
            locality_projector(E0)

    def test_tangent_rejected(self, setup):
        with pytest.raises(ValueError):
            locality_projector(tangent_algebroid())

    def test_projector_is_cinf_linear(self, setup):
        """𝒫 reuses the SectionMap machinery — linearity for free."""
        reg, f, E, u, v, P = setup
        eng = algebroid_engine(E, registry=reg, projectors=(P,))
        out, _ = eng.expand(P(Sum(Product(f, u), v)))
        assert out == Sum(Product(f, P(u)), P(v))


# --------------------------------------------------------------------- #
# The two defining rules                                                 #
# --------------------------------------------------------------------- #


class TestProjectorRules:
    def test_absorption_on_coboundary_forms(self, setup):
        """L̂ ∈ [L̃]: 𝒫(L(Df,u,v)) = L(Df,u,v)."""
        reg, f, E, u, v, P = setup
        eng = algebroid_engine(E, registry=reg, projectors=(P,))
        node = P(locality_term(E, f, u, v))
        out, steps = eng.expand(node)
        assert out == locality_term(E, f, u, v)
        assert any("L̂ ∈ [L̃]" in s.rule for s in steps)

    def test_kernel_on_generic_form(self, setup):
        """im(L̂) ⊂ ker ρ: ρ(𝒫(L(ω,u,v))) = 0 for arbitrary ω."""
        reg, f, E, u, v, P = setup
        (om,) = forms("ω", degree=1, bundle=E.bundle)
        eng = algebroid_engine(E, registry=reg, projectors=(P,))
        out, steps = eng.expand(
            E.anchor(P(LocalityOperator("E", om, u, v)))
        )
        assert out == Integer(0)
        assert any("im(L̂) ⊂ ker ρ" in s.rule for s in steps)

    def test_inert_without_projector(self, setup):
        reg, f, E, u, v, P = setup
        (om,) = forms("ω", degree=1, bundle=E.bundle)
        eng = algebroid_engine(E, registry=reg)  # no projectors kwarg
        node = E.anchor(P(LocalityOperator("E", om, u, v)))
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_scoped_to_projector_name(self, setup):
        """A different map named Q gets no projector rules."""
        from jacopy.central.algebroid import section_map

        reg, f, E, u, v, P = setup
        Q = section_map("Q", E.bundle, E.bundle)
        eng = algebroid_engine(E, registry=reg, projectors=(P,))
        node = Q(locality_term(E, f, u, v))
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_bad_projector_argument(self, setup):
        reg, _, E, *_ = setup
        with pytest.raises(TypeError):
            algebroid_engine(E, registry=reg, projectors=("P",))


# --------------------------------------------------------------------- #
# The annihilation theorem ρ(L(Df,u,v)) = 0                              #
# --------------------------------------------------------------------- #


class TestLocalityAnchorAnnihilation:
    def test_closes_on_local_pre_leibniz(self, setup):
        reg, f, E, u, v, _ = setup
        chain, thm = prove_locality_anchor_annihilation(
            E, u, v, f, registry=reg
        )
        assert chain.steps
        assert thm.lhs == E.anchor(locality_term(E, f, u, v))
        assert thm.rhs == Integer(0)
        assert thm.from_axioms == (
            "left-Leibniz (E)",
            "anchor morphism (E)",
        )

    def test_generator_agreement_is_explicit(self, setup):
        reg, f, E, u, v, _ = setup
        chain, _ = prove_locality_anchor_annihilation(
            E, u, v, f, registry=reg
        )
        assert any(
            s.rule == "agreement on generators (h)" for s in chain.steps
        )

    @pytest.mark.parametrize(
        "decl", [("local",), ("pre-leibniz",), ("almost-leibniz",)]
    )
    def test_fails_without_required_declarations(self, setup, decl):
        reg, f, *_ = setup
        E0 = algebroid("E", Bundle("E"), declare=decl)
        u, v = E0.sections("u v")
        with pytest.raises(ProofFailure):
            prove_locality_anchor_annihilation(E0, u, v, f, registry=reg)

    def test_end_to_end_with_projector(self, setup):
        """ρ(𝒫(L(Df,u,v))) = 0 — absorption reduces to ρ(L(Df,u,v)),
        the cited theorem finishes."""
        reg, f, E, u, v, P = setup
        _, thm = prove_locality_anchor_annihilation(
            E, u, v, f, registry=reg
        )
        book = TheoremBook()
        book.add(thm)
        eng = algebroid_engine(E, registry=reg, projectors=(P,))
        cite(eng, book, thm.name)
        chain = ExpandAndSimplify().prove(
            E.anchor(P(locality_term(E, f, u, v))),
            Integer(0),
            registry=reg,
            engine=eng,
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_df_case_honest_without_citation(self, setup):
        """Without the cited theorem, ρ(𝒫(L(Df,u,v))) reduces via
        absorption to ρ(L(Df,u,v)) and honestly stalls — the Df-case
        vanishing is a pre-Leibniz THEOREM, not a projector axiom."""
        reg, f, E, u, v, P = setup
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                E.anchor(P(locality_term(E, f, u, v))),
                Integer(0),
                registry=reg,
                engine=algebroid_engine(E, registry=reg, projectors=(P,)),
            )
