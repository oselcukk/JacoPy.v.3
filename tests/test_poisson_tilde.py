"""Poisson package — Phase 5.D: the tilde calculus.

The 3.F BracketCalculus instantiation on the cotangent algebroid,
with the ROLE-SWAPPED grading (tilde-sections = 1-forms, tilde-forms
= multivectors) supplied through the calculus grading hooks."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.central.objects.frame import Frame
from jacopy.central.objects.tilde_interior import TildeInterior
from jacopy.central.tangent.exterior import d
from jacopy.packages.poisson.core import poisson_structure
from jacopy.packages.poisson.tilde import (
    LieDCommutationDefinition,
    TildeAnholonomyCoefficient,
    d_tilde,
    prove_d_tilde_on_functions,
    prove_d_tilde_palais_on_vectors,
    prove_d_tilde_squared_on_exacts,
    prove_tilde_gamma_antisymmetry,
    prove_tilde_magic_on_functions,
    tilde_anholonomy_coefficient,
    tilde_calculus,
    tilde_engine,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, g, h = functions("f g h", registry=reg)
    alpha, beta = forms("α β", degree=1)
    (X,) = vector_fields("X")
    P = poisson_structure()
    return reg, f, g, h, alpha, beta, X, P


class TestTildeGrading:
    def test_sections_are_one_forms(self, setup):
        _, _, _, _, alpha, _, X, P = setup
        calc = tilde_calculus(P)
        assert calc.is_section(alpha)
        assert not calc.is_section(X)  # a vector is a tilde-FORM

    def test_forms_are_multivectors(self, setup):
        reg, f, _, _, _, _, X, P = setup
        calc = tilde_calculus(P)
        assert str(calc.form_degree(X, reg)) == "1"
        assert str(calc.form_degree(f, reg)) == "0"

    def test_operator_aware_degree(self, setup):
        reg, f, _, _, _, _, _, P = setup
        calc = tilde_calculus(P)
        assert str(calc.form_degree(d_tilde(P, f), reg)) == "1"


class TestTildeOperators:
    def test_d_tilde_on_functions(self, setup):
        reg, f, _, _, alpha, _, _, P = setup
        assert prove_d_tilde_on_functions(
            P, f, alpha, registry=reg
        ).steps

    def test_d_tilde_palais_on_vectors(self, setup):
        reg, _, _, _, alpha, beta, X, P = setup
        assert prove_d_tilde_palais_on_vectors(
            P, X, alpha, beta, registry=reg
        ).steps

    def test_tilde_magic_on_functions(self, setup):
        """L̃_α f = (d̃ι̃_α + ι̃_αd̃)f — the tilde Cartan magic."""
        reg, f, _, _, alpha, _, _, P = setup
        assert prove_tilde_magic_on_functions(
            P, alpha, f, registry=reg
        ).steps

    def test_wrong_magic_sign_fails(self, setup):
        reg, f, _, _, alpha, _, _, P = setup
        iota = TildeInterior(alpha)
        calc = tilde_calculus(P)
        from jacopy.packages.poisson.tilde import lie_tilde

        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Act(lie_tilde(P, alpha), f),
                Sum(
                    Act(calc.d, Act(iota, f)),
                    Neg(Act(iota, d_tilde(P, f))),
                ),
                registry=reg,
                engine=tilde_engine(P, registry=reg),
            )


class TestTildeDSquared:
    def test_closes_under_declaration(self, setup):
        """(d̃d̃f)(dg,dh) = 0 — the conditional Cartan of 3.F in the
        tilde world: Poisson declaration + cited Jacobi instances."""
        reg, f, g, h, _, _, _, P = setup
        chain = prove_d_tilde_squared_on_exacts(
            P, f, g, h, registry=reg
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_honest_without_declaration(self, setup):
        reg, f, g, h, _, _, _, P = setup
        lhs = MultiEval(
            d_tilde(P, d_tilde(P, f)),
            d(g),
            d(h),
            alternating=True,
            slot_kind="covector",
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                Integer(0),
                registry=reg,
                engine=tilde_engine(
                    P, registry=reg, declare_poisson=False
                ),
            )


class TestLieDCommutation:
    def test_rule_is_theorem_classified(self, setup):
        """L_X(df) → d(X(f)) carries its Phase 2 derivation in
        foundational mode, and the derivation avoids the rule."""
        reg, f, _, _, _, _, X, P = setup
        eng = tilde_engine(P, registry=reg)
        out, steps = eng.expand(
            Act(
                __import__(
                    "jacopy.central.tangent.exterior",
                    fromlist=["CARTAN_TM"],
                ).CARTAN_TM.lie(X),
                d(f),
            )
        )
        assert out == d(Act(X, f))
        assert steps[0].provenance_tag == "theorem"


class TestTildeAnholonomy:
    def test_extraction_and_canonical_order(self, setup):
        reg, _, _, _, _, _, _, P = setup
        fr = Frame("e")
        assert tilde_anholonomy_coefficient(
            P, fr, "a", "a", "c"
        ) == Integer(0)
        assert tilde_anholonomy_coefficient(
            P, fr, "b", "a", "c"
        ) == Neg(TildeAnholonomyCoefficient(P.pi, "e", "a", "b", "c"))

    def test_antisymmetry_mechanical(self, setup):
        reg, _, _, _, _, _, _, P = setup
        fr = Frame("e")
        assert prove_tilde_gamma_antisymmetry(
            P, fr, "a", "b", "c", registry=reg
        ).steps


class TestEdgeCaseAudit:
    """Edge-case audit regressions (2026-07-30): three genuine
    findings, all fixed — degenerate instances must neither loop nor
    silently mis-expand."""

    def test_zero_form_koszul(self, setup):
        """[0, β]_π = 0 by ℝ-bilinearity — the degree guard must not
        reject the zero form."""
        from jacopy.packages.poisson import koszul_bracket

        _, _, _, _, _, beta, _, P = setup
        assert koszul_bracket(P, Integer(0), beta) == Integer(0)
        assert koszul_bracket(P, beta, Integer(0)) == Integer(0)

    def test_degenerate_morphism_terminates(self, setup):
        """X_{{f,f}} = [X_f, X_f] — the degenerate instance
        normalizes its cited theorem to 0 = 0; the identity rewrite
        must stay inert instead of looping."""
        from jacopy.packages.poisson import prove_hamiltonian_morphism

        reg, f, _, h, _, _, _, P = setup
        chain = prove_hamiltonian_morphism(P, f, f, h, registry=reg)
        assert chain.steps

    def test_identity_theorem_rewrite_inert(self, setup):
        """The general fix: a cited theorem with lhs == rhs never
        fires."""
        from jacopy.proof.chain import ProofChain
        from jacopy.proof.step import ProofStep
        from jacopy.proof.theorems import Theorem, TheoremDefinition

        z = Integer(0)
        thm = Theorem(
            name="trivial",
            statement="0 = 0",
            lhs=z,
            rhs=z,
            proof=ProofChain(
                [ProofStep(z, z, rule="reflexive", justification="-")]
            ),
            generality="instance",
        )
        rule = TheoremDefinition(thm)
        assert not rule.matches(z)

    def test_d_tilde_on_one_form_is_ill_typed_and_inert(self, setup):
        """A 1-form is a tilde-SECTION, not a tilde-form: d̃α must
        stay inert (the −1 degree sentinel), never silently unroll."""
        reg, _, _, _, alpha, beta, _, P = setup
        (gamma,) = forms("γ", degree=1)
        node = MultiEval(
            d_tilde(P, alpha),
            beta,
            gamma,
            alternating=True,
            slot_kind="covector",
        )
        out, steps = tilde_engine(P, registry=reg).expand(node)
        assert not any(
            "intrinsic d" in s.rule for s in steps
        )

    def test_mixed_worlds_coexist(self, setup):
        """TM's d and the tilde d̃ in one engine: each keyed by its
        calculus, no cross-firing."""
        reg, f, _, _, alpha, _, X, P = setup
        eng = tilde_engine(P, registry=reg)
        chain = ExpandAndSimplify().prove(
            Pairing(d(f), X),
            Act(X, f),
            registry=reg,
            engine=eng,
        )
        assert chain.steps

    def test_sn_eval_exact_guard(self, setup):
        """[π,π](α, dg, dh) with a non-exact leg: the EXACT rule's
        Σ_cyc view must not fire — since 5.E.2 the GENERAL
        (anchor-defect) view handles the mixed triple instead."""
        from jacopy.central.tangent.schouten import sn_bracket
        from jacopy.packages.poisson.showcase import showcase_engine

        reg, _, g, h, alpha, _, _, P = setup
        node = MultiEval(
            sn_bracket(P.pi, P.pi),
            alpha,
            d(g),
            d(h),
            alternating=True,
            slot_kind="covector",
        )
        eng = showcase_engine(P, registry=reg, declare_poisson=False)
        _, steps = eng.expand(node)
        fired = [s.rule for s in steps if "SN evaluation" in s.rule]
        assert all("general" in r for r in fired)
        assert fired  # the general view DOES cover the mixed triple


class TestQ3TildeIdentities:
    """Midterm Q3, tilde half (5.D.2): Cartan relations for a vector
    field + the three algebroid-calculus identities in the tilde
    world. Magic and [L̃, ι̃] are definitional (general 1-forms); the
    three displayed identities carry the Jacobi obstruction and close
    on exact generators via cited Poisson-Jacobi instances."""

    def test_magic_on_vectors(self, setup):
        from jacopy.packages.poisson.tilde import (
            prove_tilde_magic_on_vectors,
        )

        reg, _, _, _, alpha, beta, X, P = setup
        assert prove_tilde_magic_on_vectors(
            P, alpha, X, beta, registry=reg
        ).steps

    def test_iota_commutator_on_vectors(self, setup):
        from jacopy.packages.poisson.tilde import (
            prove_lie_tilde_iota_commutator_on_vectors,
        )

        reg, _, _, _, alpha, beta, X, P = setup
        assert prove_lie_tilde_iota_commutator_on_vectors(
            P, alpha, beta, X, registry=reg
        ).steps

    def test_lie_commutator_on_exacts(self, setup):
        from jacopy.packages.poisson.tilde import (
            prove_lie_tilde_commutator_on_exacts,
        )

        reg, f, g, h, _, _, X, P = setup
        chain = prove_lie_tilde_commutator_on_exacts(
            P, f, g, X, h, registry=reg
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_d_iota_commutation_on_exacts(self, setup):
        from jacopy.packages.poisson.tilde import (
            prove_lie_tilde_d_iota_commutation_on_exacts,
        )

        reg, f, g, h, _, _, X, P = setup
        assert prove_lie_tilde_d_iota_commutation_on_exacts(
            P, f, g, X, h, registry=reg
        ).steps

    def test_d_iota_exact_on_exacts(self, setup):
        from jacopy.packages.poisson.tilde import (
            prove_lie_tilde_d_iota_exact_on_exacts,
        )

        reg, f, g, h, _, _, X, P = setup
        assert prove_lie_tilde_d_iota_exact_on_exacts(
            P, f, g, X, h, registry=reg
        ).steps

    def test_general_form_honest_fail(self, setup):
        """The general-form identity needs the general SN evaluation
        (Phase 5.E) — without it the Jacobi obstruction must survive
        as an honest residual."""
        from jacopy.packages.poisson.tilde import (
            d_tilde,
            lie_tilde,
            tilde_calculus,
            tilde_engine,
        )

        reg, _, _, _, alpha, beta, X, P = setup
        (gamma,) = forms("γ", degree=1)
        calc = tilde_calculus(P)
        iota_a = TildeInterior(alpha)
        lhs = Sum(
            Pairing(
                Act(lie_tilde(P, beta), d_tilde(P, Act(iota_a, X))),
                gamma,
            ),
            Neg(
                Pairing(
                    Act(
                        calc.d,
                        Act(
                            TildeInterior(beta),
                            d_tilde(P, Act(iota_a, X)),
                        ),
                    ),
                    gamma,
                )
            ),
        )
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                Integer(0),
                registry=reg,
                engine=tilde_engine(P, registry=reg),
            )


class TestMixedWorldTypeGuards:
    """5.D.2 audit: the mixed-world identities exposed two type
    holes — both must stay closed."""

    def test_usual_rules_inert_on_form_slot(self, setup):
        """⟨L_X ω, γ⟩ with a 1-FORM γ in the vector slot: the usual
        intrinsic rules must NOT unroll (they produced the ill-typed
        ``[X, γ]_VF`` before the slot guard)."""
        from jacopy.central.tangent.exterior import CARTAN_TM
        from jacopy.central.tangent.engine import tangent_engine

        reg, _, _, _, _, _, X, _ = setup
        (omega,) = forms("ω", degree=1)
        (gamma,) = forms("γ", degree=1)
        eng = tangent_engine(registry=reg)
        node = Pairing(Act(CARTAN_TM.lie(X), omega), gamma)
        out, _ = eng.expand(node)
        assert out == node
        node2 = Pairing(d(functions("q", registry=reg)[0]), gamma)
        out2, _ = eng.expand(node2)
        assert out2 == node2

    def test_d_tilde_scalar_is_not_a_tilde_section(self, setup):
        """``d̃s`` reads as ordinary wedge-degree 1 but is a
        tilde-FORM: the section test must refuse it, else
        ``L̃_β(d̃s)`` becomes the ill-typed ``[β, d̃s]_π``."""
        from jacopy.packages.poisson.tilde import (
            d_tilde,
            tilde_calculus,
        )

        reg, f, _, _, alpha, _, X, P = setup
        calc = tilde_calculus(P)
        s = Pairing(X, alpha)
        assert not calc.is_section(d_tilde(P, s))
        assert not calc.is_section(d_tilde(P, f))
        assert calc.is_section(alpha)


class TestGeneralSNEval:
    """Phase 5.E.2: the general (anchor-defect) evaluation view of
    [π,π] — consistency with the exact Σ_cyc view, and the general
    Q3 tilde identities it unlocks."""

    def test_views_agree_on_exacts(self, setup):
        """2⟨dh, [π♯df,π♯dg] − π♯[df,dg]⟩ = 2Σ_cyc π(df, dπ(dg,dh)),
        declaration withheld — the general rule's sign is a THEOREM."""
        from jacopy.packages.poisson.showcase import (
            prove_sn_eval_views_agree_on_exacts,
        )

        reg, f, g, h, _, _, _, P = setup
        assert prove_sn_eval_views_agree_on_exacts(
            P, f, g, h, registry=reg
        ).steps

    def test_id2_general(self, setup):
        from jacopy.packages.poisson.tilde import (
            prove_lie_tilde_d_iota_commutation_general,
        )

        reg, _, _, _, alpha, beta, X, P = setup
        (gamma,) = forms("γ", degree=1)
        chain = prove_lie_tilde_d_iota_commutation_general(
            P, alpha, beta, X, gamma, registry=reg
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_id3_general(self, setup):
        from jacopy.packages.poisson.tilde import (
            prove_lie_tilde_d_iota_exact_general,
        )

        reg, _, _, _, alpha, beta, X, P = setup
        (gamma,) = forms("γ", degree=1)
        assert prove_lie_tilde_d_iota_exact_general(
            P, alpha, beta, X, gamma, registry=reg
        ).steps

    def test_general_koszul_jacobi_CLOSED(self, setup):
        """5.E.2b CLOSED (2026-09-07): the general Koszul Jacobi
        identity closes via the greedy stall-difference + exact
        ℚ-linear citation phase (packages/poisson/koszul_jacobi.py),
        engine-verified. ~42 s — runs with JACOPY_RUN_SLOW=1; the
        honest-fail companion below always runs."""
        import os

        if not os.environ.get("JACOPY_RUN_SLOW"):
            pytest.skip(
                "42 s closure; set JACOPY_RUN_SLOW=1 to run "
                "(verified offline: 18 steps)"
            )
        from jacopy.packages.poisson.koszul_jacobi import (
            prove_general_koszul_jacobi,
        )

        reg, f, _, _, alpha, beta, X, P = setup
        (gamma,) = forms("γ", degree=1)
        chain = prove_general_koszul_jacobi(
            P, alpha, beta, gamma, X, f, registry=reg
        )
        assert chain.steps

    def test_general_koszul_jacobi_needs_declaration(self, setup):
        """Without the declared [π,π] = 0 the general Jacobi must
        honest-fail (fast check)."""
        from jacopy.proof.strategies import ProofFailure as PF
        from jacopy.packages.poisson.koszul_jacobi import (
            prove_general_koszul_jacobi,
        )

        reg, f, _, _, alpha, beta, X, P = setup
        (gamma,) = forms("γ", degree=1)
        with pytest.raises(PF):
            prove_general_koszul_jacobi(
                P, alpha, beta, gamma, X, f,
                registry=reg, declare_poisson=False,
            )
