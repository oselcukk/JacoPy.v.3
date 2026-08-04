"""Algebroid case — Phase 3.F: the algebroid Cartan calculus.

``BracketCalculus(ρ_E, [·,·]_E)`` instantiation → ``d_E``, ``L^E``,
``ι``; the definitional relations hold with NO hierarchy declarations
(incl. the Cartan magic formula), while the conditional ones close
exactly under their axioms: ``d_E²=0`` on functions ⟺ anchor
morphism; on 1-forms it is the LIE algebroid theorem (antisymmetric +
Jacobi + morphism). BC numbering follows the Bourbaki pre-calculus
axiom sheet (BC4 checked against the original arXiv:2210.00548
Def 9.1, not the transcription)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import TheoremBook, cite
from jacopy.central.objects import Bundle, forms, functions
from jacopy.central.objects.interior import Interior
from jacopy.central.algebroid import (
    algebroid,
    algebroid_calculus,
    algebroid_engine,
    d_E,
    lie_E,
    prove_anchor_morphism,
    prove_cartan_magic_on_functions,
    prove_cartan_magic_on_one_forms,
    prove_d_coincides_with_coboundary,
    prove_d_on_functions,
    prove_d_squared_zero_on_functions,
    prove_d_squared_zero_on_one_forms,
    prove_lie_core_cases,
    prove_palais_on_one_forms,
    tangent_algebroid,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"))  # NO declarations
    u, v, w = E.sections("u v w")
    (alpha,) = forms("α", degree=1, bundle=E.bundle)
    return reg, f, E, u, v, w, alpha


# --------------------------------------------------------------------- #
# Instantiation                                                          #
# --------------------------------------------------------------------- #


class TestInstantiation:
    def test_tangent_reduction_is_literal(self, setup):
        from jacopy.central.tangent.exterior import CARTAN_TM

        assert algebroid_calculus(tangent_algebroid()) is CARTAN_TM

    def test_calculus_identity_and_display(self, setup):
        _, f, E, *_ = setup
        calc = algebroid_calculus(E)
        assert calc.name == "Cartan-E"
        assert d_E(E, f).op.calculus_name == "Cartan-E"
        u = E.sections("u")[0]
        assert lie_E(E, u)._repr_inner().startswith("L^E_")

    def test_distinct_from_tm_d(self, setup):
        """d_E and the TM d are different operators."""
        from jacopy.central.tangent.exterior import exterior_d

        _, f, E, *_ = setup
        assert d_E(E, f).op != exterior_d()


# --------------------------------------------------------------------- #
# Definitional relations — no declarations needed                        #
# --------------------------------------------------------------------- #


class TestDefinitionalRelations:
    def test_d_on_functions(self, setup):
        reg, f, E, u, *_ = setup
        assert prove_d_on_functions(E, u, f, registry=reg).steps

    def test_d_coincides_with_coboundary(self, setup):
        """The calculus d_E and the 3.A coboundary D agree on
        functions — the design-coherence check."""
        reg, f, E, u, *_ = setup
        assert prove_d_coincides_with_coboundary(
            E, u, f, registry=reg
        ).steps

    def test_palais_on_one_forms(self, setup):
        reg, _, E, u, v, _, alpha = setup
        assert prove_palais_on_one_forms(
            E, alpha, u, v, registry=reg
        ).steps

    def test_lie_core_cases(self, setup):
        reg, f, E, u, v, *_ = setup
        chain_f, chain_v = prove_lie_core_cases(E, u, v, f, registry=reg)
        assert chain_f.steps and chain_v.steps

    def test_lie_on_nested_bracket_section(self, setup):
        """L^E_u [v,w]_E → [u,[v,w]]_E — the section recognition must
        reach algebroid-bracket atoms (wedge-degree protocol)."""
        reg, _, E, u, v, w, _ = setup
        eng = algebroid_engine(E, registry=reg)
        out, steps = eng.expand(Act(lie_E(E, u), E.bracket(v, w)))
        assert out == E.bracket(u, E.bracket(v, w))
        assert steps

    def test_interior_bc6(self, setup):
        """BC6: ι_u d_E f = ρ(u)(f)."""
        reg, f, E, u, *_ = setup
        chain = ExpandAndSimplify().prove(
            Act(Interior(u), d_E(E, f)),
            Act(E.anchor(u), f),
            registry=reg,
            engine=algebroid_engine(E, registry=reg),
        )
        assert chain.steps

    def test_cartan_magic_bc7(self, setup):
        """BC7: the magic formula — a theorem of the definitions on
        ANY anchored bundle with an ℝ-bilinear bracket."""
        reg, f, E, u, _, _, alpha = setup
        assert prove_cartan_magic_on_functions(
            E, u, f, registry=reg
        ).steps
        assert prove_cartan_magic_on_one_forms(
            E, u, alpha, u, registry=reg
        ).steps

    def test_iota_lie_commutator_bc8(self, setup):
        """BC8 on 1-forms: L^E_u(ι_v α) − ι_v(L^E_u α) = ι_[u,v] α."""
        reg, _, E, u, v, _, alpha = setup
        lhs = Sum(
            Act(lie_E(E, u), Act(Interior(v), alpha)),
            Neg(Act(Interior(v), Act(lie_E(E, u), alpha))),
        )
        rhs = Act(Interior(E.bracket(u, v)), alpha)
        chain = ExpandAndSimplify().prove(
            lhs, rhs, registry=reg, engine=algebroid_engine(E, registry=reg)
        )
        assert chain.steps

    def test_false_magic_sign_fails(self, setup):
        """L^E_u f = (d_E ι_u − ι_u d_E) f is wrong."""
        reg, f, E, u, *_ = setup
        iota = Interior(u)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                Act(lie_E(E, u), f),
                Sum(
                    Act(algebroid_calculus(E).d, Act(iota, f)),
                    Neg(Act(iota, d_E(E, f))),
                ),
                registry=reg,
                engine=algebroid_engine(E, registry=reg),
            )


# --------------------------------------------------------------------- #
# Conditional relations — the hierarchy at work                          #
# --------------------------------------------------------------------- #


class TestConditionalRelations:
    def test_d_squared_functions_needs_morphism(self, setup):
        reg, f, E, u, v, *_ = setup
        with pytest.raises(ProofFailure):
            prove_d_squared_zero_on_functions(E, u, v, f, registry=reg)

    def test_d_squared_functions_closes_under_morphism(self, setup):
        reg, f, *_ = setup
        E = algebroid("E", Bundle("E"), declare=("pre-leibniz",))
        u, v = E.sections("u v")
        assert prove_d_squared_zero_on_functions(
            E, u, v, f, registry=reg
        ).steps

    def test_d_squared_functions_via_cited_flagship(self, setup):
        """On a Leibniz algebroid the morphism is not declared — but
        the 3.D theorem, cited, closes d_E²f = 0. The hierarchy and
        the calculus meet."""
        reg, f, *_ = setup
        E = algebroid("E", Bundle("E"), declare=("leibniz",))
        u, v = E.sections("u v")
        _, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
        book = TheoremBook()
        book.add(thm)
        eng = algebroid_engine(E, registry=reg)
        cite(eng, book, thm.name)
        chain = prove_d_squared_zero_on_functions(
            E, u, v, f, registry=reg, engine=eng
        )
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

    def test_lie_derivative_d_commutation_bc5_conditional(self, setup):
        """BC5 ⟨L^E_u d_E f, v⟩ = ⟨d_E L^E_u f, v⟩ needs the anchor
        morphism even on functions — conditional, and honest."""
        reg, f, *_ = setup
        E_yes = algebroid("E", Bundle("E"), declare=("pre-leibniz",))
        u, v = E_yes.sections("u v")
        lhs = Pairing(Act(lie_E(E_yes, u), d_E(E_yes, f)), v)
        rhs = Pairing(d_E(E_yes, Act(lie_E(E_yes, u), f)), v)
        chain = ExpandAndSimplify().prove(
            lhs,
            rhs,
            registry=reg,
            engine=algebroid_engine(E_yes, registry=reg),
        )
        assert chain.steps

        E_no = algebroid("E", Bundle("E"))
        lhs = Pairing(Act(lie_E(E_no, u), d_E(E_no, f)), v)
        rhs = Pairing(d_E(E_no, Act(lie_E(E_no, u), f)), v)
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(
                lhs,
                rhs,
                registry=reg,
                engine=algebroid_engine(E_no, registry=reg),
            )

    def test_d_squared_one_forms_is_a_lie_theorem(self, setup):
        """BC9 beyond functions: closes on a Lie algebroid with the
        morphism; fails honestly without antisymmetry — for a genuine
        Leibniz bracket d_E² ≠ 0."""
        reg, f, *_ = setup
        E = algebroid(
            "E", Bundle("E"), declare=("lie", "anchor-morphism")
        )
        u, v, w = E.sections("u v w")
        (alpha,) = forms("α", degree=1, bundle=E.bundle)
        chain, used = prove_d_squared_zero_on_one_forms(
            E, alpha, u, v, w, registry=reg
        )
        assert chain.steps and used
        assert any(s.provenance_tag == "theorem" for s in chain.steps)

        E_no_anti = algebroid(
            "E", Bundle("E"), declare=("pre-leibniz", "jacobi")
        )
        with pytest.raises(ProofFailure):
            prove_d_squared_zero_on_one_forms(
                E_no_anti, alpha, u, v, w, registry=reg
            )

    def test_d_squared_one_forms_needs_jacobi(self, setup):
        reg, f, *_ = setup
        E = algebroid(
            "E", Bundle("E"), declare=("pre-leibniz", "antisymmetric")
        )
        u, v, w = E.sections("u v w")
        (alpha,) = forms("α", degree=1, bundle=E.bundle)
        with pytest.raises(ProofFailure):
            prove_d_squared_zero_on_one_forms(
                E, alpha, u, v, w, registry=reg
            )


# --------------------------------------------------------------------- #
# The antisymmetry declaration + the lie level                           #
# --------------------------------------------------------------------- #


class TestAntisymmetry:
    def test_lie_level_contents(self, setup):
        E = algebroid("E", declare=("lie",))
        assert E.declarations == frozenset(
            {"right-leibniz", "antisymmetric", "jacobi"}
        )

    def test_orientation_and_square_zero(self, setup):
        reg, _, _, *_ = setup
        E = algebroid("E", Bundle("E"), declare=("antisymmetric",))
        u, v = E.sections("u v")
        eng = algebroid_engine(E, registry=reg)
        out, _ = eng.expand(E.bracket(v, u))
        assert out == Neg(E.bracket(u, v))
        out, _ = eng.expand(E.bracket(u, u))
        assert out == Integer(0)

    def test_inert_without_declaration(self, setup):
        reg, _, E, u, v, *_ = setup
        eng = algebroid_engine(E, registry=reg)
        node = E.bracket(v, u)
        out, steps = eng.expand(node)
        assert out == node and not steps

    def test_arbitrary_names(self, setup):
        reg, f, *_ = setup
        M = algebroid("M", Bundle("M"), declare=("lie", "anchor-morphism"))
        A, B, C = M.sections("A B C")
        (theta,) = forms("θ", degree=1, bundle=M.bundle)
        chain, _ = prove_d_squared_zero_on_one_forms(
            M, theta, A, B, C, registry=reg
        )
        assert chain.steps
