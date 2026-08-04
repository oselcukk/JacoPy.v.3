"""Higher-degree audit (2026-07-31, user request): every
degree-general surface exercised to p = 5, forms AND multivectors.

Covers: TM Cartan (magic, d² = 0 with cited Jacobi instances),
algebroid 3.F Cartan, the derived d̂(∇) calculus, alternating
MultiEval signs at arity 5, Schouten-Nijenhuis degree bookkeeping,
and the documented (deferred) gaps on the multivector side."""

import pytest

from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.algebra.derivation import Act
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.objects import Bundle, forms, functions, vector_fields
from jacopy.central.objects.frame import Frame
from jacopy.central.objects.interior import Interior
from jacopy.central.objects.multivector import PVector
from jacopy.central.algebroid import algebroid, locality_projector
from jacopy.central.algebroid.calculus import (
    _engine as calc_engine,
    algebroid_calculus,
    d_E,
    lie_E,
)
from jacopy.central.tangent.cartan import prove_with_bracket_identities
from jacopy.central.tangent.engine import tangent_engine
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.schouten import (
    multivector_degree,
    sn_bracket,
)
from jacopy.packages.metric_affine.e_connection import e_connection
from jacopy.packages.metric_affine.derived import (
    d_derived,
    derived_algebroid,
    derived_engine,
    lie_derived,
)
from jacopy.algorithms.simplify import simplify

PS = list(range(1, 6))


def _ev(expr, slots):
    if len(slots) == 1:
        return Pairing(expr, slots[0])
    return MultiEval(
        expr, *slots, alternating=True, slot_kind="vector"
    )


class TestTMHigherDegrees:
    @pytest.mark.parametrize("p", PS)
    def test_magic(self, p):
        reg = PropertyRegistry()
        X = vector_fields("X")[0]
        slots = vector_fields("Y1 Y2 Y3 Y4 Y5")[:p]
        (om,) = forms(f"ω{p}", degree=p)
        iota = Interior(X)
        lhs = _ev(Act(CARTAN_TM.lie(X), om), slots)
        rhs = Sum(
            _ev(Act(CARTAN_TM.d, Act(iota, om)), slots),
            _ev(Act(iota, d(om)), slots),
        )
        assert (
            ExpandAndSimplify()
            .prove(
                lhs,
                rhs,
                registry=reg,
                engine=tangent_engine(registry=reg),
            )
            .steps
        )

    @pytest.mark.parametrize("p", PS)
    def test_d_squared_zero(self, p):
        """d² = 0 up to p = 5 via the bracket-identity repair loop;
        the cited-Jacobi count grows as C(p+2, 3)."""
        from math import comb

        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        slots = vector_fields("Y1 Y2 Y3 Y4 Y5 Y6 Y7")[: p + 2]
        (om,) = forms(f"ω{p}", degree=p)
        node = MultiEval(
            d(d(om)), *slots, alternating=True, slot_kind="vector"
        )
        chain, used = prove_with_bracket_identities(
            node, Integer(0), f, registry=reg
        )
        assert chain.steps
        assert len(used) == comb(p + 2, 3)


class TestAlgebroidHigherDegrees:
    @pytest.mark.parametrize("p", PS)
    def test_base_magic(self, p):
        reg = PropertyRegistry()
        E = algebroid("E", Bundle("E"), declare=("local", "regular"))
        u = E.sections("u")[0]
        slots = E.sections("v1 v2 v3 v4 v5")[:p]
        (om,) = forms(f"Ω{p}", degree=p, bundle=E.bundle)
        iota = Interior(u)
        calc = algebroid_calculus(E)
        lhs = _ev(Act(lie_E(E, u), om), slots)
        rhs = Sum(
            _ev(Act(calc.d, Act(iota, om)), slots),
            _ev(Act(iota, d_E(E, om)), slots),
        )
        assert (
            ExpandAndSimplify()
            .prove(lhs, rhs, registry=reg, engine=calc_engine(E, reg))
            .steps
        )

    @pytest.mark.parametrize("p", PS)
    def test_derived_magic(self, p):
        reg = PropertyRegistry()
        E = algebroid(
            "E",
            Bundle("E"),
            declare=("local", "regular", "anchor-morphism"),
        )
        u = E.sections("u")[0]
        slots = E.sections("v1 v2 v3 v4 v5")[:p]
        (om,) = forms(f"Ω{p}", degree=p, bundle=E.bundle)
        nabla = e_connection(E)
        fr = Frame("e", bundle=E.bundle)
        P = locality_projector(E)
        D = derived_algebroid(E, projected=True)
        calc = algebroid_calculus(D)
        iota = Interior(u)
        lhs = _ev(Act(lie_derived(D, u), om), slots)
        rhs = Sum(
            _ev(Act(calc.d, Act(iota, om)), slots),
            _ev(Act(iota, d_derived(D, om)), slots),
        )
        assert (
            ExpandAndSimplify()
            .prove(
                lhs,
                rhs,
                registry=reg,
                engine=derived_engine(
                    E, D, nabla, fr, registry=reg, projector=P
                ),
            )
            .steps
        )


class TestAlternatingAritySigns:
    def test_odd_swap_cancels_at_arity_five(self):
        reg = PropertyRegistry()
        (al,) = forms("α", degree=5)
        Y = vector_fields("Y1 Y2 Y3 Y4 Y5")
        m1 = MultiEval(al, *Y, alternating=True, slot_kind="vector")
        m2 = MultiEval(
            al, Y[1], Y[0], *Y[2:], alternating=True, slot_kind="vector"
        )
        assert simplify(Sum(m1, m2), reg) == Integer(0)

    def test_even_cycle_matches_at_arity_five(self):
        reg = PropertyRegistry()
        (al,) = forms("α", degree=5)
        Y = vector_fields("Y1 Y2 Y3 Y4 Y5")
        m1 = MultiEval(al, *Y, alternating=True, slot_kind="vector")
        m3 = MultiEval(
            al,
            Y[1],
            Y[2],
            Y[0],
            *Y[3:],
            alternating=True,
            slot_kind="vector",
        )
        assert simplify(Sum(m3, Neg(m1)), reg) == Integer(0)


class TestMultivectorBookkeeping:
    """The VECTOR side of the audit."""

    @pytest.mark.parametrize("p", PS)
    def test_pvector_degree(self, p):
        reg = PropertyRegistry()
        assert multivector_degree(PVector(f"m{p}", degree=p), reg) == p

    @pytest.mark.parametrize(
        "p,q", [(1, 2), (2, 2), (2, 3), (3, 4), (4, 5), (5, 5)]
    )
    def test_sn_bracket_degree(self, p, q):
        """|[P_p, Q_q]_SN| = p + q − 1 (bookkeeping gap fixed in
        this audit)."""
        reg = PropertyRegistry()
        br = sn_bracket(
            PVector("A", degree=p), PVector("B", degree=q)
        )
        assert multivector_degree(br, reg) == p + q - 1

    @pytest.mark.parametrize("p", [2, 3, 5])
    def test_tilde_degrees(self, p):
        """Tilde grading at higher multivector degree: |m| = p and
        |d̃m| = p + 1."""
        from jacopy.packages.poisson import poisson_structure
        from jacopy.packages.poisson.tilde import (
            d_tilde,
            tilde_calculus,
            tilde_multivector_degree,
        )

        reg = PropertyRegistry()
        P = poisson_structure()
        name = tilde_calculus(P).name
        m = PVector(f"m{p}", degree=p)
        assert tilde_multivector_degree(name, m, reg) == p
        assert (
            tilde_multivector_degree(name, d_tilde(P, m), reg)
            == p + 1
        )

    def test_lie_on_multivector_closes(self):
        """GAP CLOSED (was the pinned deferral): L_X P → [X, P]_SN
        for p ≥ 2 (spec 9e, the canonical multivector extension)."""
        reg = PropertyRegistry()
        X = vector_fields("X")[0]
        A3 = PVector("A", degree=3)
        node = Act(CARTAN_TM.lie(X), A3)
        out, steps = tangent_engine(registry=reg).expand(node)
        # L_X A → [X,A]_SN, then the orientation rule canonicalizes
        # to −[A,X]_SN (repr order; (p−1)(q−1) = 0 even → minus).
        assert out == Neg(sn_bracket(A3, X))
        assert any("multivector" in s.rule for s in steps)

    def test_lie_on_wedge_is_derivation(self):
        """Consistency: L_X(Y∧Z) via the new rule + SN wedge-Leibniz
        equals [X,Y]∧Z + Y∧[X,Z]."""
        from jacopy.core.wedge import Wedge
        from jacopy.algebra.lie_bracket_vf import LieBracketVF

        reg = PropertyRegistry()
        X, Y, Z = vector_fields("X Y Z")
        lhs = Act(CARTAN_TM.lie(X), Wedge(Y, Z))
        rhs = Sum(
            Wedge(LieBracketVF(X, Y), Z),
            Wedge(Y, LieBracketVF(X, Z)),
        )
        assert (
            ExpandAndSimplify()
            .prove(
                lhs,
                rhs,
                registry=reg,
                engine=tangent_engine(registry=reg),
            )
            .steps
        )

    @pytest.mark.parametrize(
        "p,q", [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
    )
    def test_sn_graded_antisymmetry_on_decomposables(self, p, q):
        """The graded antisymmetry that BACKS the orientation rule,
        verified mechanically on decomposables:
        [a,b] + (−1)^{(p−1)(q−1)}[b,a] = 0."""
        from jacopy.core.expr import Product
        from jacopy.core.wedge import Wedge

        reg = PropertyRegistry()
        vs = vector_fields("X1 X2 X3 Y1 Y2 Y3")
        a = vs[0] if p == 1 else Wedge(*vs[:p])
        b = vs[3] if q == 1 else Wedge(*vs[3 : 3 + q])
        sign = (-1) ** ((p - 1) * (q - 1))
        combo = Sum(
            sn_bracket(a, b),
            Product(Integer(sign), sn_bracket(b, a)),
        )
        assert (
            ExpandAndSimplify()
            .prove(
                combo,
                Integer(0),
                registry=reg,
                engine=tangent_engine(registry=reg),
            )
            .steps
        )

    def test_sn_orientation_canonicalizes_opaque_atoms(self):
        """The orientation rule fires on OPAQUE p-vector atoms too
        (where the expansion cannot): [B,A] with repr(B) > repr(A)
        swaps with the graded sign."""
        reg = PropertyRegistry()
        A3, B4 = PVector("A", degree=3), PVector("B", degree=4)
        out, steps = tangent_engine(registry=reg).expand(
            sn_bracket(B4, A3)
        )
        # (p−1)(q−1) = 3·2 even → sign −
        assert out == Neg(sn_bracket(A3, B4))
        assert any("orientation" in s.rule for s in steps)


class TestPairingVectorScalar:
    """2026-08-01: ⟨α, f·Y⟩ → f·⟨α,Y⟩ admitted centrally (surfaced
    by the user-bracket API prototype). Cycle analysis in the rule's
    docstring; the SUM-slot split stays excluded."""

    def test_scalar_pulls_out(self):
        from jacopy.central.tangent.engine import tangent_engine
        from jacopy.core.expr import Product
        from jacopy.core.pairing import Pairing

        reg = PropertyRegistry()
        (f,) = functions("f", registry=reg)
        (al,) = forms("α", degree=1)
        (Yv,) = vector_fields("Y")
        node = Pairing(al, Product(f, Yv))
        out, steps = tangent_engine(registry=reg).expand(node)
        assert out == Product(f, Pairing(al, Yv))

    def test_sum_slot_stays(self):
        """⟨α, X+Y⟩ split stays EXCLUDED (collect_pairings çifti)."""
        from jacopy.central.tangent.engine import tangent_engine
        from jacopy.core.pairing import Pairing

        reg = PropertyRegistry()
        (al,) = forms("α", degree=1)
        Xv, Yv = vector_fields("X Y")
        node = Pairing(al, Sum(Xv, Yv))
        out, _ = tangent_engine(registry=reg).expand(node)
        assert out == node


class TestSubsetReplacement:
    """2026-08-01 (5.E.2b part a, LANDED): subset-REPLACEMENT — a
    cited theorem whose rhs has strictly fewer terms than its Sum
    lhs may rewrite a subset in place (2→1 bracket rearrangements);
    term count strictly decreases, so termination holds."""

    def test_two_to_one_rearrangement_fires(self):
        from jacopy.proof.chain import ProofChain
        from jacopy.proof.step import ProofStep
        from jacopy.proof.theorems import Theorem, TheoremDefinition

        a, b, c, e = vector_fields("a b c e")
        lhs = Sum(a, b)
        rhs = c
        thm = Theorem(
            name="t2to1",
            statement="a + b = c",
            lhs=lhs,
            rhs=rhs,
            proof=ProofChain(
                [ProofStep(lhs, rhs, rule="r", justification="j")]
            ),
            generality="instance",
        )
        rule = TheoremDefinition(thm)
        big = Sum(a, b, e)
        assert rule.matches(big)
        out = rule.rewrite(big)
        assert out == Sum(e, c) or out == Sum(c, e)

    def test_growing_rhs_stays_inert(self):
        """rhs with ≥ as many terms as the lhs subset must NOT
        subset-fire (termination guard)."""
        from jacopy.proof.chain import ProofChain
        from jacopy.proof.step import ProofStep
        from jacopy.proof.theorems import Theorem, TheoremDefinition

        a, b, c, e, g = vector_fields("a b c e g")
        lhs = Sum(a, b)
        rhs = Sum(c, g)
        thm = Theorem(
            name="tgrow",
            statement="s",
            lhs=lhs,
            rhs=rhs,
            proof=ProofChain(
                [ProofStep(lhs, rhs, rule="r", justification="j")]
            ),
            generality="instance",
        )
        rule = TheoremDefinition(thm)
        big = Sum(a, b, e)
        assert not rule.matches(big)
