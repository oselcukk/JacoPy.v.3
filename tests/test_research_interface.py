"""The research interface (Phase 8 bricks): sections, block
matrices, engine assembly, the axiom suite and the antisymmetriser —
exercised on the two twists of examples/bracket_twist_walkthrough."""

import pytest

from jacopy.core.expr import Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, frame, functions, vector_fields
from jacopy.central.tangent.exterior import d
from jacopy.packages.drinfeld.double import (
    canonical_pairing,
    dorfman_double,
    nambu_double,
)
from jacopy.packages.drinfeld.examples import (
    boxtimes,
    exceptional_courant_bracket,
    exceptional_pairing_five,
    exceptional_pairing_two,
)
from jacopy.packages.poisson.nambu import nambu_structure
from jacopy.packages.poisson.tilde import _normalized_by
from jacopy.research import (
    AlgebroidData,
    AxiomSuite,
    BlockMatrix,
    Bracket,
    Map,
    SectionType,
    antisymmetrize,
    assemble_engine,
    assembly_report,
    swap_indices,
)


@pytest.fixture()
def gt():
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    U, V, W = vector_fields("U V W")
    om, et, mu = forms("ω η μ", degree=1)
    N = nambu_structure("θ", p=1)
    T = SectionType.generalized_tangent()
    return reg, f, h, U, V, W, om, et, mu, N, T


# ---- sections and block matrices ---------------------------------- #


def test_section_arithmetic_and_type_checks(gt):
    reg, f, h, U, V, W, om, et, mu, N, T = gt
    e = T.section(U, om)
    assert repr(T) == "TM ⊕ Λ¹T*M"
    assert (e + e - e) == T.section(
        Sum(Sum(U, U), Neg(U)), Sum(Sum(om, om), Neg(om))
    )
    assert e.scale(f) == T.section(Product(f, U), Product(f, om))
    with pytest.raises(ValueError):
        T.section(U)
    with pytest.raises(TypeError):
        e + SectionType.exceptional().section(U, om, om)


def test_unipotent_inverse_is_derived_and_cancels(gt):
    reg, f, h, U, V, W, om, et, mu, N, T = gt
    Psi = BlockMatrix(T, [[1, Map(N.sharp_vf, "θ♯")], [0, 1]])
    inv = Psi.inverse()
    assert inv[0, 1].name == "−θ♯" and inv[1, 1].is_identity
    e = T.section(U, om)
    back = inv(Psi(e))
    eng = assemble_engine(*back, registry=reg, structures=(N,))
    assert _normalized_by(eng, Sum(back[0], Neg(U)), reg) == Integer(0)
    assert back[1] == om
    with pytest.raises(ValueError):
        BlockMatrix(T, [[1, 1], [1, 1]]).inverse()


def test_exceptional_matrix_product_and_inverse():
    T3 = SectionType.exceptional()
    N3, N6 = nambu_structure("Π₃", p=2), nambu_structure("Π₆", p=5)
    P3, P6 = Map(N3.sharp_vf, "Π₃"), Map(N6.sharp_vf, "Π₆")
    BX = Map(lambda w: boxtimes(N3, w), "Π₃⊛Π₃")
    Psi = BlockMatrix(T3, [[1, P3, P6 + BX], [0, 1, 0], [0, 0, 1]])
    assert Psi.is_unipotent
    prod = Psi.inverse() @ Psi
    assert all(prod[i, i].is_identity for i in range(3))
    assert all(prod[i, j].is_zero for i in range(3) for j in range(3) if i != j)


# ---- engine assembly ---------------------------------------------- #


def test_assembly_refuses_unknown_structures(gt):
    reg, f, h, U, V, W, om, et, mu, N, T = gt
    with pytest.raises(ValueError, match="undeclared structure"):
        assemble_engine(N.sharp_vf(om), registry=reg)
    eng = assemble_engine(N.sharp_vf(om), registry=reg, structures=(N,))
    assert any("tilde-calculus engine" in line for line in assembly_report(eng))
    assert not any("DECLARED" in line for line in assembly_report(eng))


# ---- axiom suite + transport --------------------------------------- #


def _standard(T):
    return AlgebroidData(
        bracket=Bracket.from_components(T, dorfman_double, name="Dorfman"),
        anchor=lambda e: e[0],
        pairing=lambda a, b: (canonical_pairing(*a, *b),),
        D=lambda p: T.section(Integer(0), d(p)),
        name="standard",
    )


def test_suite_closes_dorfman_and_its_pi_twist(gt):
    reg, f, h, U, V, W, om, et, mu, N, T = gt
    std = _standard(T)
    e1, e2 = T.section(U, om), T.section(V, et)
    assert AxiomSuite(std, registry=reg, probe=h, structures=(N,)).run(e1, e2, f).all_closed
    Psi = BlockMatrix(T, [[1, Map(N.sharp_vf, "θ♯")], [0, 1]], name="Ψ")
    twisted = std.transport(Psi)
    rep = AxiomSuite(twisted, registry=reg, probe=h, structures=(N,)).run(e1, e2, f)
    assert rep.all_closed
    # the twisted bracket IS the Nambu double up to the derived twist R′
    # in the vector part; the form part agrees outright
    tw = twisted.bracket(e1, e2)
    nb = nambu_double(N, U, om, V, et)
    eng = assemble_engine(tw[1], nb[1], registry=reg, structures=(N,))
    assert _normalized_by(eng, Sum(tw[1], Neg(nb[1])), reg) == Integer(0)


def test_suite_reports_a_residual_instead_of_lying(gt):
    reg, f, h, U, V, W, om, et, mu, N, T = gt
    broken = AlgebroidData(
        bracket=Bracket.from_components(T, dorfman_double),
        anchor=lambda e: Sum(e[0], N.sharp_vf(e[1])),   # wrong anchor for Dorfman
    )
    rep = AxiomSuite(broken, registry=reg, probe=h, structures=(N,)).right_leibniz(
        T.section(U, om), T.section(V, et), f
    )
    assert not rep.all_closed
    assert any(r.status == "RESIDUAL" and r.residual is not None for r in rep.results)


def test_exceptional_rotation_through_the_facade():
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    U, V = vector_fields("U V")
    om2, et2 = forms("ω₂ η₂", degree=2)
    om5, et5 = forms("ω₅ η₅", degree=5)
    N3, N6 = nambu_structure("Π₃", p=2), nambu_structure("Π₆", p=5)
    T3 = SectionType.exceptional()
    data = AlgebroidData(
        bracket=Bracket.from_components(T3, exceptional_courant_bracket),
        anchor=lambda e: e[0],
        pairing=lambda a, b: (
            exceptional_pairing_two(a[0], a[1], b[0], b[1]),
            exceptional_pairing_five(*a, *b),
        ),
        D=lambda p2, p5: T3.section(Integer(0), d(p2), d(p5)),
    )
    Psi = BlockMatrix(
        T3,
        [[1, Map(N3.sharp_vf, "Π₃"), Map(N6.sharp_vf, "Π₆") + Map(lambda w: boxtimes(N3, w), "⊛")],
         [0, 1, 0], [0, 0, 1]],
        name="Ψ_Π",
    )
    x1, x2 = T3.section(U, om2, om5), T3.section(V, et2, et5)
    suite = AxiomSuite(data.transport(Psi), registry=reg, probe=h, structures=(N3, N6))
    assert suite.right_leibniz(x1, x2, f).all_closed
    assert suite.symmetric_part(x1, x2).all_closed


@pytest.mark.skipif(
    not __import__("os").environ.get("JACOPY_RUN_SLOW"),
    reason="~100 s closure; set JACOPY_RUN_SLOW=1 to run",
)
def test_twisted_dorfman_jacobi_closes_for_any_bivector(gt):
    reg, f, h, U, V, W, om, et, mu, N, T = gt
    Psi = BlockMatrix(T, [[1, Map(N.sharp_vf, "θ♯")], [0, 1]], name="Ψ")
    twisted = _standard(T).transport(Psi)
    rep = AxiomSuite(twisted, registry=reg, probe=h, structures=(N,)).jacobi(
        T.section(U, om), T.section(V, et), T.section(W, mu)
    )
    assert rep.all_closed


def test_derived_twist_vanishes_under_declared_poisson_condition(gt):
    reg, f, h, U, V, W, om, et, mu, N, T = gt
    from jacopy.packages.drinfeld.twist import r_twist

    r = r_twist(N, om, et)
    with_fi = assemble_engine(r, registry=reg, structures=(N,), declare_fi=True)
    without = assemble_engine(r, registry=reg, structures=(N,))
    assert _normalized_by(with_fi, r, reg) == Integer(0)
    assert _normalized_by(without, r, reg) != Integer(0)


# ---- antisymmetriser ---------------------------------------------- #


def test_antisymmetrize_is_alternating_in_the_named_indices():
    fr = frame()
    co = fr.dual()
    N3 = nambu_structure("Π₃", p=2)
    reg = PropertyRegistry()
    from jacopy.algorithms.simplify import simplify

    a1, a2, a3 = "a1", "a2", "a3"
    comp = MultiEval(N3.pi, co.field(a1), co.field(a2), co.field(a3), alternating=True, slot_kind="covector")
    sym = Product(comp, comp)                      # symmetric in nothing yet
    anti = antisymmetrize(sym, (a1, a2, a3), unit_weight=True)
    swapped = swap_indices(anti, a1, a2)
    assert simplify(Sum(anti, swapped), reg) == Integer(0)


# ---- the exceptional 5-form Jacobi, closed by the linear split ------ #


def _exceptional_triple():
    reg = PropertyRegistry()
    U, V, W = vector_fields("U V W")
    om2, et2, ze2 = forms("ω₂ η₂ ζ₂", degree=2)
    om5, et5, ze5 = forms("ω₅ η₅ ζ₅", degree=5)
    slots5 = vector_fields("X₁ X₂ X₃ X₄ X₅")
    N3 = nambu_structure("Π₃", p=2)
    return reg, N3, (U, om2, om5, V, et2, et5, W, ze2, ze5), slots5


def test_exceptional_jacobi_five_splits_and_cross_part_closes():
    from jacopy.packages.drinfeld.examples import prove_exceptional_jacobi_five

    reg, N3, args, slots5 = _exceptional_triple()
    chain, assumptions = prove_exceptional_jacobi_five(
        N3, *args, slots5, registry=reg, cite_dorfman=True
    )
    rules = [s.rule for s in chain.steps]
    assert any("linear split" in r for r in rules)
    assert any("cross Jacobiator" in r for r in rules)
    assert "CITED" in assumptions[0]


@pytest.mark.parametrize("p", [2, 4])
def test_top_slot_jacobi_closes_for_even_p(p):
    # the family TM ⊕ Λᵖ ⊕ Λ^{2p+1}: split identity and cross part close
    # at form level for EVEN p (the Dorfman leg is cited here)
    from jacopy.packages.drinfeld.examples import prove_top_slot_jacobi

    reg = PropertyRegistry()
    U, V, W = vector_fields("U V W")
    a, b, c = forms("α β γ", degree=p)
    A, B, C = forms("A B C", degree=2 * p + 1)
    slots = vector_fields(" ".join(f"Y{i}" for i in range(2 * p + 1)))
    N = nambu_structure("Π", p=p)
    chain, assumptions = prove_top_slot_jacobi(
        N, U, a, A, V, b, B, W, c, C, slots, registry=reg, cite_dorfman=True
    )
    assert [s.rule.split(" (")[0] for s in chain.steps[:2]] == [
        "top-slot Jacobiator = Dorfman Jacobiator + cross Jacobiator",
        "cross Jacobiator normalizes to 0",
    ]


@pytest.mark.parametrize("p", [1, 3])
def test_top_slot_jacobi_obstructed_for_odd_p(p):
    # for ODD p the single cross-term bracket is NOT Leibniz: the cross
    # Jacobiator is ∓2·dη_p ∧ ι_W dω_p (independently confirmed by brute
    # force on 3 slots at p = 1) — the prover fails honestly
    from jacopy.packages.drinfeld.examples import prove_top_slot_jacobi
    from jacopy.proof.strategies import ProofFailure

    reg = PropertyRegistry()
    U, V, W = vector_fields("U V W")
    a, b, c = forms("α β γ", degree=p)
    A, B, C = forms("A B C", degree=2 * p + 1)
    slots = vector_fields(" ".join(f"Y{i}" for i in range(2 * p + 1)))
    N = nambu_structure("Π", p=p)
    with pytest.raises(ProofFailure, match="cross Jacobiator") as info:
        prove_top_slot_jacobi(
            N, U, a, A, V, b, B, W, c, C, slots, registry=reg, cite_dorfman=True
        )
    assert "ι_W" in str(info.value)
@pytest.mark.skipif(
    not __import__("os").environ.get("JACOPY_RUN_SLOW"),
    reason="~35 s closure; set JACOPY_RUN_SLOW=1 to run",
)
def test_exceptional_jacobi_five_closes_in_full():
    from jacopy.packages.drinfeld.examples import prove_exceptional_jacobi_five

    reg, N3, args, slots5 = _exceptional_triple()
    chain, assumptions = prove_exceptional_jacobi_five(
        N3, *args, slots5, registry=reg
    )
    assert len(chain.steps) > 100          # the Dorfman leg is proven, not cited
    assert "CITED" not in " ".join(assumptions)


# ---- 2026-09-09 audit pins (F1-F4) ------------------------------- #


def test_vector_wedge_is_not_proved_commutative():
    from jacopy.core.wedge import Wedge
    from jacopy.proof import show_equal
    from jacopy.proof.strategies import ProofFailure

    reg = PropertyRegistry()
    X, Y = vector_fields("X Y")
    lhs, rhs = Wedge(Y, X), Wedge(X, Y)
    with pytest.raises(ProofFailure):
        show_equal(lhs, rhs, registry=reg, engine=assemble_engine(lhs, rhs, registry=reg))
    a, b = forms("a b", degree=1)
    eng = assemble_engine(Wedge(b, a), registry=reg)
    assert _normalized_by(eng, Sum(Wedge(b, a), Wedge(a, b)), reg) == Integer(0)


def test_suite_evaluation_symbols_are_fresh():
    from jacopy.algebra.derivation import Act
    from jacopy.core.pairing import Pairing
    from jacopy.central.objects.interior import Interior

    reg = PropertyRegistry()
    (X,) = vector_fields("ξ₁")                      # the suite's own slot name
    (tau,) = forms("tau", degree=3)
    (alpha,) = forms("alpha", degree=2)
    (s,) = forms("s", degree=1)
    T = SectionType.generalized_tangent(2)
    data = AlgebroidData(
        bracket=Bracket(T, lambda u, v: T.section(
            Integer(0), Product(Pairing(s, u[0]), Pairing(s, v[0]), Act(Interior(X), tau)))),
        anchor=lambda e: Integer(0),
        pairing=lambda u, v: (Integer(0),),
        D=lambda p: T.zero(),
    )
    e = T.section(X, alpha)
    assert not AxiomSuite(data, registry=reg).symmetric_part(e, e).all_closed
    # a user-supplied probe that occurs in the checked expression is refused
    (fs,) = functions("fs", registry=reg)
    with pytest.raises(ValueError, match="probe"):
        AxiomSuite(data, registry=reg, probe=fs)._fresh_probe({"fs"})


def test_zero_products_vanish_inside_sharp_slots():
    reg = PropertyRegistry()
    N = nambu_structure("P", p=2)
    node = N.sharp_vf(Product(Integer(0), Integer(0)))
    assert _normalized_by(assemble_engine(node, registry=reg, structures=(N,)), node, reg) == Integer(0)


def test_assembly_report_lives_on_the_engine():
    import gc, weakref

    from jacopy.research.engine_assembly import _REPORTS

    eng = assemble_engine(Integer(0))
    assert assembly_report(eng) and eng.assembly_report == assembly_report(eng)
    assert not _REPORTS


def test_metric_invariance_needs_the_r_valued_action_on_the_exceptional_bundle():
    from jacopy.core.wedge import Wedge
    from jacopy.algebra.derivation import Act
    from jacopy.central.tangent.cartan import L as Lie

    reg = PropertyRegistry()
    U, V, W = vector_fields("U V W")
    om2, et2, ze2 = forms("ω₂ η₂ ζ₂", degree=2)
    om5, et5, ze5 = forms("ω₅ η₅ ζ₅", degree=5)
    N3, N6 = nambu_structure("Π₃", p=2), nambu_structure("Π₆", p=5)
    T3 = SectionType.exceptional()
    pairing = lambda a, b: (exceptional_pairing_two(a[0], a[1], b[0], b[1]), exceptional_pairing_five(*a, *b))
    good = AlgebroidData(
        bracket=Bracket.from_components(T3, exceptional_courant_bracket), anchor=lambda e: e[0],
        pairing=pairing, D=lambda p2, p5: T3.section(Integer(0), d(p2), d(p5)),
        lie_R=lambda e, r1, r4: (Lie(e[0], r1), Sum(Lie(e[0], r4), Neg(Wedge(r1, d(e[1]))))),
    )
    naive = AlgebroidData(
        bracket=good.bracket, anchor=good.anchor, pairing=pairing, D=good.D,
        lie_R=lambda e, r1, r4: (Lie(e[0], r1), Lie(e[0], r4)),     # plain ℒ_U on both: wrong
    )
    x1, x2, x3 = T3.section(U, om2, om5), T3.section(V, et2, et5), T3.section(W, ze2, ze5)
    assert AxiomSuite(good, registry=reg, structures=(N3, N6)).metric_invariance(x1, x2, x3).all_closed
    rep = AxiomSuite(naive, registry=reg, structures=(N3, N6)).metric_invariance(x1, x2, x3)
    assert not rep.all_closed and rep.results[1].residual is not None


@pytest.mark.skipif(
    not __import__("os").environ.get("JACOPY_RUN_VERY_SLOW"),
    reason="~14 min closure (4-form component 765 s); set JACOPY_RUN_VERY_SLOW=1",
)
def test_rotated_exceptional_metric_invariance_closes_with_a_large_budget():
    from jacopy.core.wedge import Wedge
    from jacopy.central.tangent.cartan import L as Lie

    reg = PropertyRegistry()
    U, V, W = vector_fields("U V W")
    om2, et2, ze2 = forms("ω₂ η₂ ζ₂", degree=2)
    om5, et5, ze5 = forms("ω₅ η₅ ζ₅", degree=5)
    N3, N6 = nambu_structure("Π₃", p=2), nambu_structure("Π₆", p=5)
    T3 = SectionType.exceptional()
    data = AlgebroidData(
        bracket=Bracket.from_components(T3, exceptional_courant_bracket), anchor=lambda e: e[0],
        pairing=lambda a, b: (exceptional_pairing_two(a[0], a[1], b[0], b[1]), exceptional_pairing_five(*a, *b)),
        D=lambda p2, p5: T3.section(Integer(0), d(p2), d(p5)),
        lie_R=lambda e, r1, r4: (Lie(e[0], r1), Sum(Lie(e[0], r4), Neg(Wedge(r1, d(e[1]))))),
    )
    Psi = BlockMatrix(
        T3,
        [[1, Map(N3.sharp_vf, "Π₃"), Map(N6.sharp_vf, "Π₆") + Map(lambda w: boxtimes(N3, w), "⊛")],
         [0, 1, 0], [0, 0, 1]],
        name="Ψ_Π",
    )
    x1, x2, x3 = T3.section(U, om2, om5), T3.section(V, et2, et5), T3.section(W, ze2, ze5)
    suite = AxiomSuite(data.transport(Psi), registry=reg, structures=(N3, N6), expand_max_steps=200000)
    assert suite.metric_invariance(x1, x2, x3).all_closed
