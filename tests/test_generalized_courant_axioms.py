"""Phase 7.A — LWX Courant axiomatics + Uchino redundancy theorems
(uchino.pdf, math/0204010): [C3], [C4] and the D-formula follow from
[C2] + [C5] + (L) + definitional non-degeneracy/skewness."""

import pytest

from jacopy.core.expr import Integer, Product, Rational, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.algebra.derivation import Act
from jacopy.central.objects import functions
from jacopy.central.algebroid.context import AlgebroidBracket
from jacopy.packages.generalized import (
    CourantD,
    CourantInvarianceDeclaration,
    DLeibnizDefinition,
    courant_context,
    courant_engine,
    prove_anchor_annihilates_d,
    prove_d_pairing_formula,
    prove_left_leibniz_c3,
)
from jacopy.proof.strategies import ProofFailure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    E = courant_context()
    x, y = E.sections("x y")
    return reg, E, x, y, f, h


# ---- the D node and its (L) rule ---------------------------------- #


def test_courant_d_slot_protocol(setup):
    reg, E, x, y, f, h = setup
    D = CourantD(E.name, f)
    assert D.rewritable_slots == (f,)
    assert D.with_slots(h) == CourantD(E.name, h)
    assert D.function == f


def test_d_leibniz_splits_scalar_products(setup):
    reg, E, x, y, f, h = setup
    rule = DLeibnizDefinition(E, reg)
    node = CourantD(E.name, Product(f, h))
    assert rule.matches(node)
    out = rule.rewrite(node)
    assert out == Sum(
        Product(f, CourantD(E.name, h)),
        Product(h, CourantD(E.name, f)),
    )


def test_d_leibniz_inert_on_atoms_and_sums(setup):
    # (L) is the ONLY assumed structure: no ℝ-linearity, no D(c)=0.
    reg, E, x, y, f, h = setup
    rule = DLeibnizDefinition(E, reg)
    assert not rule.matches(CourantD(E.name, f))
    assert not rule.matches(CourantD(E.name, Sum(f, h)))


def test_c5_rewrite_shape(setup):
    reg, E, x, y, f, h = setup
    c5 = CourantInvarianceDeclaration(E)
    start = Act(E.anchor(x), E.metric(y, y))
    assert c5.matches(start)
    out = c5.rewrite(start)
    br = E.bracket(x, y)
    D = CourantD(E.name, E.metric(x, y))
    assert out == Sum(
        E.metric(br, y),
        E.metric(D, y),
        E.metric(y, br),
        E.metric(y, D),
    )


# ---- Uchino Prop 2.1(i): [C3] is redundant ------------------------ #


def test_c3_redundant_closes(setup):
    reg, E, x, y, f, h = setup
    chain, thm = prove_left_leibniz_c3(E, x, y, f, registry=reg)
    assert thm.lhs == E.bracket(x, Product(f, y))
    from jacopy.core.expr import Neg

    assert thm.rhs == Sum(
        Product(f, E.bracket(x, y)),
        Product(Act(E.anchor(x), f), y),
        Neg(Product(E.metric(x, y), CourantD(E.name, f))),
    )
    assert "[C5]" in thm.from_axioms[0]
    assert "(L)" in thm.from_axioms[1]
    rules = [s.rule for s in chain.steps]
    assert any("non-degeneracy" in r for r in rules)
    assert any("[C3] form recognized" in r for r in rules)


def test_c3_honest_fail_without_invariance(setup):
    reg, E, x, y, f, h = setup
    with pytest.raises(ProofFailure, match=r"\[C5\]"):
        prove_left_leibniz_c3(
            E, x, y, f, registry=reg, invariance=False
        )


def test_c3_rejects_declared_right_leibniz(setup):
    reg, E, x, y, f, h = setup
    alg = E.with_declarations("right-leibniz")
    with pytest.raises(ValueError, match="DECLARED"):
        prove_left_leibniz_c3(alg, x, y, f, registry=reg)


# ---- Uchino Prop 2.1(ii): [C4] is redundant ----------------------- #


def test_c4_redundant_closes(setup):
    reg, E, x, y, f, h = setup
    chain, thm = prove_anchor_annihilates_d(
        E, x, y, f, h, registry=reg
    )
    assert thm.lhs == Act(E.anchor(CourantD(E.name, f)), h)
    assert thm.rhs == Integer(0)
    rules = [s.rule for s in chain.steps]
    assert any("strip the generic pairing factor" in r for r in rules)
    assert any("[C2]" in r for r in rules)


# ---- Uchino Prop 2.2: the D-formula is redundant ------------------ #


def test_d_formula_closes(setup):
    reg, E, x, y, f, h = setup
    chain, thm = prove_d_pairing_formula(E, x, y, f, registry=reg)
    assert thm.lhs == E.metric(CourantD(E.name, f), y)
    assert thm.rhs == Product(
        Rational(1, 2), Act(E.anchor(y), f)
    )
    assert any(
        "skew-symmetry" in a for a in thm.from_axioms
    )
    rules = [s.rule for s in chain.steps]
    assert any("skew swap" in r for r in rules)


# ---- Uchino Rem 2.1: (i) ⟹ (L) ------------------------------------ #


def test_rem21_leibniz_from_c3(setup):
    reg, E, x, y, f, h = setup
    from jacopy.packages.generalized import prove_leibniz_from_c3

    chain, thm = prove_leibniz_from_c3(E, x, y, f, h, registry=reg)
    assert thm.lhs == CourantD(E.name, Product(f, h))
    assert thm.rhs == Sum(
        Product(f, CourantD(E.name, h)),
        Product(h, CourantD(E.name, f)),
    )
    # No (L) rule anywhere: the assumption record names only the
    # identity (i) and non-degeneracy.
    joined = " ".join(thm.from_axioms)
    assert "identity (i)" in joined
    assert "(L)" not in joined


# ---- Uchino Rem 2.2: [x,Df] and D ---------------------------------- #


def test_rem22_exact_form(setup):
    reg, E, x, y, f, h = setup
    from jacopy.packages.generalized import prove_bracket_with_d

    chain, thm = prove_bracket_with_d(E, x, y, f, registry=reg)
    D = CourantD(E.name, f)
    assert thm.lhs == Sum(
        E.bracket(x, D), CourantD(E.name, E.metric(x, D))
    )
    assert thm.rhs == CourantD(E.name, Act(E.anchor(x), f))
    joined = " ".join(thm.from_axioms)
    assert "homogeneity" not in joined


def test_rem22_classical_homogeneity_is_derived(setup):
    reg, E, x, y, f, h = setup
    from jacopy.packages.generalized import prove_bracket_with_d

    chain, thm = prove_bracket_with_d(
        E, x, y, f, registry=reg, classical=True
    )
    D = CourantD(E.name, f)
    assert thm.lhs == E.bracket(x, D)
    assert thm.rhs == CourantD(E.name, E.metric(x, D))
    assert any("DERIVED" in a and "homogeneity" in a for a in thm.from_axioms)


# ---- Dorfman-form redundancies (Rem 2.3, unwritten in the paper) -- #


def test_dorfman_c3_redundant():
    from jacopy.central.objects import functions as _fns
    from jacopy.packages.generalized import (
        dorfman_axiom_context,
        prove_dorfman_c3_redundant,
    )

    reg = PropertyRegistry()
    (f,) = _fns("f", registry=reg)
    E = dorfman_axiom_context()
    x, y = E.sections("x y")
    chain, thm = prove_dorfman_c3_redundant(
        E, x, y, f, registry=reg
    )
    # the defect-free Leibniz rule: NO D-term on the right side
    assert "D" not in thm.rhs._repr_inner()
    assert "[C'5]" in thm.from_axioms[0]


def test_dorfman_c2_redundant():
    from jacopy.central.objects import functions as _fns
    from jacopy.packages.generalized import (
        dorfman_axiom_context,
        prove_dorfman_c2_redundant,
    )

    reg = PropertyRegistry()
    (f,) = _fns("f", registry=reg)
    E = dorfman_axiom_context()
    x, y, z = E.sections("x y z")
    chain, thm = prove_dorfman_c2_redundant(
        E, x, y, z, f, registry=reg
    )
    assert thm.lhs == Act(E.anchor(E.bracket(x, y)), f)
    rules = [s.rule for s in chain.steps]
    assert any("[C'1] instance cited" in r for r in rules)
    assert any("agreement on generators" in r for r in rules)


def test_dorfman_c2_rejects_declared_anchor_morphism():
    from jacopy.central.objects import functions as _fns
    from jacopy.packages.generalized import (
        dorfman_axiom_context,
        prove_dorfman_c2_redundant,
    )

    reg = PropertyRegistry()
    (f,) = _fns("f", registry=reg)
    E = dorfman_axiom_context().with_declarations(
        "anchor-morphism"
    )
    x, y, z = E.sections("x y z")
    with pytest.raises(ValueError, match="DECLARED"):
        prove_dorfman_c2_redundant(E, x, y, z, f, registry=reg)


def test_theorems_carry_full_assumption_record(setup):
    reg, E, x, y, f, h = setup
    _, thm = prove_left_leibniz_c3(E, x, y, f, registry=reg)
    # every assumption is [C5]/(L)/non-degeneracy — no coboundary
    # pairing, no declared Leibniz anywhere in the record
    joined = " ".join(thm.from_axioms)
    assert "coboundary" not in joined
    assert "right-leibniz" not in joined.lower()
