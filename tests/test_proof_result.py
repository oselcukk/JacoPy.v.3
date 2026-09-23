"""Faz 8 step 3a — ProofResult: goal + status, assumptions (requires)
kept apart from derivation (provenance), legacy assumptions never
dropped, CLOSED always "under" its assumptions, instance generality
never upgraded, INVALID without running, BUDGET on exhaustion, the
version from the package metadata, and the old (chain, theorem)
returns wrapped rather than broken."""

import pytest

from jacopy.central.algebroid import algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.operators import Jacobiator
from jacopy.central.algebroid.theorems import prove_anchor_morphism
from jacopy.central.objects import Bundle, forms, functions, vector_fields
from jacopy.central.tangent.engine import tangent_engine
from jacopy.central.tangent.exterior import d
from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import ProofChain, ProofStep
from jacopy.proof.result import (
    Assumption,
    Budget,
    Goal,
    ProofResult,
    Provenance,
    check_goal,
    package_version,
    prove,
)
from jacopy.proof.theorems import Theorem


@pytest.fixture()
def alg():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("lie",))
    u, v, w = E.sections("u v w")
    return reg, f, E, u, v, w


def test_closed_under_declared_assumptions_and_definitions_apart(alg):
    reg, f, E, u, v, w = alg
    chain, thm = prove_anchor_morphism(E, u, v, f, registry=reg)     # the old return shape, untouched
    res = ProofResult.from_theorem(thm)
    assert res.closed and res.status == "CLOSED"
    assert res.goal == Goal(thm.lhs, thm.rhs)
    declared = {a.name.split(":")[0] for a in res.declared}
    assert any("Leibniz" in n for n in declared)                       # a declared axiom fired
    assert all(a.owner is E for a in res.declared)
    assert any(a.kind == "structure" and a.owner is E for a in res.requires)
    assert res.provenance and all(p.kind in ("definition", "engine-step", "canonicalization", "theorem") for p in res.provenance)
    # a definition is provenance, never an assumption
    names_req = {a.name for a in res.requires}
    assert not any("definition" in n.lower() for n in names_req if "from_axioms" not in n)
    text = res.summary()
    assert text.startswith("CLOSED") and "under the assumptions below" in text and "requires declared" in text
    assert res.owner is E and res.version == package_version()


def test_residual_is_not_a_disproof():
    reg = PropertyRegistry()
    E2 = algebroid("E", Bundle("E"))                                  # nothing declared
    u, v, w = E2.sections("u v w")
    res = prove(algebroid_engine(E2, registry=reg), Jacobiator("E", u, v, w), registry=reg)
    assert res.status == "RESIDUAL" and not res.closed
    assert res.residual is not None and res.residual != Integer(0)
    assert not res.declared                                            # no axiom was used
    assert "NOT a disproof" in res.summary()
    with pytest.raises(ValueError):
        ProofResult(Goal(u, u), "RESIDUAL")                            # a residual result must record it
    with pytest.raises(ValueError):
        ProofResult(Goal(u, u), "DISPROVED")                           # no such status


def test_prove_closes_with_declared_jacobi(alg):
    reg, f, E, u, v, w = alg
    res = prove(algebroid_engine(E, registry=reg), Jacobiator("E", u, v, w), registry=reg)
    assert res.closed
    assert res.declared and all(a.owner is E for a in res.declared)
    assert res.chain.initial == Jacobiator("E", u, v, w) and res.chain.final == Integer(0)
    assert res.budget == Budget(len(res.chain), 1024, False)
    # rhs is normalized too, and the chain runs lhs → nf ← rhs
    res2 = prove(algebroid_engine(E, registry=reg), Jacobiator("E", u, v, w), Sum(Integer(0), Integer(0)), registry=reg)
    assert res2.closed and res2.chain.final == Sum(Integer(0), Integer(0))


def test_legacy_assumptions_are_kept_not_dropped(alg):
    reg, f, E, u, v, w = alg
    X, Y = vector_fields("X Y")
    (om,) = forms("ω", degree=1)
    # a hand-built axiom step records no role: kept as a legacy assumption
    hand = ProofStep(om, Integer(0), rule="declared by hand: ω = 0", provenance_tag="axiom")
    res = ProofResult.from_chain(ProofChain([hand]), Goal(om, Integer(0)))
    assert res.closed and res.legacy and res.legacy[0].name == "declared by hand: ω = 0"
    assert "requires legacy" in res.summary()
    # a textual from_axioms entry no step accounts for is a legacy assumption;
    # one a step accounts for is not duplicated
    chain, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
    res = ProofResult.from_theorem(thm)
    assert not any("from_axioms" in a.name for a in res.legacy)
    thm2 = Theorem(name="t2", statement="s", lhs=thm.lhs, rhs=thm.rhs, proof=thm.proof,
                   from_axioms=thm.from_axioms + ("a hypothesis nobody recorded",), owner=E)
    res2 = ProofResult.from_theorem(thm2)
    assert [a for a in res2.legacy if "nobody recorded" in a.name]
    # a record whose chain does not literally run lhs → rhs is marked, not trusted silently
    thm3 = Theorem(name="t3", statement="s", lhs=X, rhs=Y, proof=thm.proof)
    res3 = ProofResult.from_theorem(thm3)
    assert any("does not literally run" in a.name for a in res3.legacy)
    assert res3.status == "CLOSED"                                     # the record asserts closure …
    assert res3.legacy                                                 # … and the result says on what


def test_instance_generality_is_never_upgraded(alg):
    reg, f, E, u, v, w = alg
    chain, thm = prove_anchor_morphism(E, u, v, f, registry=reg)
    inst = Theorem(name="i", statement="s", lhs=thm.lhs, rhs=thm.rhs, proof=thm.proof, generality="instance", owner=E)
    res = ProofResult.from_theorem(inst)
    assert res.generality == "instance"
    assert any(a.kind == "scope" for a in res.requires)
    assert "not a general theorem" in res.summary()
    res2 = prove(algebroid_engine(E, registry=reg), Jacobiator("E", u, v, w), registry=reg, generality="instance")
    assert res2.generality == "instance" and any(a.kind == "scope" for a in res2.requires)
    with pytest.raises(ValueError):
        ProofResult(Goal(u, u), "CLOSED", generality="everything")


def test_invalid_goal_is_refused_without_running():
    reg = PropertyRegistry()
    (om,) = forms("ω", degree=1)
    (B,) = forms("B", degree=2)
    (X,) = vector_fields("X")
    assert "degrees" in check_goal(om, B, reg)
    assert "kinds" in check_goal(X, om, reg)
    assert check_goal(om, Integer(0), reg) is None and check_goal(om, Sum(om, om), reg) is None
    assert "different known types" in check_goal(Sum(om, B), Integer(0), reg)
    res = prove(tangent_engine(registry=reg), om, B, registry=reg)
    assert res.status == "INVALID" and res.chain is None and res.budget.steps == 0
    assert "reason:" in res.summary() and "degrees" in res.reason
    with pytest.raises(ValueError):
        ProofResult(Goal(om, om), "CLOSED", residual=om)


def test_budget_exhaustion_is_reported_as_budget():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    X, Y = vector_fields("X Y")
    from jacopy.algebra.lie_bracket_vf import LieBracketVF
    from jacopy.algebra.derivation import Act

    # a generous unfolding: [X,[X,[X,Y]]](f) needs several definitional steps
    expr = Act(LieBracketVF(X, LieBracketVF(X, LieBracketVF(X, Y))), f)
    full = prove(tangent_engine(registry=reg), expr, expr, registry=reg)
    assert full.closed and full.budget.steps > 2
    tight = prove(tangent_engine(registry=reg), expr, Integer(0), registry=reg, max_steps=1)
    assert tight.status == "BUDGET" and tight.budget.exhausted and tight.budget.max_steps == 1
    assert tight.residual is not None and "(exhausted)" in tight.summary()


def test_from_chain_checks_the_goal_and_classifies_canonicalization():
    (om,) = forms("ω", degree=1)
    X, Y = vector_fields("X Y")
    with pytest.raises(ValueError, match="starts at"):
        ProofResult.from_chain(ProofChain([ProofStep(X, Y, rule="r")]), Goal(om, Y))
    s1 = ProofStep(Sum(X, Neg(X)), Integer(0), rule="simplify")
    res = ProofResult.from_chain(ProofChain([s1]), Goal(Sum(X, Neg(X)), Integer(0)))
    assert res.closed and res.provenance == (Provenance("canonicalization", "simplify"),)
    assert res.requires == ()
    assert "assumption-free" in res.summary()
    # an empty chain closes only a trivial goal
    assert ProofResult.from_chain(ProofChain(), Goal(X, X)).closed
    assert ProofResult.from_chain(ProofChain(), Goal(X, Y)).status == "RESIDUAL"
    assert Assumption("declared", "a") != Assumption("legacy", "a")
    with pytest.raises(ValueError):
        Assumption("guess", "a")


def test_version_comes_from_the_package_metadata_not_git():
    import jacopy

    v = package_version()
    assert isinstance(v, str) and v and v == jacopy.__version__
