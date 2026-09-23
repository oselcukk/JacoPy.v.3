"""Faz 8 step 3b — structural theorem requirements and checked
citations: a theorem carries the assumptions its chain used (and,
transitively, those of the theorems it cites); a citation fires only
where the engine's structures meet them and is reported otherwise; a
legacy record (unverified requirements) is never cited silently and
never enters a TheoremBook as a verified theorem."""

import pytest

from jacopy.central.algebroid import algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.operators import Jacobiator
from jacopy.central.algebroid.theorems import prove_anchor_morphism
from jacopy.central.objects import Bundle, functions
from jacopy.core.expr import Integer, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import AmbiguousOwnerError, ExpansionEngine, OwnerScope, ProofChain, ProofResult
from jacopy.proof.result import prove
from jacopy.proof.theorems import (
    LegacyTheoremError,
    Theorem,
    TheoremBook,
    TheoremDefinition,
    check_citation,
    cite,
)


def _alg(*declare, bundle="E"):
    return algebroid("E", Bundle(bundle), declare=declare)


@pytest.fixture()
def world():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E_lie = _alg("lie")                                              # right-leibniz + antisymmetric + jacobi
    E_sub = _alg("right-leibniz", "antisymmetric")                   # no jacobi
    E_sup = E_lie.with_declarations("metric-invariance")             # strictly more
    u, v, w = E_lie.sections("u v w")
    return reg, f, E_lie, E_sub, E_sup, u, v, w


def _jacobi_theorem(E, u, v, w, reg, name="jacobi_E"):
    res = prove(algebroid_engine(E, registry=reg), Jacobiator("E", u, v, w), registry=reg)
    assert res.closed
    return Theorem(name=name, statement="J^E = 0", lhs=res.goal.lhs, rhs=res.goal.rhs, proof=res.chain, owner=E).with_structural_requires()


def test_structural_requires_are_derived_not_parsed(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    thm = _jacobi_theorem(E_lie, u, v, w, reg)
    assert thm.requires_verified and not thm.legacy
    kinds = {a.kind for a in thm.requires}
    assert kinds == {"declared", "structure"}
    assert all(a.owner is E_lie for a in thm.requires)
    assert any("Jacobi" in a.name for a in thm.requires if a.kind == "declared")
    # the pre-3b shape: nothing derived, hence legacy — and the reason says what to do
    raw = Theorem(name="raw", statement="s", lhs=thm.lhs, rhs=thm.rhs, proof=thm.proof, owner=E_lie)
    assert raw.legacy and not raw.requires and "with_structural_requires" in raw.unverified_reasons()[0]
    # a migrated prover of the algebroid layer records its requirements
    chain, am = prove_anchor_morphism(E_lie, u, v, f, registry=reg)
    assert am.requires_verified and am.requires
    assert {a.owner for a in am.requires if a.kind == "declared"} == {E_lie}
    with pytest.raises(TypeError):
        Theorem(name="x", statement="s", lhs=thm.lhs, rhs=thm.rhs, proof=thm.proof, requires=("text",))


def test_missing_prerequisite_blocks_the_citation_and_reports(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    thm = _jacobi_theorem(E_lie, u, v, w, reg)
    eng_sub = algebroid_engine(E_sub, registry=reg)
    before = len(eng_sub.definitions)
    eng_sub.register(TheoremDefinition(thm))                        # not an error: inert + reported
    bound = eng_sub.definitions[-1]
    assert len(eng_sub.definitions) == before + 1 and not bound.citable
    assert any("jacobi" in r for r in bound.blocked) and any("weaker structure" in r for r in bound.blocked)
    assert any("NOT citable" in line for line in eng_sub.assembly_report)
    assert "E" in eng_sub.owners and eng_sub.owners["E"] is E_sub  # the blocked citation claims no owner
    assert prove(eng_sub, Jacobiator("E", u, v, w), registry=reg).status == "RESIDUAL"
    chk = check_citation(thm, eng_sub)
    assert not chk.ok and chk.missing and "not citable" in str(chk)


def test_sub_structure_theorems_are_citable_in_the_super_structure(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    thm = _jacobi_theorem(E_lie, u, v, w, reg)
    eng_sup = algebroid_engine(E_sup, registry=reg)                  # E_lie ⊑ E_sup
    eng_sup.register(TheoremDefinition(thm))
    assert eng_sup.definitions[-1].citable and eng_sup.owners["E"] is E_sup
    assert check_citation(thm, eng_sup).ok
    assert not any("NOT citable" in line for line in eng_sup.assembly_report)
    # and the 2c policy agrees: a sub-structure's item passes, a super-structure's does not
    scope = OwnerScope()
    scope.bind(E_sup)
    assert scope.check([thm]) == {"E": E_sup}
    scope2 = OwnerScope()
    scope2.bind(E_sub)
    with pytest.raises(AmbiguousOwnerError):
        scope2.check([thm])


def test_right_name_on_a_different_structure_is_not_enough(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    thm = _jacobi_theorem(E_lie, u, v, w, reg)
    E_other = _alg("lie", bundle="E′")                               # same name, same axioms, another bundle
    eng = algebroid_engine(E_other, registry=reg)
    eng.register(TheoremDefinition(thm))
    assert not eng.definitions[-1].citable
    assert any("different" in r for r in eng.definitions[-1].blocked)
    # an independent context never sees it either
    F = algebroid("F", Bundle("F"), declare=("lie",))
    engF = algebroid_engine(F, registry=reg)
    engF.register(TheoremDefinition(thm))
    assert not engF.definitions[-1].citable and any("no structure named 'E'" in r for r in engF.definitions[-1].blocked)


def test_indirect_prerequisites_propagate_through_citations(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    chain, thm_a = prove_anchor_morphism(E_lie, u, v, f, registry=reg)   # needs E_lie's axioms (jacobi among them)
    # thm_b: proved in E_sup by CITING thm_a (E_sup has no anchor-morphism declaration)
    eng = algebroid_engine(E_sup, registry=reg)
    eng.register(TheoremDefinition(thm_a))
    assert eng.definitions[-1].citable
    res = prove(eng, thm_a.lhs, thm_a.rhs, registry=reg)
    assert res.closed and res.theorems_cited
    thm_b = Theorem(name="b", statement="via a", lhs=thm_a.lhs, rhs=thm_a.rhs, proof=res.chain, owner=E_sup).with_structural_requires()
    assert thm_b.requires_verified and not thm_b.legacy
    # transitive: thm_a's Jacobi requirement is carried over, folded onto the
    # dominating structure of the citing proof (E_lie ⊑ E_sup, which declares it too)
    assert any(a.kind == "declared" and a.owner is E_sup and "Jacobi" in a.name for a in thm_b.requires)
    # a structure with E_sup's extra axiom but WITHOUT jacobi cannot cite thm_b
    E4 = _alg("right-leibniz", "antisymmetric", "metric-invariance")
    eng4 = algebroid_engine(E4, registry=reg)
    eng4.register(TheoremDefinition(thm_b))
    assert not eng4.definitions[-1].citable and any("jacobi" in r for r in eng4.definitions[-1].blocked)


def test_legacy_records_are_never_cited_silently(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    verified = _jacobi_theorem(E_lie, u, v, w, reg)
    legacy = Theorem(name="old", statement="J = 0", lhs=verified.lhs, rhs=verified.rhs, proof=verified.proof,
                     from_axioms=("Poisson (π) — a text nobody can check",), owner=E_lie)
    assert legacy.legacy                                               # never derived: legacy
    # prose alone does not decide: once derived, the unmatched text is a visible
    # TEXTUAL dependency and the record is citable on its structural merits
    derived = legacy.with_structural_requires()
    assert not derived.legacy and any(a.kind == "textual" for a in derived.requires)
    td = TheoremDefinition(legacy)
    assert not td.citable and not td.matches(legacy.lhs)              # even unbound
    eng = algebroid_engine(E_sub, registry=reg)                        # a context without the axiom
    eng.register(td)
    assert not eng.definitions[-1].citable
    # explicit opt-in: it fires, and the result carries the marker
    eng2 = ExpansionEngine([TheoremDefinition(legacy, allow_legacy=True)])
    assert "LEGACY" in eng2.definitions[-1].name
    res = prove(eng2, legacy.lhs, Integer(0), registry=reg)
    assert res.closed and any(a.legacy and "unverified citation" in a.name for a in res.requires)
    assert any("LEGACY citation allowed" in line for line in eng2.assembly_report)
    # … and propagates: a theorem built on it is legacy too
    thm_c = Theorem(name="c", statement="s", lhs=legacy.lhs, rhs=Integer(0), proof=res.chain, owner=E_lie).with_structural_requires()
    assert thm_c.legacy and any("unverified citation" in r for r in thm_c.unverified_reasons())
    # the same TheoremDefinition object bound to two engines keeps two verdicts (no shared cache)
    td2 = TheoremDefinition(verified)
    e_ok, e_no = algebroid_engine(E_lie, registry=reg), algebroid_engine(E_sub, registry=reg)
    e_ok.register(td2)
    e_no.register(td2)
    assert e_ok.definitions[-1].citable and not e_no.definitions[-1].citable and td2._check is None


def test_theorem_book_refuses_unverified_records_as_theorems(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    verified = _jacobi_theorem(E_lie, u, v, w, reg)
    legacy = Theorem(name="old", statement="s", lhs=verified.lhs, rhs=verified.rhs, proof=verified.proof, owner=E_lie)
    book = TheoremBook()
    book.add(verified)
    assert book.is_verified("jacobi_E")
    with pytest.raises(LegacyTheoremError, match="unverified=True"):
        book.add(legacy)
    book.add(legacy, unverified=True)                                  # recorded as an assumption, visibly
    assert not book.is_verified("old") and book.unverified_names() == ("old",)
    eng = algebroid_engine(E_lie, registry=reg)
    cite(eng, book, "jacobi_E")                                        # verified: fine
    with pytest.raises(LegacyTheoremError):
        cite(eng, book, "old")
    cite(eng, book, "old", allow_legacy=True)
    assert eng.definitions[-1].citable and "LEGACY" in eng.definitions[-1].name
    book.replace(verified)
    assert book.is_verified("jacobi_E")
    book.remove("old")
    assert book.unverified_names() == ()


def test_citation_after_the_assumption_is_removed(world):
    reg, f, E_lie, E_sub, E_sup, u, v, w = world
    thm = _jacobi_theorem(E_lie, u, v, w, reg)
    eng = algebroid_engine(E_lie, registry=reg)
    eng.register(TheoremDefinition(thm))
    assert prove(eng, Sum(Jacobiator("E", u, v, w), Jacobiator("E", u, v, w)), registry=reg).closed
    # the "same" structure with the axiom retracted: a fresh engine, a fresh verdict
    E_retracted = _alg("right-leibniz", "antisymmetric")
    eng2 = algebroid_engine(E_retracted, registry=reg)
    eng2.register(TheoremDefinition(thm))
    res = prove(eng2, Sum(Jacobiator("E", u, v, w), Jacobiator("E", u, v, w)), registry=reg)
    assert res.status == "RESIDUAL" and not res.theorems_cited
    # transfer (2c) does not launder requirements either: the target must declare them
    with pytest.raises(AmbiguousOwnerError):
        OwnerScope().transfer(thm, E_retracted)
