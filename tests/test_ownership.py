"""Faz 8 step 2c — minimal context identity. Two structures with the
SAME tree-visible name build identical expressions; the system must
tell the owners apart or reject the ambiguous mixture explicitly —
never guess. The tree does not change."""

import pytest

from jacopy.central.algebroid import algebroid
from jacopy.central.algebroid.declarations import declaration_rules
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.operators import Jacobiator
from jacopy.central.objects import Bundle
from jacopy.core.expr import Integer
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import (
    AmbiguousOwnerError,
    ExpansionEngine,
    OwnerScope,
    ProofChain,
    chain_owners,
    check_owners,
    owner_name,
    owner_of,
)
from jacopy.proof.theorems import Theorem, TheoremBook, TheoremDefinition


@pytest.fixture()
def twins():
    """E₁ = (E, lie) and E₂ = (E, nothing declared): same name, same
    bundle, different assumptions — and identical expressions."""
    reg = PropertyRegistry()
    E1 = algebroid("E", Bundle("E"), declare=("lie",))
    E2 = algebroid("E", Bundle("E"))
    u, v, w = E1.sections("u v w")
    assert E1.bracket(u, v) == E2.bracket(u, v)      # the tree cannot tell them apart
    assert E1 != E2 and owner_name(E1) == owner_name(E2) == "E"
    return reg, E1, E2, u, v, w


def _jacobi_theorem(E, u, v, w, reg):
    eng = algebroid_engine(E, registry=reg)
    J = Jacobiator("E", u, v, w)
    out, steps = eng.expand(J)
    assert out == Integer(0)
    return Theorem(
        name="jacobi_E", statement="J^E(u,v,w) = 0", lhs=J, rhs=out,
        proof=ProofChain(steps), from_axioms=("Leibniz-Jacobi (E)",), owner=E,
    ).with_structural_requires()


def test_owners_are_recorded_on_rules_steps_and_theorems(twins):
    reg, E1, E2, u, v, w = twins
    rules = declaration_rules(E1, reg)
    assert rules and all(r.owner is E1 for r in rules)
    assert all(owner_of(r) is E1 for r in rules)
    thm = _jacobi_theorem(E1, u, v, w, reg)
    assert thm.owner is E1
    assert chain_owners(thm.proof) == (E1,)            # the assumption record names the OBJECT
    assert all(s.owner is E1 for s in thm.proof.steps if s.provenance_tag == "axiom")
    assert "(E)" in thm.proof.steps[0].rule             # the display name is unchanged
    assert algebroid_engine(E1, registry=reg).owners == {"E": E1}
    assert TheoremDefinition(thm).owner is E1


def test_mixing_rules_of_two_same_named_structures_is_rejected(twins):
    reg, E1, E2, u, v, w = twins
    # E₂'s definitional rules assume nothing E₁ does not: registering them
    # first and then E₁'s declared rules is a legitimate STRENGTHENING
    # (the name's owner becomes E₁); the other order would smuggle E₁'s
    # axioms into E₂'s engine and is refused
    eng = ExpansionEngine([*algebroid_engine(E2, registry=reg).definitions, *declaration_rules(E1, reg)])
    assert eng.owners == {"E": E1}
    eng2 = algebroid_engine(E2, registry=reg)
    eng2.register(declaration_rules(E1, reg)[0])
    assert eng2.owners == {"E": E1}
    E_other = algebroid("E", Bundle("F"), declare=("lie",))       # same name, another bundle: different
    with pytest.raises(AmbiguousOwnerError, match="cannot distinguish them|share the tree-visible name"):
        ExpansionEngine([*declaration_rules(E1, reg), *declaration_rules(E_other, reg)])
    with pytest.raises(AmbiguousOwnerError):
        algebroid_engine(E1, registry=reg).register(declaration_rules(E_other, reg)[0])
    # the same structure twice is one owner (equal objects)
    E1b = algebroid("E", Bundle("E"), declare=("lie",))
    eng = ExpansionEngine([*declaration_rules(E1, reg), *declaration_rules(E1b, reg)])
    assert eng.owners == {"E": E1}
    # different names never conflict
    F = algebroid("F", Bundle("F"), declare=("lie",))
    eng = ExpansionEngine([*declaration_rules(E1, reg), *declaration_rules(F, reg)])
    assert eng.owners == {"E": E1, "F": F}


def test_citing_a_theorem_in_the_wrong_same_named_structure_is_rejected(twins):
    reg, E1, E2, u, v, w = twins
    thm = _jacobi_theorem(E1, u, v, w, reg)
    eng2 = algebroid_engine(E2, registry=reg)
    # without ownership the citation would fire: the lhs is structurally E₂'s Jacobiator too
    assert TheoremDefinition(thm).matches(Jacobiator("E", u, v, w))
    # in E₂'s engine the citation is registered INERT and reported (3b): the
    # requirement "jacobi of E₁" is not met by E₂, and E₂ stays the owner
    eng2.register(TheoremDefinition(thm))
    assert not eng2.definitions[-1].citable and eng2.owners == {"E": E2}
    assert not eng2.definitions[-1].matches(Jacobiator("E", u, v, w))
    # in its own structure the citation is welcome
    eng1 = algebroid_engine(E1, registry=reg)
    eng1.register(TheoremDefinition(thm))
    assert eng1.owners == {"E": E1} and eng1.definitions[-1].citable
    # the book refuses the record under the other owner
    book = TheoremBook()
    book.add(thm)
    assert book.get("jacobi_E") is thm and book.get("jacobi_E", owner=E1) is thm
    with pytest.raises(AmbiguousOwnerError):
        book.get("jacobi_E", owner=E2)


def test_scope_binds_one_structure_per_name(twins):
    reg, E1, E2, u, v, w = twins
    scope = OwnerScope()
    assert scope.bind(E1) is E1
    assert scope.bind(algebroid("E", Bundle("E"), declare=("lie",))) is E1  # equal: no-op, first kept
    with pytest.raises(AmbiguousOwnerError, match="already bound"):
        scope.bind(E2)
    assert scope.owner("E") is E1 and scope.bound == {"E": E1}
    with pytest.raises(KeyError):
        scope.owner("F")
    # checking items: E₁'s rules pass; E₂ = (E, ∅) is a SUB-structure of E₁
    # (same name and bundle, fewer declarations) — its items assume less
    # and pass too (3b refinement); a SUPER-structure's items are refused
    assert scope.check(declaration_rules(E1, reg)) == {"E": E1}
    assert scope.check_engine(algebroid_engine(E1, registry=reg)) == {"E": E1}
    assert scope.check(algebroid_engine(E2, registry=reg).definitions) == {"E": E1}
    assert scope.check([E2]) == {"E": E1}
    E_more = E1.with_declarations("metric-invariance")
    with pytest.raises(AmbiguousOwnerError, match="is bound to"):
        scope.check(declaration_rules(E_more, reg))
    with pytest.raises(AmbiguousOwnerError):
        scope.check([algebroid("E", Bundle("F"), declare=("lie",))])  # another bundle: a different structure
    assert check_owners([]) == {} and check_owners([None]) == {}   # nothing owned, nothing claimed


def test_transfer_is_explicit_and_checked(twins):
    reg, E1, E2, u, v, w = twins
    thm = _jacobi_theorem(E1, u, v, w, reg)
    scope = OwnerScope()
    # to a structure assuming MORE: allowed, recorded in the notes, tree unchanged
    E3 = E1.with_declarations("metric-invariance")
    moved = scope.transfer(thm, E3, note="metric case")
    assert moved.owner is E3 and moved.lhs == thm.lhs and moved.rhs == thm.rhs
    assert "transferred from" in moved.notes and "metric case" in moved.notes
    assert moved.proof is thm.proof
    # to a structure assuming LESS: refused, naming the missing declaration
    with pytest.raises(AmbiguousOwnerError, match="lacks"):
        scope.transfer(thm, E2)
    # to another name: refused (the expressions say "E")
    with pytest.raises(AmbiguousOwnerError, match="without changing the expressions"):
        scope.transfer(thm, algebroid("F", Bundle("F"), declare=("lie",)))
    # an unowned (legacy) theorem cannot be transferred
    legacy = Theorem(name="legacy", statement="x", lhs=thm.lhs, rhs=thm.rhs, proof=thm.proof)
    assert legacy.owner is None
    with pytest.raises(AmbiguousOwnerError, match="records no owner"):
        scope.transfer(legacy, E1)
    # the scope's own binding wins over a transfer target
    scope.bind(E1)
    with pytest.raises(AmbiguousOwnerError, match="is bound to"):
        scope.transfer(thm, E3)


def test_same_named_poisson_and_nambu_structures_are_ambiguous():
    from jacopy.packages.poisson.core import poisson_structure
    from jacopy.packages.poisson.nambu import nambu_structure
    from jacopy.research import assemble_engine

    P = poisson_structure("π")
    N = nambu_structure("π", p=2)
    assert owner_name(P) == owner_name(N) == "π" and P != N
    with pytest.raises(AmbiguousOwnerError):
        check_owners([P, N])
    with pytest.raises(AmbiguousOwnerError):
        assemble_engine(structures=(N, P))
    assert check_owners([P, poisson_structure("π")]) == {"π": P}   # equal structures: one owner


def test_the_tree_itself_is_unchanged(twins):
    reg, E1, E2, u, v, w = twins
    from jacopy.central.algebroid.context import AlgebroidBracket

    b = E1.bracket(u, v)
    assert isinstance(b, AlgebroidBracket) and b.algebroid_name == "E"
    assert not hasattr(b, "owner")
