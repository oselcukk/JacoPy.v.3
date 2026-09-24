"""Faz 8 step 4a — structure providers: every structure contributes
ITS OWN rules (``engine_rules``) and carries ITS OWN declarations; a
flag is never distributed across structures; the order of structures
does not change what is assumed; rules are deduplicated by structural
key; the old builders are thin adapters over the new contract."""

import pytest

from jacopy.central.algebroid import algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.objects import Bundle, forms, functions
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.core.expr import Integer
from jacopy.core.registry import PropertyRegistry
from jacopy.packages.drinfeld.tilde_calculus import NambuMorphismDeclaration, _tilde_engine
from jacopy.packages.poisson.core import PoissonSNDeclaration, poisson_engine, poisson_structure
from jacopy.packages.poisson.nambu import NAMBU_DECLARATIONS, nambu_structure
from jacopy.proof import ExplicitAssumption
from jacopy.proof.ownership import is_substructure
from jacopy.proof.result import prove
from jacopy.research import assemble_engine, assembly_report
from jacopy.research.providers import declared_axioms, is_provider, require_provider, structure_rules


@pytest.fixture()
def world():
    reg = PropertyRegistry()
    (h,) = functions("h", registry=reg)
    a, b = forms("a b", degree=2)
    N1 = nambu_structure("Π₁", p=2, declare=("fundamental-identity",))
    N2 = nambu_structure("Π₂", p=2)
    return reg, h, a, b, N1, N2


def _morphism_goal(N, a, b):
    # the FI's first consequence: [Πa, Πb] = Π[a, b]_Kos — closes only where N declares the FI
    from jacopy.packages.poisson.nambu import nambu_koszul_bracket

    return lie_bracket(N.sharp_vf(a), N.sharp_vf(b)), N.sharp_vf(nambu_koszul_bracket(N, a, b))


def test_structures_carry_their_own_declarations(world):
    reg, h, a, b, N1, N2 = world
    assert N1.declares("fundamental-identity") and not N2.declares("fundamental-identity")
    assert N1 != nambu_structure("Π₁", p=2) and is_substructure(nambu_structure("Π₁", p=2), N1)
    assert not is_substructure(N1, nambu_structure("Π₁", p=2))
    assert N1.name == "Π₁" and "declare=" in repr(N1)
    with pytest.raises(ValueError):
        nambu_structure("Π", p=2, declare=("closure",))
    assert NAMBU_DECLARATIONS == ("fundamental-identity",)
    P = poisson_structure("π", declare=("poisson",))
    assert P.declares("poisson") and not poisson_structure("π").declares("poisson")
    for s in (N1, N2, P, algebroid("E", Bundle("E"))):
        assert is_provider(s) and require_provider(s) is s
    with pytest.raises(TypeError):
        require_provider(Integer(1))


def test_each_structure_contributes_its_own_rules(world):
    reg, h, a, b, N1, N2 = world
    r1, r2 = structure_rules(N1, reg), structure_rules(N2, reg)
    assert any(isinstance(r, NambuMorphismDeclaration) for r in r1)
    assert not any(isinstance(r, NambuMorphismDeclaration) for r in r2)
    assert all(r.owner is N1 for r in r1) and all(r.owner is N2 for r in r2)
    assert all(r.identity == (type(r).__name__, "Π₁") for r in r1)
    assert [type(r).__name__ for r in structure_rules(N1, reg, phase="declared")] == ["NambuMorphismDeclaration", "FISharpPairingSwapDefinition"]
    assert structure_rules(N2, reg, phase="declared") == []
    with pytest.raises(ValueError):
        structure_rules(N1, reg, phase="magic")
    assert declared_axioms([N1, N2]) == {"Π₁": ["fundamental-identity"], "Π₂": []}


def test_the_flag_is_not_distributed(world):
    # ACCEPTANCE: with the FI declared on N₁ only, an identity that needs N₂'s FI does not close
    reg, h, a, b, N1, N2 = world
    lhs1, rhs1 = _morphism_goal(N1, a, b)
    lhs2, rhs2 = _morphism_goal(N2, a, b)
    eng = assemble_engine(lhs1, lhs2, registry=reg, structures=(N1, N2))
    assert eng.owners == {"Π₁": N1, "Π₂": N2}
    res1 = prove(eng, lhs1, rhs1, registry=reg)
    res2 = prove(eng, lhs2, rhs2, registry=reg)
    assert res1.closed and any(a_.kind == "declared" and a_.owner is N1 for a_ in res1.requires)
    assert res2.status == "RESIDUAL" and not res2.declared
    # N₂ still has its DEFINITIONAL capabilities (symmetric): its sharp rules are registered
    names = {type(r).__name__ for r in eng.definitions if getattr(r, "owner", None) is N2}
    assert {"NambuSharpActionDefinition", "NambuSharpLinearityDefinition", "NambuSharpPairingEvalDefinition"} <= names
    assert any("Π₂: no fundamental-identity declaration" in line for line in assembly_report(eng))


def test_structure_order_does_not_change_the_assumption_scope(world):
    reg, h, a, b, N1, N2 = world
    lhs1, rhs1 = _morphism_goal(N1, a, b)
    lhs2, rhs2 = _morphism_goal(N2, a, b)
    e12 = assemble_engine(lhs1, lhs2, registry=reg, structures=(N1, N2))
    e21 = assemble_engine(lhs1, lhs2, registry=reg, structures=(N2, N1))
    assert {r.key for r in e12.definitions} == {r.key for r in e21.definitions}
    assert e12.owners == e21.owners
    for eng in (e12, e21):
        assert prove(eng, lhs1, rhs1, registry=reg).closed
        assert prove(eng, lhs2, rhs2, registry=reg).status == "RESIDUAL"
    req12 = {(x.kind, x.name) for x in prove(e12, lhs1, rhs1, registry=reg).requires}
    req21 = {(x.kind, x.name) for x in prove(e21, lhs1, rhs1, registry=reg).requires}
    assert req12 == req21


def test_legacy_declare_fi_applies_to_the_first_structure_only_and_is_reported(world):
    reg, h, a, b, N1, N2 = world
    N1_bare = nambu_structure("Π₁", p=2)
    lhs1, rhs1 = _morphism_goal(N1_bare, a, b)
    lhs2, rhs2 = _morphism_goal(N2, a, b)
    eng = assemble_engine(lhs1, lhs2, registry=reg, structures=(N1_bare, N2), declare_fi=True)
    assert eng.owners["Π₁"].declares("fundamental-identity") and not eng.owners["Π₂"].declares("fundamental-identity")
    assert any("pre-4a spelling" in line for line in assembly_report(eng))
    assert prove(eng, lhs1, rhs1, registry=reg).closed
    assert prove(eng, lhs2, rhs2, registry=reg).status == "RESIDUAL"
    with pytest.raises(ValueError):
        assemble_engine(registry=reg, structures=(N2, nambu_structure("Π₂", p=2)))   # one multivector twice


def test_rules_are_deduplicated_by_key_not_by_class(world):
    reg, h, a, b, N1, N2 = world
    x, y = forms("x y", degree=1)
    e1 = ExplicitAssumption(x, owner=None)
    e2 = ExplicitAssumption(y, owner=None)
    eng = assemble_engine(registry=reg, structures=(N1, N2), extra=(e1, e2))
    assert sum(isinstance(r, ExplicitAssumption) for r in eng.definitions) == 2
    # a structure-free rule appearing in both structures' stacks is registered once
    from jacopy.packages.drinfeld.tilde_calculus import LieIotaCommutatorDefinition

    assert sum(isinstance(r, LieIotaCommutatorDefinition) for r in eng.definitions) == 1
    # two structures' same-class rules are two rules (different keys)
    from jacopy.packages.poisson.nambu import NambuSharpLinearityDefinition

    assert sum(isinstance(r, NambuSharpLinearityDefinition) for r in eng.definitions) == 2


def test_old_builders_are_adapters_over_the_contract(world):
    reg, h, a, b, N1, N2 = world
    N1_bare = nambu_structure("Π₁", p=2)
    old = _tilde_engine(N1_bare, reg, declare_fi=True)
    new = _tilde_engine(N1, reg)
    assert [r.key for r in old.definitions] == [r.key for r in new.definitions]
    assert old.owners["Π₁"] == N1 and new.owners["Π₁"] is N1
    bare = _tilde_engine(N1_bare, reg)
    assert not any(isinstance(r, NambuMorphismDeclaration) for r in bare.definitions)
    # Poisson: the legacy poisson_engine default declares; the 4a form declares on the structure
    P = poisson_structure("π")
    legacy = poisson_engine(registry=reg, structures=(P,))
    strict = poisson_engine(registry=reg, structures=(P,), declare_sn=False)
    declared = poisson_engine(registry=reg, structures=(P.with_declarations("poisson"),), declare_sn=False)
    assert any(isinstance(r, PoissonSNDeclaration) for r in legacy.definitions)
    assert not any(isinstance(r, PoissonSNDeclaration) for r in strict.definitions)
    assert [r.key for r in declared.definitions] == [r.key for r in legacy.definitions]
    assert legacy.owners["π"].declares("poisson") and strict.owners["π"] == P


def test_algebroid_is_a_provider_too():
    reg = PropertyRegistry()
    E = algebroid("E", Bundle("E"), declare=("lie",))
    rules = structure_rules(E, reg)
    assert all(r.owner is E for r in rules)
    phases = [structure_rules(E, reg, phase=p) for p in ("declared", "coboundary", "expansion")]
    assert [len(p) for p in phases] == [3, 2, 3]
    keys = {r.key for r in rules}
    assert keys <= {r.key for r in algebroid_engine(E, registry=reg).definitions}
