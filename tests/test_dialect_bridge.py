"""The Nambu ↔ Poisson dialect bridge (2026-09-10): the order-1
Nambu sharp is the Poisson sharp, the two Koszul brackets agree, and
the form component of [C'1] closes by CITING the 5.E.2b theorem."""

import os

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.generalized.dialect_bridge import (
    prove_koszul_dialects_agree,
    prove_theta_form_jacobi_by_citation,
)
from jacopy.packages.poisson.core import SharpVF
from jacopy.packages.poisson.nambu import (
    NambuToPoissonSharpDefinition,
    nambu_structure,
    poisson_view,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    (X,) = vector_fields("X")
    om, et, ze = forms("ω η ζ", degree=1)
    return reg, f, X, om, et, ze, nambu_structure("θ", p=1)


def test_bridge_rewrites_the_sharp_node(setup):
    reg, f, X, om, et, ze, N = setup
    rule = NambuToPoissonSharpDefinition(N)
    node = N.sharp_vf(om)
    assert rule.matches(node)
    assert rule.rewrite(node) == SharpVF(N.pi, om)
    other = nambu_structure("θ′", p=1)
    assert not rule.matches(other.sharp_vf(om))
    assert poisson_view(N).pi == N.pi


def test_bridge_is_a_p1_statement():
    with pytest.raises(ValueError):
        NambuToPoissonSharpDefinition(nambu_structure(p=2))
    with pytest.raises(ValueError):
        poisson_view(nambu_structure(p=2))


def test_koszul_dialects_agree_for_any_bivector(setup):
    reg, f, X, om, et, ze, N = setup
    chain, thm = prove_koszul_dialects_agree(N, om, et, X, registry=reg)
    assert len(chain.steps) == 1
    assert "declaration-free" in thm.notes


@pytest.mark.skipif(
    not os.environ.get("JACOPY_RUN_SLOW"),
    reason="~45 s (the cited 5.E.2b theorem); set JACOPY_RUN_SLOW=1",
)
def test_theta_form_jacobi_closes_by_citation(setup):
    reg, f, X, om, et, ze, N = setup
    chain, thm = prove_theta_form_jacobi_by_citation(
        N, om, et, ze, X, f, registry=reg
    )
    assert len(chain.steps) == 2
    assert chain.steps[1].children
    assert any("CITED" in a for a in thm.from_axioms)
