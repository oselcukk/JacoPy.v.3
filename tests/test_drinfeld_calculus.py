"""Drinfeld package — Phase 6.A: the usual Cartan calculus IS a
calculus in the bialgebroid sense (drinfeld §4 conditions + 𝒦
composition laws), mechanically at p = 1, 2."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import forms, functions, vector_fields
from jacopy.packages.drinfeld import (
    prove_calculus_condition_one,
    prove_calculus_condition_three,
    prove_calculus_condition_two,
    prove_kappa_bracket,
    prove_kappa_kappa,
)


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    U, V, W = vector_fields("U V W")
    slots = vector_fields("Y1 Y2")
    return reg, f, U, V, W, slots


@pytest.mark.parametrize("p", [1, 2])
class TestCalculusConditions:
    def test_condition_one(self, setup, p):
        reg, f, U, V, _, slots = setup
        (mu,) = forms(f"μ{p}", degree=p)
        chain, _ = prove_calculus_condition_one(
            U, V, mu, f, slots[:p], registry=reg
        )
        assert chain.steps

    def test_condition_two(self, setup, p):
        reg, f, U, _, W, slots = setup
        (eta,) = forms(f"η{p}", degree=p)
        chain, _ = prove_calculus_condition_two(
            U, W, eta, f, slots[:p], registry=reg
        )
        assert chain.steps

    def test_condition_three(self, setup, p):
        reg, f, _, V, W, slots = setup
        (om,) = forms(f"ω{p}", degree=p)
        chain, _ = prove_calculus_condition_three(
            W, V, om, f, slots[:p], registry=reg
        )
        assert chain.steps


@pytest.mark.parametrize("p", [1, 2])
class TestKappaLaws:
    def test_kappa_kappa(self, setup, p):
        reg, f, U, V, _, slots = setup
        (om,) = forms(f"ω{p}", degree=p)
        chain, _ = prove_kappa_kappa(
            U, V, om, f, slots[:p], registry=reg
        )
        assert chain.steps

    def test_kappa_bracket(self, setup, p):
        reg, f, U, V, _, slots = setup
        (om,) = forms(f"ω{p}", degree=p)
        chain, _ = prove_kappa_bracket(
            U, V, om, f, slots[:p], registry=reg
        )
        assert chain.steps
