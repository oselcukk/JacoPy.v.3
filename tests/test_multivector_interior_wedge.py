"""Soundness pin (2026-09-09): the multivector interior ``ι_P``
(``p ≥ 2``) is ``leibniz = False`` — a COMPOSITION of derivations,
not a derivation — and the product rule must honour that on the
WEDGE path too, not only on ``Product`` (the Wedge branch was
unguarded: ``ι_{Π₃}(η₂ ∧ dω₂)`` was split as a graded Leibniz
rule, which is false for a trivector)."""

from jacopy.algebra.derivation import Act
from jacopy.algorithms.product_rule import product_rule
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.central.objects import forms, vector_fields
from jacopy.central.objects.multivector_interior import (
    MultivectorInterior,
)
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import d
from jacopy.packages.poisson.nambu import nambu_structure


def test_trivector_interior_is_not_split_over_a_wedge():
    reg = PropertyRegistry()
    N3 = nambu_structure("Π₃", p=2)
    om2, et2 = forms("ω₂ η₂", degree=2)
    node = Act(MultivectorInterior(N3.pi), Wedge(et2, d(om2)))
    assert product_rule(node, reg) == node


def test_ordinary_interior_still_splits_over_a_wedge():
    reg = PropertyRegistry()
    (X,) = vector_fields("X")
    a, b = forms("a b", degree=1)
    node = Act(Interior(X), Wedge(a, b))
    assert product_rule(node, reg) != node
