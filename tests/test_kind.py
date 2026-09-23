"""The common type query ``kind_of`` (Faz 8 step 1c/2a; audit 7a11162):
kind is not degree, a foreign bundle is not unknown, a known mixed sum
is a mismatch, and bilinear combinations / calculus operators are
looked through."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.central.objects import (
    Bundle,
    Form,
    Metric,
    PVector,
    Tensor,
    forms,
    frame,
    functions,
    kind_of,
    musical_view,
    vector_fields,
)
from jacopy.central.objects.bundle import TangentBundle
from jacopy.central.objects.interior import Interior
from jacopy.central.objects.multivector_interior import MultivectorInterior
from jacopy.central.tangent.cartan import L
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge


@pytest.fixture()
def cast():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    U, V = vector_fields("U V")
    om, et = forms("ω η", degree=1)
    (B,) = forms("B", degree=2)
    return reg, f, g, U, V, om, et, B


def test_atoms(cast):
    reg, f, g, U, V, om, et, B = cast
    assert kind_of(f, reg).kind == "function"
    assert kind_of(Symbol("x")).kind == "unknown"          # undeclared symbol
    assert kind_of(Integer(0)).kind == "zero" and kind_of(Integer(3)).kind == "function"
    assert kind_of(U).kind == "vector" and kind_of(U).bundle.is_tangent
    assert kind_of(B) == kind_of(B) and kind_of(B).kind == "form" and kind_of(B).degree == 2
    assert kind_of(PVector("Π", degree=2)).kind == "multivector"     # NOT a 2-form
    assert kind_of(Tensor("T", upper=1, lower=2)).signature == (1, 2)
    assert kind_of(Metric("g")).signature == (0, 2)
    assert kind_of(Form("w_F", degree=1, bundle=Bundle("F"))).bundle == Bundle("F")


def test_bilinear_combinations(cast):
    reg, f, g, U, V, om, et, B = cast
    assert kind_of(Sum(U, V)).kind == "vector"
    assert kind_of(Product(f, U), reg).kind == "vector"
    assert kind_of(Neg(om)).degree == 1
    assert kind_of(Sum(om, B)).kind == "mismatch"           # known, wrong — not unknown
    assert kind_of(Sum(om, Symbol("x"))).kind == "unknown"
    assert kind_of(Sum(U, Neg(U))).kind == "vector"
    assert kind_of(Product(Integer(0), U)).kind == "zero"
    assert kind_of(Product(f, g), reg).kind == "function"


def test_calculus_operators_shift_degrees(cast):
    reg, f, g, U, V, om, et, B = cast
    assert kind_of(d(f), reg) == kind_of(om).__class__("form", 1)
    assert kind_of(d(om)).degree == 2
    assert kind_of(Act(Interior(U), B)).degree == 1
    assert kind_of(Act(Interior(U), om), reg).kind == "function"
    assert kind_of(L(U, B)).degree == 2
    assert kind_of(Act(MultivectorInterior(PVector("Π", degree=2)), B), reg).kind == "function"
    assert kind_of(lie_bracket(U, V)).kind == "vector"
    assert kind_of(Wedge(om, et)).degree == 2 and kind_of(Wedge(U, V)).kind == "multivector"
    assert kind_of(musical_view(Metric("g"), U)).degree == 1
    assert kind_of(Act(U, f), reg).kind == "function"


def test_frame_fields_carry_their_bundle():
    fr = frame(bundle=TangentBundle(dim=3))
    k = kind_of(fr.dual().field("0"))
    assert k.kind == "form" and k.degree == 1 and k.bundle.dim == 3
    assert kind_of(fr.field("1")).kind == "vector"
