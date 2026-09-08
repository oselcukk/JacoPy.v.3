"""PDF items 11e / 12f — the given-input evaluation workflow: the
user supplies component data (Γ, g, γ, θ) and the engine evaluates
the package's symbolic component expressions against it."""

import pytest

from jacopy.core.expr import Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.algebra.derivation import Act
from jacopy.central.objects import frame, functions
from jacopy.central.objects.connection import connection
from jacopy.central.tangent.anholonomy import (
    anholonomy_coefficient,
)
from jacopy.packages.metric_affine import metric as affine_metric
from jacopy.packages.metric_affine.component_workflow import (
    ComponentInput,
)
from jacopy.packages.metric_affine.frame_components import (
    ConnectionCoefficient,
    curvature_component_formula,
)
from jacopy.packages.poisson import poisson_structure


@pytest.fixture()
def setup():
    reg = PropertyRegistry()
    f, h = functions("f h", registry=reg)
    fr = frame()
    conn = connection()
    g = affine_metric("g")
    return reg, f, h, fr, conn, g


def _torsion_component(conn, fr, a, b, c):
    return Sum(
        ConnectionCoefficient(conn.name, fr.name, a, b, c),
        Neg(ConnectionCoefficient(conn.name, fr.name, a, c, b)),
        Neg(anholonomy_coefficient(fr, a, b, c)),
    )


def test_torsion_component_from_given_gamma(setup):
    reg, f, h, fr, conn, g = setup
    data = ComponentInput(
        fr, dim=2, connection=conn, total=True, holonomic=True
    )
    data.set_gamma("0", "0", "1", f)
    assert (
        data.evaluate(
            _torsion_component(conn, fr, "0", "0", "1"),
            registry=reg,
        )
        == f
    )
    # the symmetric-in-lower part with zero data dies
    assert data.evaluate(
        _torsion_component(conn, fr, "1", "0", "1"),
        registry=reg,
    ) == Integer(0)


def test_curvature_component_formula_evaluates(setup):
    # The PACKAGE's symbolic R^a_bcd (IndexedSum over the frame)
    # evaluates against given data: only −e_1(Γ^0_01) survives.
    reg, f, h, fr, conn, g = setup
    data = ComponentInput(
        fr, dim=2, connection=conn, total=True, holonomic=True
    )
    data.set_gamma("0", "0", "1", f)
    R = curvature_component_formula(
        conn, fr, "0", "1", "0", "1"
    )
    assert data.evaluate(R, registry=reg) == Neg(
        Act(fr.field("1"), f)
    )


def test_unset_components_stay_symbolic_without_total(setup):
    # Soundness: without total=True nothing defaults to zero.
    reg, f, h, fr, conn, g = setup
    data = ComponentInput(fr, dim=2, connection=conn)
    gamma = ConnectionCoefficient(
        conn.name, fr.name, "1", "1", "0"
    )
    assert data.evaluate(gamma, registry=reg) == gamma


def test_metric_components_substitute(setup):
    reg, f, h, fr, conn, g = setup
    data = ComponentInput(
        fr, dim=2, metric=g, total=True, holonomic=True
    )
    data.set_metric("0", "0", 1)
    data.set_metric("1", "1", h)
    assert data.evaluate(
        g(fr.field("1"), fr.field("1")), registry=reg
    ) == h
    assert data.evaluate(
        g(fr.field("0"), fr.field("1")), registry=reg
    ) == Integer(0)


def test_pi_components_substitute_antisymmetrically(setup):
    # PDF 12f: given θ^{01}, the coframe evaluation returns it and
    # the swapped order returns the negative.
    reg, f, h, fr, conn, g = setup
    P = poisson_structure()
    co = fr.dual()
    data = ComponentInput(
        fr, dim=2, pi=P.pi, total=True, holonomic=True
    )
    data.set_pi("0", "1", f)
    ev = lambda a, b: MultiEval(
        P.pi,
        co.field(a),
        co.field(b),
        alternating=True,
        slot_kind="covector",
    )
    assert data.evaluate(ev("0", "1"), registry=reg) == f
    assert data.evaluate(ev("1", "0"), registry=reg) == Neg(f)
    assert data.evaluate(ev("0", "0"), registry=reg) == Integer(0)
