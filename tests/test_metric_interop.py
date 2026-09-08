"""PDF item 5 — package compatibility of the metric object
(2026-09-08 audit, compliance finding 3): the central Metric atom
and the metric-affine Metric context bridge in BOTH directions, so
one ``g`` travels across central, metric-affine and generalized
computations."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import (
    forms,
    hodge,
    metric as central_metric,
    vector_fields,
)
from jacopy.central.objects.connection import connection
from jacopy.packages.metric_affine import (
    metric as affine_metric,
    metric_affine_engine,
    nonmetricity,
)


def test_central_metric_usable_by_metric_affine():
    # The audit counterexample: this used to raise TypeError.
    X, Y, Z = vector_fields("X Y Z")
    g = central_metric("g")
    q = nonmetricity(connection(), g, X, Y, Z)
    assert q is not None


def test_affine_metric_usable_by_central_hodge():
    (w,) = forms("ω", degree=1)
    ga = affine_metric("g")
    gc = central_metric("g")
    # the two spellings of the same g give the SAME Hodge star
    assert hodge(w, ga) == hodge(w, gc)


def test_same_g_expands_identically_either_way():
    # Q(X,Y,Z) built from the central atom expands to the same
    # normal form as the one built from the affine context.
    from jacopy.packages.poisson.tilde import _normalized_by

    reg = PropertyRegistry()
    X, Y, Z = vector_fields("X Y Z")
    conn = connection()
    q_central = nonmetricity(
        conn, central_metric("g"), X, Y, Z
    )
    q_affine = nonmetricity(
        conn, affine_metric("g"), X, Y, Z
    )
    eng = metric_affine_engine(registry=reg)
    assert _normalized_by(eng, q_central, reg) == _normalized_by(
        eng, q_affine, reg
    )


def test_bridge_rejects_non_metrics():
    from jacopy.packages.metric_affine.metric import (
        as_metric_context,
    )

    with pytest.raises(TypeError):
        as_metric_context("not a metric")
    (w,) = forms("ω", degree=1)
    with pytest.raises(TypeError):
        hodge(w, "not a metric")
