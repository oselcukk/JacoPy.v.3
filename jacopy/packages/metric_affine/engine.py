"""
The metric-affine package engine (Phase 4.A).

The tangent engine plus the package's connection rules — the package
pattern: central code supplies the calculus, a package registers its
own definitional layer on top (never the other way around).
"""

from __future__ import annotations

from typing import Optional

from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine
from jacopy.packages.metric_affine.rules import (
    ConnectionArgumentLeibnizDefinition,
    ConnectionDirectionLinearityDefinition,
)


def metric_affine_engine(
    *,
    registry: Optional[PropertyRegistry] = None,
    mode: str = "efficient",
    holonomic_frames=(),
    compatible=(),
    torsion_free=(),
    decompositions=(),
) -> ExpansionEngine:
    """The TM engine extended with the package rules.

    ``compatible``: iterable of ``(Connection, Metric)`` pairs declared
    metric-compatible (``∇g = 0``); ``torsion_free``: iterable of
    connections declared torsion-free; ``decompositions``: iterable of
    ``(Connection, Frame)`` pairs declaring frame completeness
    (``∇_{e_b}e_c = Σ_s Γ^s_bc e_s`` and ``[e_a,e_b] = Σ_s γ^s_ab e_s``)
    — the package's opt-in declaration mechanism (the tangent analogue
    of the algebroid declaration system).
    """
    from jacopy.central.tangent.engine import tangent_engine

    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )
    from jacopy.packages.metric_affine.metric import (
        MetricCompatibilityDefinition,
        MetricInverseContractionDefinition,
        MetricValueBilinearityDefinition,
        MetricValueSymmetryDefinition,
        NonMetricityExpansionDefinition,
        TorsionFreeDefinition,
    )
    from jacopy.packages.metric_affine.frame_components import (
        FrameConnectionCoefficientDefinition,
        FramePairingScalarDefinition,
        FramePairingSplitDefinition,
    )
    from jacopy.packages.metric_affine.torsion_curvature import (
        CurvatureExpansionDefinition,
        TorsionExpansionDefinition,
    )

    engine = tangent_engine(
        registry=registry, mode=mode, holonomic_frames=holonomic_frames
    )
    from jacopy.packages.metric_affine.decomposition import (
        BracketFrameDecompositionDefinition,
        ConnectionFrameDecompositionDefinition,
        IndexedSumOperatorPushDefinition,
    )

    # Declared (opt-in) rules first: precedence follows registration.
    for conn, g in compatible:
        engine.register(MetricCompatibilityDefinition(conn, g))
    for conn in torsion_free:
        engine.register(TorsionFreeDefinition(conn))
    seen_frames = set()
    for conn, fr in decompositions:
        engine.register(ConnectionFrameDecompositionDefinition(conn, fr))
        if fr not in seen_frames:
            seen_frames.add(fr)
            engine.register(BracketFrameDecompositionDefinition(fr))
    engine.register(IndexedSumOperatorPushDefinition())
    engine.register(ConnectionDirectionLinearityDefinition(registry))
    engine.register(ConnectionArgumentLeibnizDefinition(registry))
    engine.register(TorsionExpansionDefinition())
    engine.register(CurvatureExpansionDefinition())
    engine.register(NonMetricityExpansionDefinition())
    engine.register(FrameConnectionCoefficientDefinition())
    engine.register(FramePairingSplitDefinition())
    engine.register(FramePairingScalarDefinition(registry))
    from jacopy.central.calculus.indexed_rules import (
        IndexedSumEvalPushInDefinition,
        IndexedSumLinearityDefinition,
        KroneckerContractionDefinition,
        WedgeEvalDefinition,
    )

    engine.register(IndexedSumEvalPushInDefinition())
    engine.register(IndexedSumLinearityDefinition())
    engine.register(KroneckerContractionDefinition())
    engine.register(WedgeEvalDefinition(registry))
    engine.register(MetricValueSymmetryDefinition())
    engine.register(MetricValueBilinearityDefinition(registry))
    engine.register(MetricInverseContractionDefinition())
    # The bracket's C∞-module structure — a DERIVED rule (theorem-
    # classified; the tangent engine deliberately omits it, packages
    # opt in): torsion/curvature tensoriality lives on it.
    engine.register(LieBracketLeibnizDefinition(registry))
    from jacopy.central.calculus.tensor_calculus import (
        TensorCovariantEvalDefinition,
    )

    engine.register(TensorCovariantEvalDefinition(registry))
    return engine
