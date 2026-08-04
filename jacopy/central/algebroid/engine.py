"""
The algebroid expansion engine (Phase 3.A).

Assembles the definitional rules valid on ANY algebroid: the generic
structural rules (operator linearity, scalar action, slot linearity)
plus the anchored-bundle definitions (anchor ``C^∞``-linearity,
bracket ℝ-bilinearity, coboundary evaluation).

**The E = TM reduction is literal**: for the tangent algebroid the
factory simply returns the tangent engine — the usual calculus IS the
instantiation (PDF item 8a), not a parallel rule set.

Declared hierarchy rules (right-Leibniz, anchor morphism, …) arrive in
Phase 3.B and are registered per algebroid instance on top of this
base, exactly like holonomic frames on the tangent engine.
"""

from __future__ import annotations

from typing import Optional

from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ActOverSumOpDefinition, ExpansionEngine
from jacopy.central.calculus import (
    HeadScalarDefinition,
    HeadSumDefinition,
    IntrinsicDDefinition,
    IntrinsicLDefinition,
    PairingVectorScalarDefinition,
    SlotNegDefinition,
    SlotZeroDefinition,
)
from jacopy.central.calculus.interior_rules import (
    InteriorActDefinition,
    InteriorEvalDefinition,
)
from jacopy.central.calculus.symmetrize_rules import SymAltEvalDefinition
from jacopy.central.algebroid.context import Algebroid
from jacopy.central.algebroid.declarations import declaration_rules
from jacopy.central.algebroid.operators import (
    DerivatorExpansionDefinition,
    JacobiatorExpansionDefinition,
    MappedSectionLinearityDefinition,
    PredatorExpansionDefinition,
)
from jacopy.central.algebroid.rules import (
    AnchorLinearityDefinition,
    BracketBilinearityDefinition,
    CoboundaryLinearityDefinition,
    CoboundaryPairingDefinition,
    LocalityMultilinearityDefinition,
    MetricBilinearityDefinition,
    MetricSharpEvaluationDefinition,
    MetricSharpLinearityDefinition,
    MetricSymmetryDefinition,
)


def algebroid_engine(
    alg: Algebroid,
    *,
    registry: Optional[PropertyRegistry] = None,
    mode: str = "efficient",
    projectors=(),
) -> ExpansionEngine:
    """Build the engine for the algebroid ``alg``: definitional rules
    plus the rules licensed by its declarations, plus (optionally) the
    defining rules of the given locality projectors (Phase 3.E.4) —
    the same opt-in pattern as ``tangent_engine``'s holonomic frames."""
    if not isinstance(alg, Algebroid):
        raise TypeError("algebroid_engine expects an Algebroid")
    if alg.is_tangent:
        # PDF item 8a in code: the TM instantiation is the usual
        # machinery, not a parallel one.
        from jacopy.central.tangent.engine import tangent_engine

        return tangent_engine(registry=registry, mode=mode)

    # Generic rules shared with the tangent layer. The Lie-bracket
    # action definition is needed because the anchor-morphism rule
    # (and the 3.D theorem) produce [ρ(u), ρ(v)]_Lie nodes whose
    # action on functions must expand.
    from jacopy.central.tangent.lie_bracket import (
        LieBracketActionDefinition,
        ScalarActAsMultiplicationDefinition,
    )
    from jacopy.central.algebroid.calculus import algebroid_calculus

    calc = algebroid_calculus(alg)

    # Declared axioms FIRST: registration order is match precedence,
    # and e.g. the declared J -> 0 must outrank the Jacobiator's
    # definitional expansion.
    from jacopy.central.algebroid.locality import LocalityProjector

    projector_rules = []
    for proj in projectors:
        if not isinstance(proj, LocalityProjector):
            raise TypeError(
                "algebroid_engine projectors must be LocalityProjector "
                f"instances, got {proj!r}"
            )
        projector_rules.extend(proj.rules())

    return ExpansionEngine(
        [
            *declaration_rules(alg, registry),
            *projector_rules,
            ActOverSumOpDefinition(),
            LieBracketActionDefinition(),
            ScalarActAsMultiplicationDefinition(registry),
            IntrinsicDDefinition(calc, registry),
            IntrinsicLDefinition(calc, registry),
            InteriorActDefinition(registry),
            InteriorEvalDefinition(registry),
            SymAltEvalDefinition(),
            HeadSumDefinition(),
            HeadScalarDefinition(registry),
            PairingVectorScalarDefinition(registry),
            SlotNegDefinition(),
            SlotZeroDefinition(),
            AnchorLinearityDefinition(registry),
            BracketBilinearityDefinition(),
            CoboundaryPairingDefinition(alg),
            CoboundaryLinearityDefinition(alg, registry),
            LocalityMultilinearityDefinition(registry),
            MetricBilinearityDefinition(registry),
            MetricSymmetryDefinition(),
            MetricSharpEvaluationDefinition(),
            MetricSharpLinearityDefinition(registry),
            MappedSectionLinearityDefinition(registry),
            JacobiatorExpansionDefinition(alg),
            DerivatorExpansionDefinition(alg),
            PredatorExpansionDefinition(alg),
        ],
        mode=mode,
    )
