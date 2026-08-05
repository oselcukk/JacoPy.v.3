"""
The tangent-case expansion engine (Phase 2).

Assembles all **definitional** rules of the TM case landed so far
(definition policy §2 in the ROADMAP): Act operator-linearity (core),
the Lie-bracket action (2.A), scalar action as multiplication (2.A),
the connection's action on functions and the frame duality (2.B).
Derived rules (proved theorems) are added in later sub-phases behind
explicit citation.
"""

from __future__ import annotations

from typing import Optional

from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ActOverSumOpDefinition, ExpansionEngine
from jacopy.central.calculus import (
    IntrinsicDDefinition,
    IntrinsicLDefinition,
    InteriorActDefinition,
    HeadSumDefinition,
    HeadScalarDefinition,
    InteriorEvalDefinition,
    PairingVectorScalarDefinition,
    SlotNegDefinition,
    SlotZeroDefinition,
)
from jacopy.central.calculus.symmetrize_rules import SymAltEvalDefinition
from jacopy.central.tangent.lie_bracket import (
    LieBracketActionDefinition,
    ScalarActAsMultiplicationDefinition,
)
from jacopy.central.tangent.definitions import (
    CovariantScalarActionDefinition,
    FrameDualityDefinition,
)


def tangent_engine(
    *,
    registry: Optional[PropertyRegistry] = None,
    mode: str = "efficient",
    holonomic_frames=(),
) -> ExpansionEngine:
    """Build the TM-case engine with the definitional rules only.

    ``holonomic_frames``: frames declared holonomic (coordinate
    frames); their brackets rewrite to zero.
    """
    # Late imports: exterior.py / anholonomy.py proof helpers
    # late-import this module.
    from jacopy.central.tangent.exterior import CARTAN_TM
    from jacopy.central.tangent.anholonomy import (
        FrameBracketCoefficientDefinition,
        HolonomicFrameDefinition,
    )
    from jacopy.central.tangent.lie_bracket import BracketOrientationDefinition
    from jacopy.central.tangent.schouten import (
        LieOnMultivectorDefinition,
        SNExpansionDefinition,
        SNOrientationDefinition,
    )

    engine = ExpansionEngine(
        [
            ActOverSumOpDefinition(),
            LieBracketActionDefinition(),
            ScalarActAsMultiplicationDefinition(registry),
            CovariantScalarActionDefinition(registry),
            FrameDualityDefinition(),
            IntrinsicDDefinition(CARTAN_TM, registry),
            IntrinsicLDefinition(CARTAN_TM, registry),
            InteriorActDefinition(registry),
            InteriorEvalDefinition(registry),
            HeadSumDefinition(),
            HeadScalarDefinition(registry),
            PairingVectorScalarDefinition(registry),
            SlotNegDefinition(),
            SlotZeroDefinition(),
            FrameBracketCoefficientDefinition(),
            SymAltEvalDefinition(),
            # Theorem-backed canonical form (see the class docstring):
            # orientation normalization for Lie-bracket atoms.
            BracketOrientationDefinition(),
            SNOrientationDefinition(registry),
            SNExpansionDefinition(registry),
            LieOnMultivectorDefinition(registry),
        ],
        mode=mode,
    )
    for fr in holonomic_frames:
        engine.register(HolonomicFrameDefinition(fr))
    return engine
