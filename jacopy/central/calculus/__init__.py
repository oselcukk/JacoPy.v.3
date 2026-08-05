"""Generic bracket calculus (definition policy §4).

The intrinsic-``d`` and ``L`` machinery parameterized by an
``(anchor, bracket)`` pair, plus the calculus-independent interior
slot-insertion rules. Instantiations: TM = ``(id, Lie)`` (Phase 2),
algebroid = ``(ρ_E, [·,·]_E)`` (Phase 3), Poisson = ``(π^♯, Koszul)``
(Phase 5).
"""

from jacopy.central.calculus.bracket_calculus import (
    ActExpansionDefinition,
    HeadFormProductLiftDefinition,
    BracketCalculus,
    ExteriorDerivative,
    IntrinsicDDefinition,
    IntrinsicLDefinition,
    HeadScalarDefinition,
    HeadSumDefinition,
    LieDerivative,
    MultiEvalArgLinearityDefinition,
    PairingVectorScalarDefinition,
    SlotNegDefinition,
    SlotZeroDefinition,
)
from jacopy.central.calculus.symmetrize_rules import SymAltEvalDefinition
from jacopy.central.calculus.interior_rules import (
    InteriorActDefinition,
    InteriorEvalDefinition,
    InteriorVectorLinearityDefinition,
)
from jacopy.central.calculus.scalars import is_scalar_function

__all__ = [
    "ActExpansionDefinition",
    "HeadFormProductLiftDefinition",
    "BracketCalculus",
    "ExteriorDerivative",
    "IntrinsicDDefinition",
    "IntrinsicLDefinition",
    "LieDerivative",
    "HeadScalarDefinition",
    "HeadSumDefinition",
    "MultiEvalArgLinearityDefinition",
    "PairingVectorScalarDefinition",
    "SlotNegDefinition",
    "SlotZeroDefinition",
    "SymAltEvalDefinition",
    "InteriorActDefinition",
    "InteriorEvalDefinition",
    "InteriorVectorLinearityDefinition",
    "is_scalar_function",
]
