"""
The research interface (Phase 8 — PDF item 15): the layer on which a
user writes *their own* question. Five bricks, each born from a piece
that ``examples/bracket_twist_walkthrough.ipynb`` had to do by hand:

* :mod:`sections` — typed generalized sections ``U ⊕ ω₂ ⊕ ω₅``;
* :mod:`block_matrix` — block-operator matrices with derived action,
  product and (unipotent) inverse;
* :mod:`engine_assembly` — build the rewrite engine from the
  expressions to be checked, with a self-explaining report;
* :mod:`axiom_suite` — the generic Courant-type condition runner and
  the Ψ-transport of a whole structure;
* :mod:`antisymmetrize` — index antisymmetrisation for frame formulas.
"""

from jacopy.research.antisymmetrize import (
    antisymmetrize,
    permutation_sign,
    rename_indices,
    swap_indices,
)
from jacopy.research.axiom_suite import (
    AlgebroidData,
    AxiomSuite,
    Bracket,
    CheckResult,
    SuiteReport,
)
from jacopy.research.block_matrix import (
    BlockMatrix,
    Identity,
    Map,
    Operator,
    Zero,
    as_operator,
)
from jacopy.research.engine_assembly import (
    Scan,
    WedgeGradedOrderDefinition,
    WedgeScalarFactorDefinition,
    assemble_engine,
    assembly_report,
)
from jacopy.research.sections import (
    GeneralizedSection,
    SectionType,
    Slot,
)

__all__ = [
    "AlgebroidData",
    "AxiomSuite",
    "BlockMatrix",
    "Bracket",
    "CheckResult",
    "GeneralizedSection",
    "Identity",
    "Map",
    "Operator",
    "Scan",
    "SectionType",
    "Slot",
    "SuiteReport",
    "WedgeGradedOrderDefinition",
    "WedgeScalarFactorDefinition",
    "Zero",
    "antisymmetrize",
    "as_operator",
    "assemble_engine",
    "assembly_report",
    "permutation_sign",
    "rename_indices",
    "swap_indices",
]
