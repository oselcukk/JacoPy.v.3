"""Bracket base infrastructure (foundation).

For now this package carries only the **base bracket abstraction**:

* :class:`GradedBracket` — the abstract graded-bracket interface
  (antisymmetry, Leibniz, Jacobi axiom flags),
* :class:`BracketApply` — the inert application of a bracket to two
  operands (the ``B(a, b)`` node),
* :class:`DerivedBracket` / :class:`VanishingCondition` — the
  Q-twisted derived bracket and its vanishing condition.

Concrete brackets (Lie, Schouten-Nijenhuis, Koszul, Dorfman, Courant,
Roytenberg, Vinogradov, …) will be added step by step in v3 under the
**central code** and the **generalized geometry / Poisson** packages,
following the order in the PDF specification.
"""

from jacopy.brackets.base import (
    BracketApply,
    GradedBracket,
    expand_bracket,
)
from jacopy.brackets.derived import (
    DerivedBracket,
    VanishingCondition,
    derived_bracket,
)

__all__ = [
    "GradedBracket",
    "BracketApply",
    "expand_bracket",
    "DerivedBracket",
    "VanishingCondition",
    "derived_bracket",
]
