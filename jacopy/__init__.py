"""jacopy (v3) — graded algebra & bracket calculus with step-by-step proofs.

The v3 architecture follows the PDF specification (``Math 595 - JacoPy``):

* **Foundation** (preserved from v2) — ``core`` (expr tree),
  ``algorithms`` (simplify / product-rule / distribute …), ``algebra``
  (Act, Derivation), ``proof`` (ProofChain engine), ``brackets`` (the
  base ``GradedBracket`` abstraction).
* **central** — basic objects valid on both TM and an algebroid E
  (items 8-10). *(under construction)*
* **packages** — metric-affine, Poisson, algebroid calculus,
  generalized geometry, research interface (items 11-15).
  *(under construction)*

For now only the core expression tree (`Expr` and its offspring) is
exported at the top level; the object factories (`Functions`,
`VectorFields`, `Forms`, …) will arrive together with the central code.
"""

from jacopy.core.expr import (
    Expr,
    Symbol,
    Integer,
    Rational,
    Sum,
    Product,
    Power,
    Neg,
    Zero,
    One,
    NegOne,
)

__version__ = "0.0.1"

__all__ = [
    "Expr",
    "Symbol",
    "Integer",
    "Rational",
    "Sum",
    "Product",
    "Power",
    "Neg",
    "Zero",
    "One",
    "NegOne",
]
