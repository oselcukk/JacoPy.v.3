"""
Conservative scalar-function recognition (shared by calculus rules).

Moved here from :mod:`jacopy.central.tangent.definitions` (which
re-exports it) so the generic calculus layer can use it without
importing the tangent instantiation.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum, Symbol
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


def is_scalar_function(
    expr: Expr, registry: Optional[PropertyRegistry] = None
) -> bool:
    """Conservative check: is ``expr`` certainly a ``C^∞(M)`` function?

    Recognized shapes (recursively): registered degree-0 scalar
    symbols, numeric literals, :class:`Pairing` / :class:`MultiEval`
    (scalars by construction), a degree-0 :class:`Derivation` applied
    to a scalar function (``X(f)``, ``∇_X f``, ``[X,Y](f)``,
    ``L_X f``), and sums / products / negations of scalar functions.
    Returns ``False`` on anything unrecognized — soundness over
    completeness.
    """
    if isinstance(expr, (Integer, Rational)):
        return True
    if isinstance(expr, (Pairing, MultiEval)):
        return True
    if isinstance(expr, Symbol):
        if registry is None:
            return False
        try:
            return degree_of(expr, registry) == Degree.const(0)
        except ValueError:
            return False
    if expr.is_atom and not isinstance(expr, Derivation):
        # Atoms carrying their own degree (Kronecker delta, anholonomy
        # coefficients, 0-forms) are functions exactly when that degree
        # is 0. Derivations are operators, never functions.
        deg = getattr(expr, "degree", None)
        if isinstance(deg, Degree):
            return deg == Degree.const(0)
    if isinstance(expr, Neg):
        return is_scalar_function(expr.arg, registry)
    if isinstance(expr, (Sum, Product)):
        return all(is_scalar_function(c, registry) for c in expr.children)
    if isinstance(expr, Act):
        op = expr.op
        return (
            isinstance(op, Derivation)
            and op.degree == Degree.const(0)
            and is_scalar_function(expr.arg, registry)
        )
    return False
