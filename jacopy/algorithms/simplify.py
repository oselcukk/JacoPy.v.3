"""
Full simplification pipeline.

Composes the Faz 2 passes into a single fix-point driver:

    flatten → distribute → canonicalize → sort_product → collect_terms

The canonicalize step handles per-node normalizations that the other
passes skip: Neg folding (``−(−x) → x``, ``−(a + b) → (−a) + (−b)``),
Power trivialities, numeric-coefficient consolidation in Products.
The Neg-over-Sum distribution in particular is what allows
cancellations trapped inside a ``Neg(Sum(...))`` envelope to reach
``collect_terms``, without it, identities like Lie Jacobi fail to
close even after full expansion.

The Koszul sort step requires a :class:`PropertyRegistry`, pass one
to enable it, or leave ``registry=None`` to skip sorting entirely (the
rest of the pipeline is registry-free). Skipping sort is useful when
factors are intentionally left in non-canonical order or when grading
hasn't been declared yet.

The pipeline iterates until a full pass makes no change, with a
configurable :attr:`max_iterations` safeguard against a misbehaving
rule cycling forever. The default of 64 is deliberately generous for
the algorithm mix we have, each pass is monotone enough in practice
that two or three rounds suffice.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algorithms.canonicalize import canonicalize
from jacopy.algorithms.collect_terms import collect_terms
from jacopy.algorithms.distribute import distribute
from jacopy.algorithms.factor_terms import collect_pairings, factor_common
from jacopy.algorithms.flatten import flatten
from jacopy.algorithms.normalize_alternating import normalize_alternating
from jacopy.algorithms.sort_product import apply_sign, sort_product
from jacopy.core.expr import Expr, Product
from jacopy.core.registry import PropertyRegistry


def _node_count(expr: Expr) -> int:
    return sum(1 for _ in expr.walk())


def simplify(
    expr: Expr,
    registry: Optional[PropertyRegistry] = None,
    *,
    max_iterations: int = 64,
) -> Expr:
    """Run the full simplification pipeline to a fix-point.

    If ``registry`` is provided, every :class:`Product` subtree is
    passed through :func:`sort_product` with decidable signs folded
    via :func:`apply_sign`. Symbolic signs are left as-is for now,
    the caller should run :func:`sort_product` directly if they need
    the :class:`Degree` polynomial.

    After the expansion-oriented fix-point, a **factoring phase**
    (:func:`factor_common`) is attempted: it is the inverse direction
    of ``distribute``, so it runs outside the loop and its result is
    kept only when strictly smaller (or structurally zero). This
    catches residuals of the shape ``f·A + f·B`` without ever
    ping-ponging against ``distribute``.
    """
    current = expr
    for _ in range(max_iterations):
        # 1. Flatten associative chains.
        step = flatten(current)
        # 2. Normalize Neg / numeric folds. In particular, Neg-over-Sum
        #    distribution here produces new Sums that the subsequent
        #    ``distribute`` pass then pushes through any enclosing
        #    Products. Running canonicalize *before* distribute is what
        #    lets identities like Lie Jacobi close.
        step = canonicalize(step)
        # 2b. Alternating canonical form: sorted arguments + parity
        #     sign + repeated-argument zero. Makes orientation-flipped
        #     evaluation terms cancel in collect_terms for free.
        step = normalize_alternating(step, registry)
        # 3. Multiply out Products over Sums.
        step = distribute(step)
        # 4. Re-flatten, distribute can leave freshly exposed
        #    associative chains (a * (b*c) branches that were behind
        #    a Sum) that would confuse sort_product.
        step = flatten(step)
        if registry is not None:
            step = _sort_all_products(step, registry)
        step = collect_terms(step)
        if step == current:
            break
        current = step

    # Final collecting phases (outside the expansion loop; see
    # docstring). Both run in the reverse direction of ``distribute``,
    # so they only keep their result under the smaller-or-zero
    # acceptance criterion.
    for pass_fn in (factor_common, collect_pairings):
        for _ in range(8):
            candidate = pass_fn(current)
            if candidate == current:
                break
            candidate = canonicalize(collect_terms(flatten(candidate)))
            if _node_count(candidate) < _node_count(current):
                current = candidate
            else:
                break
    return current


def _sort_all_products(expr: Expr, registry: PropertyRegistry) -> Expr:
    """Apply :func:`sort_product` recursively, folding decidable signs.

    A Product whose sign parity is symbolic is left unsorted: we
    cannot fold an unknown ``(-1)^{|α||β|}`` into the tree without a
    dedicated representation, and silently dropping it would be
    wrong. :func:`sort_product` still gets called so grading is
    validated even in that case.
    """
    if expr.is_atom:
        return expr
    new_children = tuple(
        _sort_all_products(c, registry) for c in expr.children
    )
    rebuilt = expr._rebuild(new_children) if new_children else expr
    if isinstance(rebuilt, Product):
        sorted_expr, sign_exp = sort_product(rebuilt, registry)
        parity = sign_exp.parity()
        if parity is None:
            # Sign is symbolic, leave the Product in its pre-sort
            # order. An unsorted-but-valid product is strictly better
            # than a wrong one.
            return rebuilt
        return apply_sign(sorted_expr, sign_exp)
    return rebuilt
