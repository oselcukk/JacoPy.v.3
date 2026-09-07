"""
Graded Leibniz expansion.

Rewrites every :class:`Act` node ``D(x)`` by pushing the operator
through sums (R-linearity) and products (graded Leibniz):

* ``D(a + b) → D(a) + D(b)``
* ``D(−a) → −D(a)``
* ``D(a*b*c) → D(a)*b*c + (−1)^{|D||a|} a*D(b)*c
               + (−1)^{|D|(|a|+|b|)} a*b*D(c)``

For the Leibniz expansion the degrees of the factors must be
determinable, either because the factor is a :class:`Derivation`
(self-describing), a numeric literal (degree 0), or registered as
:class:`Scalar`/:class:`Graded` in the supplied registry. An
undetermined factor raises :class:`ValueError`: the modelling gap is
something the user needs to fix, not something we should silently paper
over with a zero-degree assumption.

The pass is bottom-up and inert on nodes that aren't :class:`Act` at
their head: inner structure inside the argument is expanded first, so
``D(E(a*b))`` first expands the inner ``E`` Leibniz then the outer
``D``. Outer derivation degrees see the *expanded* operand structure.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.algorithms.base import Algorithm
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Public entry                                                           #
# --------------------------------------------------------------------- #


def product_rule(
    expr: Expr, registry: Optional[PropertyRegistry] = None
) -> Expr:
    """Recursively expand every ``Act`` via graded Leibniz and linearity.

    Bottom-up: children are expanded first, so inner ``Act`` nodes are
    simplified before their outer enclosing ``Act`` looks at them.
    Atoms pass through untouched.
    """
    if expr.is_atom:
        return expr

    new_children = tuple(product_rule(c, registry) for c in expr.children)

    if isinstance(expr, Act):
        op, arg = new_children
        return _expand_act(op, arg, registry)

    return expr._rebuild(new_children)


# --------------------------------------------------------------------- #
# Act-specific expansion                                                 #
# --------------------------------------------------------------------- #


def _expand_act(
    op: Expr, arg: Expr, registry: Optional[PropertyRegistry]
) -> Expr:
    """Apply linearity + graded Leibniz to a single ``Act`` node.

    When ``op`` is itself a :class:`Product` of operators, i.e. an
    explicit composition ``D1 ∘ D2 ∘ … ∘ Dn``, the application is
    unfolded right-to-left into nested ``Act`` nodes and each layer is
    expanded in turn. That is the only place in the algorithm where
    derivation composition actually acts on an operand; at
    :class:`Act` construction it stays inert.

    Scalar-level linearity on the operator side, ``Neg(op)`` and the
    zero operator ``Integer(0)``, is peeled here too. Commutator
    expansion routinely generates ``Act(Neg(compose(D1, D2)), x)``
    shapes; pulling the sign out and recursing lets the composition
    unfold through the negation instead of stalling.
    """
    if isinstance(op, Neg):
        return Neg(_expand_act(op.arg, arg, registry))
    if isinstance(op, Sum):
        # Operator-side additivity: a sum of operators acts pointwise,
        # (D1 + D2)(x) = D1(x) + D2(x). Slot rewrites routinely leave
        # Sum-headed operators inside Act nodes (e.g. a sharp whose
        # 1-form slot expanded into a sum).
        return Sum.make(
            *(_expand_act(o, arg, registry) for o in op.children)
        )
    if isinstance(op, Integer) and op == Integer(0):
        return Integer(0)
    if isinstance(arg, Integer) and arg == Integer(0):
        # Every graded derivation is linear, so D(0) = 0 regardless of
        # the specific operator. Dropping this eagerly keeps axiom
        # residues like Act(d, Act(d², x))=Act(d, 0) from surviving into
        # later passes that would otherwise stall waiting for an engine
        # rewrite that never fires on this surviving shape.
        return Integer(0)
    if isinstance(arg, (Integer, Rational)) and isinstance(op, Derivation):
        # Derivations annihilate numeric constants: D(1) = D(1·1) =
        # 2·D(1) forces D(1) = 0, and linearity gives D(c) = 0. This is
        # what makes ℝ-homogeneity [c·X, Y] = c·[X, Y] close — the
        # Leibniz split of Y(c·g) leaves a Y(c)·g term that must die.
        return Integer(0)
    if getattr(arg, "is_constant", False) and isinstance(op, Derivation):
        # The is_constant protocol: atoms marking themselves constant
        # (e.g. the Kronecker delta) are likewise annihilated. This is
        # what closes frame identities like de^c(e_a, e_b) = -γ^c_ab,
        # whose Palais unroll produces e_a(δ^c_b) terms.
        return Integer(0)
    if isinstance(op, Product) and op.children:
        result = arg
        for d in reversed(op.children):
            result = _expand_act(d, result, registry)
        return result
    if not isinstance(op, Derivation):
        # A degree-0 non-derivation "operator" is a scalar coefficient
        # (compositions like f·X put scalars in operator position): it
        # acts by multiplication, NEVER by Leibniz — splitting s(ab)
        # into s(a)b + a·s(b) double-counts to 2s·ab.
        try:
            if degree_of(op, registry) == Degree.const(0):
                return Product(op, arg)
        except ValueError:
            pass
    if isinstance(arg, Sum):
        return Sum.make(
            *(_expand_act(op, c, registry) for c in arg.children)
        )
    if isinstance(arg, Neg):
        inner = _expand_act(op, arg.arg, registry)
        return Neg(inner)
    if isinstance(arg, Product):
        if getattr(op, "leibniz", True) is False:
            # LINEAR-ONLY operator (e.g. the multivector interior
            # ι_P for p ≥ 2 — a COMPOSITION of derivations, not a
            # derivation): scalar factors pull out, no Leibniz
            # split (6.H.2 soundness catch, 2026-09-07).
            from jacopy.central.calculus.scalars import (
                is_scalar_function,
            )

            scalars = [
                c
                for c in arg.children
                if is_scalar_function(c, registry)
            ]
            rest = [
                c
                for c in arg.children
                if not is_scalar_function(c, registry)
            ]
            if scalars and rest:
                core = (
                    rest[0]
                    if len(rest) == 1
                    else Product(*rest)
                )
                return Product(
                    *scalars, _expand_act(op, core, registry)
                )
            return Act(op, arg)
        return _expand_leibniz(op, arg, registry)
    from jacopy.core.wedge import Wedge as _Wedge

    if isinstance(arg, _Wedge):
        # Graded Leibniz over the WEDGE product (the wedge is the
        # graded product of forms — Phase 5.E.4):
        # D(α∧β) = D(α)∧β + (−1)^{|D||α|} α∧D(β) + …
        return _expand_wedge_leibniz(op, arg, registry)
    # Atom or other compound (Power, Act, Commutator, ...): leave inert.
    return Act(op, arg)


def _expand_wedge_leibniz(op: Expr, wedge: Expr, registry) -> Expr:
    """``D(a₁∧…∧a_n) → Σ_i sign_i · a₁∧…∧D(a_i)∧…∧a_n`` with
    ``sign_i = (−1)^{|D|·(|a₁|+…+|a_{i−1}|)}`` — the Wedge twin of
    :func:`_expand_leibniz`."""
    from jacopy.core.wedge import Wedge as _Wedge

    factors = tuple(wedge.children)
    deg_op = degree_of(op, registry)
    deg_op_is_zero = deg_op == Degree.const(0)
    running = Degree.const(0)
    terms: List[Expr] = []
    for i, fac in enumerate(factors):
        if deg_op_is_zero:
            parity = 0
        else:
            sign_exp = deg_op * running
            parity = sign_exp.parity()
            if parity is None:
                raise ValueError(
                    f"Cannot expand wedge Leibniz at factor {i} of "
                    f"{wedge!r}: sign parity is symbolic."
                )
        new_factors = list(factors)
        new_factors[i] = _expand_act(op, fac, registry)
        term: Expr = (
            _Wedge(*new_factors)
            if len(new_factors) > 1
            else new_factors[0]
        )
        if parity == 1:
            term = Neg(term)
        terms.append(term)
        if not deg_op_is_zero and i + 1 < len(factors):
            running = running + _factor_degree(fac, registry)
    return Sum.make(*terms)


def _expand_leibniz(
    op: Expr, prod: Product, registry: Optional[PropertyRegistry]
) -> Expr:
    """``D(a1*...*an) → Σ_i sign_i * a1*...*D(a_i)*...*an``.

    ``sign_i = (−1)^{|D| * (|a1|+...+|a_{i−1}|)}``. Undecidable sign
    parity raises, the caller needs to narrow down the degrees. Scalar
    factors (degree 0) contribute nothing to the running sign, so
    symbolic-degree factors only cause trouble when they actually
    precede a splitting point.
    """
    factors: Tuple[Expr, ...] = prod.children
    if not factors:
        # Empty product, D(1) = 0. Shouldn't normally arise but
        # handle defensively; Sum.make([]) returns Zero.
        return Integer(0)

    deg_op = degree_of(op, registry)
    # When the derivation has degree 0 the sign vanishes identically,
    # so we don't need factor degrees at all. This is the common case
    # for ordinary (ungraded) derivations and spares the caller from
    # declaring a grading for operand factors that don't have one.
    deg_op_is_zero = deg_op == Degree.const(0)

    running = Degree.const(0)  # |a_1| + ... + |a_{i-1}|
    terms: List[Expr] = []
    for i, fac in enumerate(factors):
        if deg_op_is_zero:
            parity = 0
        else:
            sign_exp = deg_op * running
            parity = sign_exp.parity()
            if parity is None:
                raise ValueError(
                    f"Cannot expand Leibniz at factor index {i} of "
                    f"{prod!r}: sign parity is symbolic. Narrow the "
                    "degrees of the left-of-split factors."
                )
        new_factors = list(factors)
        # Recurse on the freshly created Act (the composition branch
        # already does): numeric factors annihilate and Sum factors
        # distribute in the same pass instead of stranding
        # ``D(2)``/``D(a+b)`` nodes into the simplify pipeline.
        new_factors[i] = _expand_act(op, fac, registry)
        # (running-degree bookkeeping uses _factor_degree below)
        term: Expr = Product(*new_factors) if len(new_factors) > 1 else new_factors[0]
        if parity == 1:
            term = Neg(term)
        terms.append(term)
        if not deg_op_is_zero and i + 1 < len(factors):
            running = running + _factor_degree(fac, registry)

    return Sum.make(*terms)


def _factor_degree(fac: Expr, registry) -> Degree:
    """``degree_of`` plus the homogeneous-Sum policy the sign
    bookkeeping needs (degree_of deliberately leaves Sums to the
    caller): a Sum whose children all share one determinable degree
    has that degree."""
    try:
        return degree_of(fac, registry)
    except ValueError:
        if isinstance(fac, Sum) and fac.children:
            degs = []
            for c in fac.children:
                degs.append(_factor_degree(c, registry))
            first = degs[0]
            if all(d == first for d in degs[1:]):
                return first
        raise


# --------------------------------------------------------------------- #
# Algorithm wrapper                                                      #
# --------------------------------------------------------------------- #


class ProductRule(Algorithm):
    """:class:`Algorithm` wrapper around :func:`product_rule`.

    ``registry`` is stored on the instance so the generic
    :meth:`Algorithm.run` plumbing stays registry-free.
    """

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def can_apply(self, expr: Expr) -> bool:
        for node in expr.walk():
            if isinstance(node, Act):
                return True
        return False

    def apply(self, expr: Expr) -> Expr:
        return product_rule(expr, self._registry)
