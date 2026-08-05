"""
Common-factor collection over sums (Engine pass E.2).

The main pipeline is expansion-oriented (``distribute`` always
multiplies out), which is the right default for cancellation — but it
leaves sums like ``f·A + f·B`` in fully expanded form. This pass pulls
the shared *leading* factor back out: ``f·A + f·B → f·(A + B)``.

Only the leading (leftmost) factor is grouped, which is always sound
for the non-commutative graded products in this codebase: factoring
from the left never reorders anything.

The pass itself is order-reversing w.r.t. ``distribute``, so it must
NOT run inside the same fix-point loop (they would ping-pong). It is
applied by :func:`~jacopy.algorithms.simplify.simplify` as a *final
phase* with an acceptance criterion — the factored candidate (after a
cheap re-collect) is kept only when it is strictly smaller or reaches
structural zero.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from jacopy.core.expr import Expr, Neg, Product, Sum


def _split_term(term: Expr) -> Tuple[int, Tuple[Expr, ...]]:
    """Decompose ``term`` into ``(sign, factors)``; sign is 0/1 for +/−."""
    sign = 0
    while isinstance(term, Neg):
        sign ^= 1
        term = term.arg
    if isinstance(term, Product):
        return sign, tuple(term.children)
    return sign, (term,)


def _rebuild_term(sign: int, factors: Tuple[Expr, ...]) -> Expr:
    if len(factors) == 1:
        core = factors[0]
    else:
        core = Product(*factors)
    return Neg(core) if sign else core


def factor_common(expr: Expr) -> Expr:
    """Group sum terms sharing a leading factor: ``f·A + f·B → f·(A+B)``.

    Bottom-up over the tree. Terms whose whole body *is* the candidate
    factor (no remaining factors) are left ungrouped — ``f + f·A`` is
    not rewritten to ``f·(1 + A)``, that shape rarely helps and would
    need a unit literal.
    """
    if expr.is_atom:
        return expr
    new_children = tuple(factor_common(c) for c in expr.children)
    rebuilt = expr._rebuild(new_children)
    if not isinstance(rebuilt, Sum):
        return rebuilt

    groups: Dict[Expr, List[Tuple[int, Tuple[Expr, ...]]]] = {}
    order: List[Expr] = []
    passthrough: List[Expr] = []
    for term in rebuilt.children:
        sign, factors = _split_term(term)
        if len(factors) < 2:
            passthrough.append(term)
            continue
        lead = factors[0]
        if lead not in groups:
            groups[lead] = []
            order.append(lead)
        groups[lead].append((sign, factors[1:]))

    changed = False
    new_terms: List[Expr] = []
    for lead in order:
        members = groups[lead]
        if len(members) < 2:
            new_terms.extend(_rebuild_term(s, (lead,) + rest) for s, rest in members)
            continue
        changed = True
        inner = Sum.make(*(_rebuild_term(s, rest) for s, rest in members))
        new_terms.append(Product(lead, inner))
    new_terms.extend(passthrough)

    if not changed:
        return rebuilt
    return Sum.make(*new_terms)


def collect_pairings(expr: Expr) -> Expr:
    """Collect evaluations sharing a head: ``Σ sᵢ⟨α, Vᵢ⟩ → ⟨α, Σ sᵢVᵢ⟩``
    and ``Σ sᵢ η(Y…, Vᵢ) → η(Y…, Σ sᵢVᵢ)`` (last slot).

    Multilinearity of the pairing / evaluation primitives, applied in
    the collecting direction. This is what turns a residual like
    ``⟨ω,[[X,Y],Z]⟩ − ⟨ω,[[X,Z],Y]⟩ + ⟨ω,[[Y,Z],X]⟩`` into
    ``⟨ω, (Jacobi sum)⟩`` — a shape a cited vector-level bracket
    identity can then rewrite away. MultiEval terms group by
    ``(head, all-but-last args)`` — after the alternating canonical
    sort, compound (bracket) slots land last, so the varying slot is
    the last one. Like :func:`factor_common` it runs in simplify's
    *final phase* with a smaller-or-zero acceptance criterion.
    """
    from jacopy.core.multi_eval import MultiEval
    from jacopy.core.pairing import Pairing

    if expr.is_atom:
        return expr
    new_children = tuple(collect_pairings(c) for c in expr.children)
    rebuilt = expr._rebuild(new_children)
    if not isinstance(rebuilt, Sum):
        return rebuilt

    # Group key -> (rebuilder, [(sign, varying slot)]).
    groups: Dict[Tuple, List[Tuple[int, Expr]]] = {}
    keyinfo: Dict[Tuple, Expr] = {}
    order: List[Tuple] = []
    passthrough: List[Expr] = []
    from jacopy.core.expr import Product as _Product

    for term in rebuilt.children:
        sign = 0
        core = term
        while isinstance(core, Neg):
            sign ^= 1
            core = core.arg
        # Scalar COEFFICIENTS fold into the varying slot
        # (``s·⟨α,V⟩`` contributes ``s·V`` — the full bilinearity
        # the docstring promises; 2026-08-04, the Dorfman
        # right-Leibniz residual needed the coefficient case).
        coeffs: tuple = ()
        if isinstance(core, _Product) and len(core.children) >= 2:
            evals = [
                c
                for c in core.children
                if isinstance(c, (Pairing, MultiEval))
            ]
            rest = [
                c
                for c in core.children
                if not isinstance(c, (Pairing, MultiEval))
            ]
            if len(evals) == 1:
                coeffs = tuple(rest)
                core = evals[0]
        if isinstance(core, Pairing):
            key = ("pairing", core.alpha)
            varying = core.X
        elif isinstance(core, MultiEval) and core.arity >= 1:
            key = (
                "multieval",
                core.head,
                core.args[:-1],
                core.alternating,
                core.slot_kind,
            )
            varying = core.args[-1]
        else:
            passthrough.append(term)
            continue
        if coeffs:
            varying = _Product(*coeffs, varying)
        if key not in groups:
            groups[key] = []
            keyinfo[key] = core
            order.append(key)
        groups[key].append((sign, varying))

    def _rebuild_one(key: Tuple, slot: Expr) -> Expr:
        proto = keyinfo[key]
        if key[0] == "pairing":
            return Pairing(proto.alpha, slot)
        return proto.with_args(*(proto.args[:-1] + (slot,)))

    changed = False
    new_terms: List[Expr] = []
    for key in order:
        members = groups[key]
        if len(members) < 2:
            for s, v in members:
                node = _rebuild_one(key, v)
                new_terms.append(Neg(node) if s else node)
            continue
        changed = True
        inner = Sum.make(*((Neg(v) if s else v) for s, v in members))
        new_terms.append(_rebuild_one(key, inner))
    new_terms.extend(passthrough)

    if not changed:
        return rebuilt
    return Sum.make(*new_terms)
