"""
Symmetry-aware canonical form for alternating multilinear evaluations
(Engine pass E.2).

An alternating :class:`~jacopy.core.multi_eval.MultiEval` satisfies

    ω(…, Y_i, …, Y_j, …) = −ω(…, Y_j, …, Y_i, …),
    ω(…, Y, …, Y, …) = 0,

so every such node has a canonical representative: arguments sorted by
a stable structural key, with the permutation's parity folded out as an
overall sign, and a structural zero whenever two arguments coincide.
Normalizing to that representative makes term cancellation *free* in
``collect_terms`` — ``ω(X, Y) + ω(Y, X)`` becomes
``ω(X, Y) − ω(X, Y)`` and dies — instead of requiring a dedicated
engine rule per orientation (the v2 approach).

Only ``alternating=True`` nodes are touched. ``alternating=False``
declares *no* symmetry (a symmetric evaluation like a metric is merely
not anti-symmetric), so reordering those arguments would assert an
equation nobody stated.
"""

from __future__ import annotations

from jacopy.core.expr import Expr, Integer, Neg
from jacopy.core.multi_eval import MultiEval, has_repeated_arg


def _sort_key(expr: Expr) -> str:
    """Stable structural key for argument ordering."""
    return expr._repr_inner()


def _permutation_parity(before, after) -> int:
    """Parity (0 even / 1 odd) of the permutation taking ``before`` to
    ``after``. Assumes ``after`` is a rearrangement of ``before`` with
    no structurally equal duplicates (the repeated-argument case is
    zeroed before this runs)."""
    remaining = list(after)
    swaps = 0
    for i, item in enumerate(before):
        j = remaining.index(item)
        if j != i:
            remaining[i], remaining[j] = remaining[j], remaining[i]
            swaps += 1
    # ``remaining.index`` over the prefix already fixed never fires
    # because each fixed slot holds the matched element.
    return swaps % 2


def _odd_wedge_degree(child: Expr) -> bool:
    """True when ``child`` certainly has odd degree in the wedge
    grading (its ``wedge_degree`` lift if present, else its own
    ``degree`` attribute). Unknown degrees return False — the Wedge is
    then left unsorted (soundness over completeness)."""
    from jacopy.core.symbolic_degree import Degree

    deg = getattr(child, "wedge_degree", None)
    if not isinstance(deg, Degree):
        deg = getattr(child, "degree", None)
    if not isinstance(deg, Degree):
        return False
    parity = deg.parity()
    return parity == 1


def _certainly_scalar(child: Expr, registry) -> bool:
    """True when ``child`` certainly has wedge-degree exactly 0 (a
    function): such factors turn ``∧`` into plain multiplication and
    commute freely."""
    from jacopy.core.symbolic_degree import Degree

    deg = getattr(child, "wedge_degree", None)
    if not isinstance(deg, Degree):
        deg = getattr(child, "degree", None)
    if not isinstance(deg, Degree):
        try:
            from jacopy.algebra.derivation import degree_of

            deg = degree_of(child, registry)
        except ValueError:
            return False
    return deg == Degree.const(0)


def _normalize_wedge(node: Expr, registry=None) -> Expr:
    """Canonical form for a Wedge: pull child Negs out (multilinearity),
    extract certainly-scalar factors as ordinary coefficients
    (``f ∧ X = f·X`` — degree-0 factors commute freely), and, when
    every remaining child is certainly of odd wedge-degree, sort the
    children antisymmetrically (adjacent swaps flip the sign; a
    repeated child kills the node)."""
    from jacopy.core.expr import Product
    from jacopy.core.wedge import Wedge

    sign = 0
    children = []
    scalars = []
    stack = list(node.children)
    flattened_nested = False
    idx = 0
    while idx < len(stack):
        c = stack[idx]
        idx += 1
        while isinstance(c, Neg):
            sign ^= 1
            c = c.arg
        if isinstance(c, Wedge):
            # Associativity: flatten nested wedges (sign-free) —
            # higher-degree audit 2026-07-31, the SN wedge-Leibniz
            # recursion builds (a ∧ b) ∧ c shapes.
            stack[idx:idx] = list(c.children)
            flattened_nested = True
            continue
        if c == Integer(0):
            # Multilinearity: a zero factor kills the wedge.
            return Integer(0)
        if _certainly_scalar(c, registry):
            scalars.append(c)
            continue
        children.append(c)

    changed_neg = (
        sign == 1
        or bool(scalars)
        or flattened_nested
        or any(isinstance(c0, Neg) for c0 in node.children)
    )

    if all(_odd_wedge_degree(c) for c in children):
        seen = []
        for c in children:
            if any(c == s for s in seen):
                return Integer(0)
            seen.append(c)
        sorted_children = sorted(children, key=_sort_key)
        if sorted_children != children:
            sign ^= _permutation_parity(children, sorted_children)
            children = sorted_children
            changed_neg = True

    if not changed_neg:
        return node
    if not children:
        core: Expr = Product(*scalars) if len(scalars) > 1 else scalars[0]
    else:
        wedge_part: Expr = (
            children[0] if len(children) == 1 else Wedge(*children)
        )
        core = Product(*scalars, wedge_part) if scalars else wedge_part
    return Neg(core) if sign else core


def normalize_alternating(expr: Expr, registry=None) -> Expr:
    """Rewrite every alternating :class:`MultiEval` (and every
    :class:`~jacopy.core.wedge.Wedge` of odd-degree factors) to
    canonical form.

    Bottom-up: children first, so nested evaluations are canonical
    before their parents are inspected. ``registry`` (optional) only
    sharpens scalar-factor detection inside wedges.
    """
    from jacopy.core.wedge import Wedge

    if expr.is_atom:
        return expr
    new_children = tuple(
        normalize_alternating(c, registry) for c in expr.children
    )
    rebuilt = expr._rebuild(new_children)
    if isinstance(rebuilt, Wedge):
        return _normalize_wedge(rebuilt, registry)
    if not isinstance(rebuilt, MultiEval) or not rebuilt.alternating:
        return rebuilt
    if has_repeated_arg(rebuilt):
        return Integer(0)
    args = list(rebuilt.args)
    sorted_args = sorted(args, key=_sort_key)
    if sorted_args == args:
        return rebuilt
    parity = _permutation_parity(args, sorted_args)
    canonical = rebuilt.with_args(*sorted_args)
    return Neg(canonical) if parity == 1 else canonical
