"""
IndexedSum and Wedge engine rules (Phase 4.D.2).

The core nodes (:class:`~jacopy.core.indexed_sum.IndexedSum`,
:class:`~jacopy.core.wedge.Wedge`) are inert structural containers;
these are their definitional semantics:

* :class:`IndexedSumLinearityDefinition` — ``Σ_s (a + b) → Σ_s a +
  Σ_s b``, ``Neg`` pull-out, ``Σ_s 0 → 0``, and pulling out factors
  free of the bound index.
* :class:`IndexedSumEvalPushInDefinition` — evaluation commutes with
  the (finite) sum: ``(Σ_s ω_s)(X…) → Σ_s ω_s(X…)`` (guarded: the
  arguments must not capture the bound index).
* :class:`WedgeEvalDefinition` — the determinant convention on two
  1-forms: ``(α ∧ β)(X, Y) → α(X)β(Y) − α(Y)β(X)`` (matching the
  Palais convention fixed by the definition policy; higher arities
  arrive when a consumer needs them).
* :class:`KroneckerContractionDefinition` — ``Σ_s δ^s_c · t(s) →
  t(c)`` (and ``Σ_s δ^s_c → 1``): the bound index contracts against a
  Kronecker delta via the name-based substitution protocol.

Index occurrence is detected through the ``index_names`` protocol on
indexed atoms (frame fields, coefficients, deltas), walking children
AND operator-atom slots (``rewritable_slots``).
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.proof.expansion import Definition


BOUND_CANDIDATES = ("s", "t", "u", "v", "w", "x", "y", "z")


def collect_index_names(expr: Expr, into=None) -> set:
    """All index names mentioned anywhere in ``expr`` (via the
    ``index_names`` protocol; children, operator-atom slots and
    operator heads included)."""
    if into is None:
        into = set()
    names = getattr(expr, "index_names", None)
    if names is not None:
        into.update(names)
    for c in expr.children:
        collect_index_names(c, into)
    slots = getattr(expr, "rewritable_slots", None)
    if slots:
        for s in slots:
            collect_index_names(s, into)
    op = getattr(expr, "op", None)
    if isinstance(op, Expr):
        collect_index_names(op, into)
    return into


def fresh_bound_name(*taken_exprs: Expr, extra=()) -> str:
    """A bound-index name free in all the given expressions."""
    taken = set(extra)
    for e in taken_exprs:
        collect_index_names(e, taken)
    for name in BOUND_CANDIDATES:
        if name not in taken:
            return name
    raise ValueError(
        "no fresh bound-index name available; rename the free indices"
    )


def contains_index(expr: Expr, name: str) -> bool:
    """Does ``expr`` mention the index ``name`` (via the
    ``index_names`` protocol), anywhere — children and operator-atom
    slots included?"""
    names = getattr(expr, "index_names", None)
    if names is not None and name in names:
        return True
    for c in expr.children:
        if contains_index(c, name):
            return True
    slots = getattr(expr, "rewritable_slots", None)
    if slots:
        for s in slots:
            if contains_index(s, name):
                return True
    op = getattr(expr, "op", None)
    if isinstance(op, Expr) and contains_index(op, name):
        return True
    return False


class IndexedSumLinearityDefinition(Definition):
    """``Σ_s (a + b) → Σ_s a + Σ_s b``; ``Neg``/zero; factors free of
    the bound index pull out."""

    name = "indexed sum linearity: Σ_s(a + b) = Σ_s a + Σ_s b, free factors pull out"
    anchor = IndexedSum

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, IndexedSum):
            return False
        body = expr.body
        if isinstance(body, (Sum, Neg)) or body == Integer(0):
            return True
        if isinstance(body, Product):
            dummy_name = expr.dummy._repr_inner()
            # Both a free factor (to pull) AND a bound one (to keep):
            # an all-free body needs the symbolic dimension and stays
            # inert; matching it would loop (caught in 4.E.2).
            return any(
                not contains_index(c, dummy_name) for c in body.children
            ) and any(
                contains_index(c, dummy_name) for c in body.children
            )
        return False

    def rewrite(self, expr: Expr) -> Expr:
        body = expr.body

        def rebuilt(new_body: Expr) -> Expr:
            return IndexedSum(expr.dummy, expr.range_, new_body)

        if body == Integer(0):
            return Integer(0)
        if isinstance(body, Sum):
            return Sum(*(rebuilt(c) for c in body.children))
        if isinstance(body, Neg):
            return Neg(rebuilt(body.arg))
        dummy_name = expr.dummy._repr_inner()
        free = [
            c for c in body.children if not contains_index(c, dummy_name)
        ]
        bound = [c for c in body.children if contains_index(c, dummy_name)]
        if not bound:
            # Nothing depends on the index: a symbolic-dimension count
            # would be needed; leave inert (honest diagnostics beat a
            # wrong `n·body`).
            return expr
        inner = bound[0] if len(bound) == 1 else Product(*bound)
        return Product(*free, rebuilt(inner))


class IndexedSumEvalPushInDefinition(Definition):
    """``(Σ_s ω_s)(X…) → Σ_s ω_s(X…)`` — evaluation commutes with the
    finite sum (guarded against index capture in the arguments)."""

    name = "indexed sum evaluation: (Σ_s ω_s)(X…) = Σ_s ω_s(X…)"
    anchor = (Pairing, MultiEval)

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Pairing):
            head = expr.alpha
        elif isinstance(expr, MultiEval):
            head = expr.head
        else:
            return False
        return isinstance(head, IndexedSum)

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.objects.frame import FrameIndex

        if isinstance(expr, Pairing):
            head, args = expr.alpha, (expr.X,)
        else:
            head, args = expr.head, expr.args
        dummy_name = head.dummy._repr_inner()
        if any(contains_index(a, dummy_name) for a in args):
            # α-convert first: the arguments mention the bound name.
            fresh = fresh_bound_name(head.body, *args)
            head = head.with_dummy(FrameIndex(fresh))
            if isinstance(expr, Pairing):
                return Pairing(head, expr.X)
            return MultiEval(
                head,
                *expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        if isinstance(expr, Pairing):
            new_body = Pairing(head.body, expr.X)
        else:
            new_body = MultiEval(
                head.body,
                *expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        return IndexedSum(head.dummy, head.range_, new_body)


class WedgeEvalDefinition(Definition):
    """``(α₁ ∧ … ∧ α_n)(X₁, …, X_n) → Σ_σ sgn(σ) Π_i ⟨α_i, X_{σ(i)}⟩``
    — the determinant convention on n 1-form factors (definition
    policy §1). Generalized from the 2-factor case in Phase 5.E.4
    (closing the "Wedge n-factor eval" deferral); the 1-form guard is
    ``degree_of``-based so composite factors like ``d(f)`` qualify."""

    name = (
        "wedge evaluation: (α₁∧…∧α_n)(X₁,…,X_n) = det⟨α_i, X_j⟩"
    )
    anchor = MultiEval

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _is_one_form(self, factor: Expr) -> bool:
        from jacopy.algebra.derivation import degree_of

        deg = getattr(factor, "degree", None)
        if isinstance(deg, Degree) and deg == Degree.const(1):
            return True
        try:
            return degree_of(factor, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        head = expr.head
        return (
            isinstance(head, Wedge)
            and len(head.children) >= 2
            and expr.arity == len(head.children)
            and all(self._is_one_form(c) for c in head.children)
        )

    def rewrite(self, expr: Expr) -> Expr:
        from itertools import permutations

        factors = expr.head.children
        args = expr.args
        n = len(factors)
        terms = []
        for perm in permutations(range(n)):
            # Permütasyon paritesi (çift/tek transpozisyon sayısı).
            parity = 0
            seen = [False] * n
            for start in range(n):
                if seen[start]:
                    continue
                length = 0
                j = start
                while not seen[j]:
                    seen[j] = True
                    j = perm[j]
                    length += 1
                parity ^= (length - 1) & 1
            term: Expr = Product(
                *(
                    Pairing(factors[i], args[perm[i]])
                    for i in range(n)
                )
            )
            terms.append(Neg(term) if parity else term)
        return Sum(*terms)


class KroneckerContractionDefinition(Definition):
    """``Σ_s δ^s_c · t(s) → t(c)`` — the bound index contracts against
    a Kronecker delta (name-based substitution)."""

    name = "Kronecker contraction: Σ_s δ^s_c · t(s) = t(c)"
    anchor = IndexedSum

    def _delta_split(self, expr: "IndexedSum"):
        from jacopy.central.objects.frame import KroneckerDelta

        dummy_name = expr.dummy._repr_inner()
        body = expr.body
        sign = False
        if isinstance(body, Neg):
            sign = True
            body = body.arg
        factors = (
            list(body.children) if isinstance(body, Product) else [body]
        )
        for k, c in enumerate(factors):
            if not isinstance(c, KroneckerDelta):
                continue
            if c.upper == dummy_name and not c.lower == dummy_name:
                other = c.lower
            elif c.lower == dummy_name and not c.upper == dummy_name:
                other = c.upper
            else:
                continue
            rest = factors[:k] + factors[k + 1 :]
            return sign, rest, other
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, IndexedSum)
            and self._delta_split(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import One  # singleton instance
        from jacopy.central.objects.frame import FrameIndex

        sign, rest, other = self._delta_split(expr)
        if not rest:
            out: Expr = One
        else:
            inner = rest[0] if len(rest) == 1 else Product(*rest)
            out = inner.substitute_atom(
                expr.dummy, FrameIndex(other)
            )
        return Neg(out) if sign else out
