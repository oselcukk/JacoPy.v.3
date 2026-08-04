"""
The Schouten-Nijenhuis bracket ``[·,·]_SN`` of multivector fields
(PDF item 9f; Phase 2.F).

**Canonical definition (decomposable formula, v2 sign convention).**
In the shifted grading ``s(A) = m(A) − 1`` (``m`` = multivector
degree: functions 0, vector fields 1, ``X ∧ Y`` 2, …) the bracket is
fixed by four base cases and wedge-Leibniz recursion:

    [X, Y]_SN   = [X, Y]_Lie,          [f, g]_SN = 0,
    [X, f]_SN   = X(f),                [f, X]_SN = −X(f),
    [X∧Y, Z]_SN = X ∧ [Y,Z] + (−1)^{s(Z)·m(Y)} [X,Z] ∧ Y,
    [Z, X∧Y]_SN = [Z,X] ∧ Y + (−1)^{s(Z)·m(X)} X ∧ [Z,Y],

with the Gerstenhaber-Leibniz exponent mixing the SHIFTED degree of
the outer operand and the UNSHIFTED degree of the passed-over factor
(audit fix 2026-07-23 — the s·s variant inherited from v2 agreed on
all-vector operands but broke the alternation of ``[X∧Y, f]``).

Unfolding the recursion reproduces the closed double-sum decomposable
formula ``Σ_{i,j} (−1)^{i+j} [X_i, Y_j] ∧ …``. The graded
antisymmetry ``[A,B] = −(−1)^{s(A)s(B)}[B,A]``, the graded Jacobi
identity and the ``∧``-Leibniz property are **theorems**.

Multivector wedges are :class:`~jacopy.core.wedge.Wedge` nodes of
vector fields (the Wedge lift: ``|X∧Y| = 2``); v2 used raw ``Product``
nodes here — the Wedge representation is the v3 upgrade that keeps
multivectors in the same node family as forms.

``m`` is computed structurally (:func:`multivector_degree`); the
bracket node itself is a foundation
:class:`~jacopy.brackets.base.BracketApply` under the singleton
:data:`SN`. The engine rule :class:`SNExpansionDefinition` performs
the definitional expansion; unsupported shapes (symbolic degree,
opaque higher atoms like a bare Poisson bivector) stay inert — the
Poisson layer (Phase 5) reasons about those symbolically.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, degree_of
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.multivector import PVector
from jacopy.central.objects.vector_field import VectorField
from jacopy.central.tangent.lie_bracket import lie_bracket


class SchoutenBracket(GradedBracket):
    """``[·,·]_SN`` — degree 0 in the shifted grading, graded
    antisymmetric, graded Leibniz, graded Jacobi (the latter three as
    theorem obligations, not axioms)."""

    def __init__(self) -> None:
        super().__init__(
            "[·,·]_SN",
            degree=0,
            is_graded_antisymmetric=True,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=True,
        )

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """Definitional expansion; unsupported shapes stay opaque."""
        out = _sn_expand(a, b, registry)
        return out if out is not None else BracketApply(self, a, b)

    def apply_degree(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Degree:
        """``m([a, b]_SN) = m(a) + m(b) − 1`` in the MULTIVECTOR
        grading (audit fix 2026-07-23: the operator-grading default
        |a|+|b|+deg would report m([π,π]) = 4 instead of 3, because a
        vector field's operator degree is 0 while its multivector
        degree is 1)."""
        m_a = multivector_degree(a, registry)
        m_b = multivector_degree(b, registry)
        if m_a is None or m_b is None:
            raise ValueError(
                "SN bracket degree needs determinable multivector "
                f"degrees; got {a!r}, {b!r}"
            )
        return Degree.const(m_a + m_b - 1)


#: The singleton SN bracket.
SN = SchoutenBracket()


def sn_bracket(a: Expr, b: Expr) -> BracketApply:
    """``[a, b]_SN`` — an inert node; expansion is definitional."""
    if not isinstance(a, Expr) or not isinstance(b, Expr):
        raise TypeError("sn_bracket expects Expr arguments")
    return BracketApply(SN, a, b)


def multivector_degree(
    expr: Expr, registry: Optional[PropertyRegistry] = None
) -> Optional[int]:
    """The multivector degree ``m`` (functions 0, vectors 1, wedges
    additive), or ``None`` when undeterminable.

    Structural: vector fields and their brackets are 1-vectors,
    :class:`PVector` carries its own degree, a :class:`Wedge` sums its
    children, a certain scalar function is 0. Anything else — symbolic
    degrees included — is ``None`` (the SN expansion then stays inert).
    """
    if isinstance(expr, Neg):
        return multivector_degree(expr.arg, registry)
    if isinstance(expr, (VectorField, LieBracketVF)):
        return 1
    if isinstance(expr, PVector):
        return expr.degree.as_int()
    if isinstance(expr, Wedge):
        total = 0
        for c in expr.children:
            m = multivector_degree(c, registry)
            if m is None:
                return None
            total += m
        return total
    if isinstance(expr, BracketApply) and expr.bracket is SN:
        # |[a, b]_SN| = |a| + |b| − 1 (higher-degree audit
        # 2026-07-31: the bracket node must report its own degree so
        # downstream bookkeeping sees p + q − 1).
        ma = multivector_degree(expr.a, registry)
        mb = multivector_degree(expr.b, registry)
        if ma is None or mb is None:
            return None
        return ma + mb - 1
    if is_scalar_function(expr, registry):
        return 0
    return None


class SNExpansionDefinition(Definition):
    """The canonical definitional expansion of ``[a, b]_SN``."""

    anchor = BracketApply

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry
        self.name = (
            "Schouten-Nijenhuis definition: base cases + wedge-Leibniz "
            "recursion (shifted grading, v2 sign convention)"
        )

    # ---- Definition API ----------------------------------------------- #

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, BracketApply)
            and isinstance(expr.bracket, SchoutenBracket)
            and _sn_expand(expr.a, expr.b, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        out = _sn_expand(expr.a, expr.b, self._registry)
        assert out is not None  # guaranteed by matches()
        return out


def _sn_expand(
    a: Expr, b: Expr, registry: Optional[PropertyRegistry]
) -> Optional[Expr]:
    """The shared recursive expansion (base cases + wedge Leibniz)."""
    if True:
        m_a = multivector_degree(a, registry)
        m_b = multivector_degree(b, registry)
        if m_a is None or m_b is None:
            return None

        # Base cases.
        if not isinstance(a, Wedge) and not isinstance(b, Wedge):
            if m_a == 1 and m_b == 1:
                return lie_bracket(a, b)
            if m_a == 0 and m_b == 0:
                return Integer(0)
            if m_a == 1 and m_b == 0:
                return Act(a, b)
            if m_a == 0 and m_b == 1:
                return Neg(Act(b, a))

        # Wedge-Leibniz, first slot (Gerstenhaber sign — note the
        # exponent mixes the SHIFTED degree of the outer operand with
        # the UNSHIFTED degree of the tail):
        # [X∧Y, Z] = X ∧ [Y,Z] + (−1)^{s(Z)·m(Y)} [X,Z] ∧ Y.
        # For all-vector operands s(Z) = 0 and every sign is +; the
        # s·s variant carried over from v2 agreed there but broke the
        # alternation of [X∧Y, f] (audit finding, 2026-07-23).
        if isinstance(a, Wedge) and len(a.children) >= 2:
            X = a.children[0]
            tail = a.children[1:]
            Y = tail[0] if len(tail) == 1 else Wedge(*tail)
            m_Y = multivector_degree(Y, registry)
            if m_Y is None:
                return None
            s_Z = m_b - 1
            inner1 = _sn_expand(Y, b, registry)
            inner2 = _sn_expand(X, b, registry)
            if inner1 is None or inner2 is None:
                return None
            term1: Expr = Wedge.make(X, inner1)
            term2: Expr = Wedge.make(inner2, Y)
            if (s_Z * m_Y) % 2:
                term2 = Neg(term2)
            return Sum(term1, term2)

        # Wedge-Leibniz, second slot:
        # [Z, X∧Y] = [Z,X] ∧ Y + (−1)^{s(Z)·m(X)} X ∧ [Z,Y].
        if isinstance(b, Wedge) and len(b.children) >= 2:
            X = b.children[0]
            tail = b.children[1:]
            Y = tail[0] if len(tail) == 1 else Wedge(*tail)
            m_X = multivector_degree(X, registry)
            if m_X is None:
                return None
            s_Z = m_a - 1
            inner1 = _sn_expand(a, X, registry)
            inner2 = _sn_expand(a, Y, registry)
            if inner1 is None or inner2 is None:
                return None
            term1 = Wedge.make(inner1, Y)
            term2 = Wedge.make(X, inner2)
            if (s_Z * m_X) % 2:
                term2 = Neg(term2)
            return Sum(term1, term2)

        return None


# --------------------------------------------------------------------- #
# Theorems (PDF item 9f)                                                 #
# --------------------------------------------------------------------- #


def _engine(registry):
    from jacopy.central.tangent.engine import tangent_engine

    return tangent_engine(registry=registry)


def prove_reduces_to_lie(
    X: Expr, Y: Expr, f: Expr, *, registry=None
) -> ProofChain:
    """``[X, Y]_SN (f) = [X, Y]_Lie (f)`` — on 1-vectors SN is the Lie
    bracket (PDF 8a-style reduction, on a generic function)."""
    lhs = Act(sn_bracket(X, Y), f)
    rhs = Act(lie_bracket(X, Y), f)
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_sn_vector_function(
    X: Expr, f: Expr, *, registry=None
) -> ProofChain:
    """``[X, f]_SN = X(f)`` and (by the same expansion) the shifted
    antisymmetry ``[f, X]_SN = −X(f)``: proved as
    ``[X, f]_SN + [f, X]_SN = 0``."""
    lhs = Sum(sn_bracket(X, f), sn_bracket(f, X))
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=_engine(registry)
    )


def prove_sn_wedge_leibniz(
    X: Expr, Y: Expr, Z: Expr, *, registry=None
) -> ProofChain:
    """``[X, Y ∧ Z]_SN = [X, Y] ∧ Z + Y ∧ [X, Z]`` — the Leibniz
    property of a vector field over the wedge (no signs: ``s(X)=0``)."""
    lhs = sn_bracket(X, Wedge(Y, Z))
    rhs = Sum(
        Wedge(lie_bracket(X, Y), Z),
        Wedge(Y, lie_bracket(X, Z)),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_sn_graded_antisymmetry(
    X: Expr, Y: Expr, Z: Expr, *, registry=None
) -> ProofChain:
    """``[X∧Y, Z]_SN = −(−1)^{s(X∧Y)s(Z)} [Z, X∧Y]_SN = −[Z, X∧Y]_SN``
    (here ``s(X∧Y) = 1``, ``s(Z) = 0``).

    Closes through the canonical bracket orientation (theorem-backed):
    the two expansions produce opposite-orientation Lie brackets inside
    the wedges, which normalize against each other.
    """
    lhs = sn_bracket(Wedge(X, Y), Z)
    rhs = Neg(sn_bracket(Z, Wedge(X, Y)))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


class SNOrientationDefinition(Definition):
    """Canonical slot order for the SN bracket, with the GRADED sign
    (higher-degree audit 2026-07-31):

        [a, b]_SN → −(−1)^{(p−1)(q−1)} [b, a]_SN
                                    when repr(a) > repr(b),

    for determinable multivector degrees ``p, q``. The sign is the
    graded antisymmetry of the Schouten-Nijenhuis bracket — VERIFIED
    mechanically on decomposables at (1,1), (2,1), (2,2), (3,2),
    (3,3) before this rule was admitted (the rule is the
    normal-form orientation of that theorem, the BracketOrientation
    pattern of Phase 2). Fires at most once per node (the rewrite is
    order-canonical); inert on equal slots and undeterminable
    degrees."""

    anchor = BracketApply

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry
        self.name = (
            "SN orientation: [a,b]_SN = −(−1)^{(p−1)(q−1)}[b,a]_SN "
            "(canonical order)"
        )

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, BracketApply) and expr.bracket is SN
        ):
            return False
        a, b = expr.a, expr.b
        if a == b:
            return False
        if a._repr_inner() <= b._repr_inner():
            return False
        return (
            multivector_degree(a, self._registry) is not None
            and multivector_degree(b, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        a, b = expr.a, expr.b
        p = multivector_degree(a, self._registry)
        q = multivector_degree(b, self._registry)
        swapped = BracketApply(SN, b, a)
        if (p - 1) * (q - 1) % 2 == 0:
            return Neg(swapped)
        return swapped


class LieOnMultivectorDefinition(Definition):
    """``L_X P → [X, P]_SN`` for a multivector ``P`` of degree ≥ 2 —
    the canonical extension of the Lie derivative to multivectors
    (PDF item 9e; closes the audit gap where ``L_X P`` stayed
    inert). Degree ≤ 1 is untouched: functions and vector fields
    keep their intrinsic-L branches."""

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        from jacopy.algebra.derivation import Act as _Act

        self.anchor = _Act
        self._registry = registry
        self.name = (
            "Lie derivative on multivectors: L_X P = [X, P]_SN "
            "(degree ≥ 2)"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.algebra.derivation import Act as _Act
        from jacopy.central.calculus.bracket_calculus import (
            LieDerivative,
        )
        from jacopy.central.tangent.exterior import CARTAN_TM

        if not isinstance(expr, _Act):
            return False
        op = expr.op
        if not (
            isinstance(op, LieDerivative)
            and op.calculus_name == CARTAN_TM.name
        ):
            return False
        m = multivector_degree(expr.arg, self._registry)
        return m is not None and m >= 2

    def rewrite(self, expr: Expr) -> Expr:
        return sn_bracket(expr.op.vector, expr.arg)
