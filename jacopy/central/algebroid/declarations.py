"""
Declared-axiom rules of the algebroid hierarchy (Phase 3.B).

Each atomic declaration of :data:`~jacopy.central.algebroid.context.DECLARATIONS`
maps to one engine rule, registered **per algebroid instance and only
when declared** — the mechanized form of "I am working in an
almost-Leibniz / local / pre-Leibniz / Leibniz algebroid". Undeclared
axioms simply have no rule: the shapes they govern stay inert and
proofs that need them fail with an honest residual.

The rules (papers: B = pre-metric-bourbaki.pdf, MC =
metric-connection.pdf):

* ``right-leibniz`` [B 4.4 / MC 3.10]:
  ``[u, fv]_E → ρ(u)(f)·v + f·[u, v]_E``.
* ``left-leibniz`` [MC 3.11]:
  ``[fu, v]_E → −ρ(v)(f)·u + f·[u, v]_E + L(Df, u, v)`` — introduces
  the (still opaque) locality term; the locality structure itself is
  Phase 3.E.
* ``anchor-morphism`` [B 4.5 / MC 3.19]:
  ``ρ([u, v]_E) → [ρ(u), ρ(v)]_Lie``.
* ``jacobi`` (Leibniz-Jacobi, [B 4.6]): carried by the
  **Jacobiator node** — ``J^E(u,v,w) → 0``. A naive bracket-level
  rewrite ``[u,[v,w]] → [[u,v],w] + [v,[u,w]]`` would not terminate
  (it reproduces its own shape), so the declaration rewrites the
  defect operator instead; the node's definitional expansion
  (:class:`~jacopy.central.algebroid.operators.JacobiatorExpansionDefinition`)
  is registered at LOWER precedence, so with the declaration on, the
  vanishing wins.

Every declared-rule ``ProofStep`` carries the rule name including the
algebroid and the axiom (``"right-Leibniz (E): …"``) — a closed
proof's step list is therefore also its exact assumption record.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.algebroid.context import (
    Algebroid,
    AlgebroidBracket,
    AnchoredVF,
    EMetric,
    locality_term,
)
from jacopy.central.algebroid.operators import Jacobiator


def _scalar_split(expr: Expr, registry) -> Optional[tuple]:
    """``(f, rest)`` when ``expr = f · rest`` with a certainly-scalar
    leading factor, else ``None``."""
    if isinstance(expr, Product) and len(expr.children) >= 2:
        head = expr.children[0]
        if is_scalar_function(head, registry):
            rest = expr.children[1:]
            return head, (rest[0] if len(rest) == 1 else Product(*rest))
    return None


class RightLeibnizDeclaration(Definition):
    """``[u, fv]_E → ρ(u)(f)·v + f·[u, v]_E`` (declared axiom)."""

    anchor = AlgebroidBracket

    def __init__(
        self, alg: Algebroid, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._alg = alg
        self._registry = registry
        self.name = f"right-Leibniz ({alg.name}): [u, fv] = ρ(u)(f)v + f[u,v]"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._alg.name
            and _scalar_split(expr.v, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        f, v = _scalar_split(expr.v, self._registry)
        u = expr.u
        return Sum(
            Product(Act(self._alg.anchor(u), f), v),
            Product(f, AlgebroidBracket(self._alg.name, u, v)),
        )


class LeftLeibnizDeclaration(Definition):
    """``[fu, v]_E → −ρ(v)(f)·u + f·[u, v]_E + L(Df, u, v)``
    (declared axiom; the locality term stays opaque until 3.E)."""

    anchor = AlgebroidBracket

    def __init__(
        self, alg: Algebroid, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._alg = alg
        self._registry = registry
        self.name = (
            f"left-Leibniz ({alg.name}): "
            "[fu, v] = −ρ(v)(f)u + f[u,v] + L(Df,u,v)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._alg.name
            and _scalar_split(expr.u, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        f, u = _scalar_split(expr.u, self._registry)
        v = expr.v
        return Sum(
            Neg(Product(Act(self._alg.anchor(v), f), u)),
            Product(f, AlgebroidBracket(self._alg.name, u, v)),
            locality_term(self._alg, f, u, v),
        )


class AnchorMorphismDeclaration(Definition):
    """``ρ([u, v]_E) → [ρ(u), ρ(v)]_Lie`` (declared axiom — the
    "pre" property; PROVABLE from right-Leibniz + Jacobi, which is the
    Phase 3.D theorem)."""

    anchor = AnchoredVF

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = f"anchor morphism ({alg.name}): ρ([u,v]) = [ρ(u), ρ(v)]"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AnchoredVF)
            and expr.algebroid_name == self._alg.name
            and isinstance(expr.section, AlgebroidBracket)
            and expr.section.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        br = expr.section
        return LieBracketVF(
            self._alg.anchor(br.u), self._alg.anchor(br.v)
        )


class JacobiDeclaration(Definition):
    """``J^E(u, v, w) → 0`` (declared Leibniz-Jacobi identity, carried
    by the Jacobiator node)."""

    anchor = Jacobiator

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = f"Leibniz-Jacobi ({alg.name}): J(u,v,w) = 0"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Jacobiator)
            and expr.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Integer

        return Integer(0)


class AntisymmetryDeclaration(Definition):
    """``[v, u]_E → −[u, v]_E`` (canonical slot order) and
    ``[u, u]_E → 0`` — the declared antisymmetry axiom that turns a
    Leibniz bracket into a Lie(-algebroid) bracket. The algebroid
    counterpart of the tangent layer's bracket-orientation rule;
    unlike the metric symmetric-part axiom the swap carries no
    correction term, so the plain orientation rewrite terminates.
    Fires on plain section slots only — linearity/Leibniz splits keep
    precedence via registration order."""

    anchor = AlgebroidBracket

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"antisymmetry ({alg.name}): [v,u] = −[u,v], [u,u] = 0"
        )

    def _plain(self, slot: Expr) -> bool:
        return not (
            isinstance(slot, (Sum, Neg, Product)) or slot == Integer(0)
        )

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._alg.name
        ):
            return False
        if not (self._plain(expr.u) and self._plain(expr.v)):
            return False
        return expr.u._repr_inner() >= expr.v._repr_inner()

    def rewrite(self, expr: Expr) -> Expr:
        if expr.u == expr.v:
            return Integer(0)
        return Neg(
            AlgebroidBracket(self._alg.name, expr.v, expr.u)
        )


class MetricInvarianceDeclaration(Definition):
    """``ρ(u)(g(v, w)) → g([u,v], w) + g(v, [u,w])`` (declared axiom
    C1 [B 4.10] — the metric-compatibility of the bracket)."""

    anchor = Act

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"metric invariance ({alg.name}): "
            "ρ(u)(g(v,w)) = g([u,v],w) + g(v,[u,w])"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, AnchoredVF)
            and expr.op.algebroid_name == self._alg.name
            and isinstance(expr.arg, EMetric)
            and expr.arg.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        u = expr.op.section
        v, w = expr.arg.u, expr.arg.v
        g = self._alg.metric
        br = self._alg.bracket
        return Sum(g(br(u, v), w), g(v, br(u, w)))


class SymmetricPartDeclaration(Definition):
    """The symmetric-part axiom C2 [B Def 4.2-4.3]:
    ``[u,v]_E + [v,u]_E = g⁻¹D g(u,v)``, applied in the COLLECTION
    direction: a same-signed pair ``±([u,v] … [v,u])`` inside a sum is
    replaced by ``±g⁻¹D g(u,v)``.

    The single-bracket swap orientation ``[v,u] → −[u,v] + g⁻¹Dg(u,v)``
    is deliberately NOT used: together with metric invariance (C1),
    the sharp evaluation and the coboundary pairing it forms a genuine
    rewrite cycle (``ρ(w)(g(u,v)) → … → ρ(v)(g(u,w)) → … →
    ρ(w)(g(u,v))``), so the system would not terminate. Collection is
    strictly decreasing and closes every equation-form use of C2.

    Two collection shapes, both up to a common cofactor tuple
    (ℝ-bilinearity of the axiom instance):

    * pair: ``c·[a,b] + c·[b,a] → c·g⁻¹D g(a,b)``,
    * equal-slot summand: ``c·[u,u] → ½c·g⁻¹D g(u,u)`` — safe as a
      SUM-TERM rule where the bracket-anchored version would cycle
      with C1: C1's output puts ``[u,u]`` inside EMetric slots, never
      as a section-level summand, so the loop path cannot re-fire it.
    """

    anchor = Sum

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"symmetric part ({alg.name}): [u,v] + [v,u] = g⁻¹D g(u,v)"
        )

    def _signed_bracket(self, term: Expr):
        """``(sign, cofactors, bracket)`` for ``±(c₁·…·[a,b]_E·…)``
        terms carrying exactly one bracket of this algebroid as a
        factor (cofactors may be empty), else ``None``. Pairs with
        identical cofactors collect by ℝ-bilinearity:
        ``c·[a,b] + c·[b,a] = c·g⁻¹D g(a,b)``."""
        sign = 1
        core = term
        if isinstance(core, Neg):
            sign = -1
            core = core.arg
        if (
            isinstance(core, AlgebroidBracket)
            and core.algebroid_name == self._alg.name
        ):
            return sign, (), core
        if isinstance(core, Product):
            brackets = [
                (k, c)
                for k, c in enumerate(core.children)
                if isinstance(c, AlgebroidBracket)
                and c.algebroid_name == self._alg.name
            ]
            if len(brackets) == 1:
                k, br = brackets[0]
                cofactors = tuple(
                    c for m, c in enumerate(core.children) if m != k
                )
                return sign, cofactors, br
        return None

    @staticmethod
    def _cofactor_key(cofactors):
        return tuple(sorted(c._repr_inner() for c in cofactors))

    def _find_collection(self, expr: Sum):
        """``(indices, sign, cofactors, bracket, half)`` for the first
        collectable shape: a transposed same-sign same-cofactor pair,
        else an equal-slot summand (``half=True``)."""
        terms = expr.children
        singleton = None
        for i, ti in enumerate(terms):
            si = self._signed_bracket(ti)
            if si is None:
                continue
            sign_i, co_i, br_i = si
            for j in range(i + 1, len(terms)):
                sj = self._signed_bracket(terms[j])
                if sj is None:
                    continue
                sign_j, co_j, br_j = sj
                if (
                    sign_i == sign_j
                    and br_i.u == br_j.v
                    and br_i.v == br_j.u
                    and self._cofactor_key(co_i)
                    == self._cofactor_key(co_j)
                ):
                    return (i, j), sign_i, co_i, br_i, False
            if singleton is None and br_i.u == br_i.v:
                singleton = (i,), sign_i, co_i, br_i, True
        return singleton

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Sum)
            and self._find_collection(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Rational

        indices, sign, cofactors, br, half = self._find_collection(expr)
        alg = self._alg
        a, b = br.u, br.v
        if a._repr_inner() > b._repr_inner():
            a, b = b, a
        correction: Expr = alg.sharp(alg.D(alg.metric(a, b)))
        factors = ((Rational(1, 2),) if half else ()) + tuple(cofactors)
        if factors:
            correction = Product(*factors, correction)
        if sign < 0:
            correction = Neg(correction)
        rest = [
            t for k, t in enumerate(expr.children) if k not in indices
        ]
        if not rest:
            return correction
        return Sum(*rest, correction)


def declaration_rules(
    alg: Algebroid, registry: Optional[PropertyRegistry] = None
):
    """The engine rules licensed by ``alg.declarations``, in a fixed
    order (registered BEFORE the definitional base so e.g. the
    declared ``J → 0`` outranks the Jacobiator expansion; the
    symmetric-part swap comes LAST so Leibniz/linearity splits win on
    composite slots)."""
    rules = []
    if alg.declares("right-leibniz"):
        rules.append(RightLeibnizDeclaration(alg, registry))
    if alg.declares("left-leibniz"):
        rules.append(LeftLeibnizDeclaration(alg, registry))
    if alg.declares("anchor-morphism"):
        rules.append(AnchorMorphismDeclaration(alg))
    if alg.declares("jacobi"):
        rules.append(JacobiDeclaration(alg))
    if alg.declares("antisymmetric"):
        rules.append(AntisymmetryDeclaration(alg))
    if alg.declares("metric-invariance"):
        rules.append(MetricInvarianceDeclaration(alg))
    if alg.declares("symmetric-part"):
        rules.append(SymmetricPartDeclaration(alg))
    return rules
