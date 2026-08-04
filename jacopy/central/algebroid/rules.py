"""
Definitional rules of the algebroid layer (Phases 3.A + 3.E.1).

Exactly the structure that holds *by definition* on any anchored
vector bundle with an ℝ-bilinear bracket — nothing from the declared
hierarchy (right-Leibniz etc., Phase 3.B) appears here:

* :class:`AnchorLinearityDefinition` — ``ρ`` is a bundle morphism,
  hence ``C^∞``-linear: ``ρ(u + v) = ρ(u) + ρ(v)``,
  ``ρ(fu) = f·ρ(u)``, ``ρ(−u) = −ρ(u)``, ``ρ(0) = 0``.
* :class:`BracketBilinearityDefinition` — ``[·,·]_E`` is ℝ-bilinear
  [B Def 4.1]: sums, negations and *numeric* scalars pull out of both
  slots. ``C^∞`` factors do **not** — ``[u, fv]`` and ``[fu, v]``
  stay inert until right-/left-Leibniz is declared (3.B). This
  distinction is the whole point of the layer split.
* :class:`CoboundaryPairingDefinition` — ``⟨Df, u⟩ = ρ(u)(f)``
  [MC Def 3.5], the defining evaluation of ``D = ρ* ∘ d``.
* :class:`LocalityMultilinearityDefinition` — the locality operator
  is ``C^∞``-multilinear in all three slots BY DEFINITION
  [MC Def 3.5] (Phase 3.E.1).
* :class:`CoboundaryLinearityDefinition` — ``D(f·g) = f·Dg + g·Df``,
  ``D(f+g) = Df + Dg``, ``D(c) = 0``. These are THEOREMS: the
  coboundary is defined only through its pairing ``⟨Df,u⟩ = ρ(u)(f)``,
  and each rule follows by evaluating against a generic section
  (agreement on generators). The rule is theorem-classified and its
  proof builder derives exactly that argument — from an engine that
  does NOT contain this rule, so the derivation is non-circular
  (Phase 3.E.1).
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.algebroid.context import (
    Algebroid,
    AlgebroidBracket,
    AnchoredVF,
    CoboundaryForm,
    EMetric,
    LocalityOperator,
    MetricSharp,
)


def _split_real_linear(expr: Expr):
    """Decompose ``expr`` for ℝ-linear pull-out: returns
    ``(kind, payload)`` where kind is one of ``"sum"``, ``"neg"``,
    ``"num"`` (numeric coefficient × rest), or ``None``."""
    if isinstance(expr, Sum):
        return "sum", expr.children
    if isinstance(expr, Neg):
        return "neg", expr.arg
    if isinstance(expr, Product) and len(expr.children) >= 2:
        head = expr.children[0]
        if isinstance(head, (Integer, Rational)):
            rest = expr.children[1:]
            rest_expr = rest[0] if len(rest) == 1 else Product(*rest)
            return "num", (head, rest_expr)
    return None, None


class AnchorLinearityDefinition(Definition):
    """``C^∞``-linearity of the anchor (definitional: ρ is a bundle
    morphism). Fires on :class:`AnchoredVF` atoms whose section slot
    is a sum, negation, zero, or a scalar multiple — including smooth
    (``C^∞``) coefficients."""

    name = "anchor linearity: ρ(fu + v) = f·ρ(u) + ρ(v)"
    anchor = AnchoredVF

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        """``(scalar, rest)`` when ``expr`` is a Product with a
        certainly-scalar leading factor, else ``None``."""
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                rest_expr = rest[0] if len(rest) == 1 else Product(*rest)
                return head, rest_expr
        return None

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, AnchoredVF):
            return False
        sec = expr.section
        if isinstance(sec, (Sum, Neg)) or sec == Integer(0):
            return True
        return self._scalar_split(sec) is not None

    def rewrite(self, expr: Expr) -> Expr:
        alg_name = expr.algebroid_name
        sec = expr.section

        def rho(s: Expr) -> Expr:
            return AnchoredVF(alg_name, s)

        if sec == Integer(0):
            return Integer(0)
        if isinstance(sec, Sum):
            return Sum(*(rho(c) for c in sec.children))
        if isinstance(sec, Neg):
            return Neg(rho(sec.arg))
        scalar, rest = self._scalar_split(sec)
        return Product(scalar, rho(rest))


class BracketBilinearityDefinition(Definition):
    """ℝ-bilinearity of ``[·,·]_E`` (definitional [B Def 4.1]).

    Sums, negations, zeros and **numeric** coefficients pull out of
    both slots. Smooth-function coefficients deliberately do NOT —
    ``[u, fv]`` is right-Leibniz territory (declared in 3.B)."""

    name = "bracket ℝ-bilinearity: [u + cv, w] = [u,w] + c[v,w] (c ∈ ℝ)"
    anchor = AlgebroidBracket

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, AlgebroidBracket):
            return False
        for slot in (expr.u, expr.v):
            kind, _ = _split_real_linear(slot)
            if kind is not None or slot == Integer(0):
                return True
        return False

    def rewrite(self, expr: Expr) -> Expr:
        alg_name = expr.algebroid_name

        def br(a: Expr, b: Expr) -> Expr:
            return AlgebroidBracket(alg_name, a, b)

        u, v = expr.u, expr.v
        if u == Integer(0) or v == Integer(0):
            return Integer(0)
        kind, payload = _split_real_linear(u)
        if kind == "sum":
            return Sum(*(br(c, v) for c in payload))
        if kind == "neg":
            return Neg(br(payload, v))
        if kind == "num":
            coeff, rest = payload
            return Product(coeff, br(rest, v))
        kind, payload = _split_real_linear(v)
        if kind == "sum":
            return Sum(*(br(u, c) for c in payload))
        if kind == "neg":
            return Neg(br(u, payload))
        coeff, rest = payload
        return Product(coeff, br(u, rest))


class CoboundaryPairingDefinition(Definition):
    """``⟨Df, u⟩ → ρ(u)(f)`` — the defining evaluation of the
    coboundary [MC Def 3.5]. Constructed per algebroid (the rewrite
    needs the context to build the anchor)."""

    anchor = Pairing

    def __init__(self, alg: Algebroid) -> None:
        if not isinstance(alg, Algebroid):
            raise TypeError("CoboundaryPairingDefinition expects an Algebroid")
        self._alg = alg
        self.name = f"coboundary ({alg.name}): ⟨Df, u⟩ = ρ(u)(f)"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Pairing)
            and isinstance(expr.alpha, CoboundaryForm)
            and expr.alpha.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        f = expr.alpha.function
        return Act(self._alg.anchor(expr.X), f)


class LocalityMultilinearityDefinition(Definition):
    """``C^∞``-multilinearity of the locality operator (definitional
    [MC Def 3.5]): sums, negations, zeros and smooth-function factors
    pull out of ALL THREE slots —
    ``L(fω + Υ, u, v) = f·L(ω,u,v) + L(Υ,u,v)`` and likewise in the
    two section slots."""

    name = "locality multilinearity: L(fω+Υ, u, v) = f·L(ω,u,v) + L(Υ,u,v) (all slots)"
    anchor = LocalityOperator

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    def _splittable(self, slot: Expr) -> bool:
        if isinstance(slot, (Sum, Neg)) or slot == Integer(0):
            return True
        return self._scalar_split(slot) is not None

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, LocalityOperator) and any(
            self._splittable(s) for s in expr.rewritable_slots
        )

    def rewrite(self, expr: Expr) -> Expr:
        slots = tuple(expr.rewritable_slots)
        for i, slot in enumerate(slots):
            if not self._splittable(slot):
                continue

            def rebuild(sub: Expr) -> Expr:
                new = slots[:i] + (sub,) + slots[i + 1 :]
                return expr.with_slots(*new)

            if slot == Integer(0):
                return Integer(0)
            if isinstance(slot, Sum):
                return Sum(*(rebuild(c) for c in slot.children))
            if isinstance(slot, Neg):
                return Neg(rebuild(slot.arg))
            scalar, rest = self._scalar_split(slot)
            return Product(scalar, rebuild(rest))
        raise AssertionError("rewrite called without a splittable slot")


class MetricBilinearityDefinition(Definition):
    """``C^∞``-bilinearity of the E-metric (definitional: g is a
    (0,2)-tensor [MC Def 3.1]): sums, negations, zeros and
    smooth-function factors pull out of both slots."""

    name = "metric bilinearity: g(fu + w, v) = f·g(u,v) + g(w,v) (both slots)"
    anchor = EMetric

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    @staticmethod
    def _is_pushable_indexed_sum(slot: Expr, other: Expr) -> bool:
        # g(x, Σ_s b) = Σ_s g(x, b) — additivity over an indexed sum,
        # sound only when the OTHER slot is free of the bound index
        # (capture guard; Phase 4.F.2b).
        from jacopy.core.indexed_sum import IndexedSum
        from jacopy.central.calculus.indexed_rules import (
            contains_index,
        )

        return isinstance(slot, IndexedSum) and not contains_index(
            other, slot.dummy._repr_inner()
        )

    def _splittable(self, slot: Expr) -> bool:
        if isinstance(slot, (Sum, Neg)) or slot == Integer(0):
            return True
        return self._scalar_split(slot) is not None

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, EMetric):
            return False
        slots = tuple(expr.rewritable_slots)
        if any(self._splittable(s) for s in slots):
            return True
        return self._is_pushable_indexed_sum(
            slots[0], slots[1]
        ) or self._is_pushable_indexed_sum(slots[1], slots[0])

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.indexed_sum import IndexedSum

        slots = tuple(expr.rewritable_slots)
        for i, slot in enumerate(slots):
            other = slots[1 - i]

            def rebuild(sub: Expr) -> Expr:
                new = slots[:i] + (sub,) + slots[i + 1 :]
                return expr.with_slots(*new)

            if self._is_pushable_indexed_sum(slot, other):
                return IndexedSum(
                    slot.dummy, slot.range_, rebuild(slot.body)
                )
            if not self._splittable(slot):
                continue
            if slot == Integer(0):
                return Integer(0)
            if isinstance(slot, Sum):
                return Sum(*(rebuild(c) for c in slot.children))
            if isinstance(slot, Neg):
                return Neg(rebuild(slot.arg))
            scalar, rest = self._scalar_split(slot)
            return Product(scalar, rebuild(rest))
        raise AssertionError("rewrite called without a splittable slot")


class MetricSymmetryDefinition(Definition):
    """``g(v, u) → g(u, v)`` — canonical slot order (the metric is
    symmetric BY DEFINITION [MC Def 3.1]; part of the canonical-form
    family, so proofs never stall on slot order)."""

    name = "metric symmetry: g(v, u) = g(u, v) (canonical order)"
    anchor = EMetric

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, EMetric)
            and expr.u._repr_inner() > expr.v._repr_inner()
        )

    def rewrite(self, expr: Expr) -> Expr:
        return expr.with_slots(expr.v, expr.u)


class MetricSharpEvaluationDefinition(Definition):
    """``g(g⁻¹(ω), w) → ⟨ω, w⟩`` — the defining evaluation of the
    inverse metric (either slot; well-defined by non-degeneracy)."""

    name = "inverse metric: g(g⁻¹(ω), w) = ⟨ω, w⟩"
    anchor = EMetric

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, EMetric):
            return False
        return any(
            isinstance(s, MetricSharp)
            and s.algebroid_name == expr.algebroid_name
            and s.metric_name == expr.metric_name
            for s in expr.rewritable_slots
        )

    def rewrite(self, expr: Expr) -> Expr:
        for a, b in ((expr.u, expr.v), (expr.v, expr.u)):
            if (
                isinstance(a, MetricSharp)
                and a.algebroid_name == expr.algebroid_name
                and a.metric_name == expr.metric_name
            ):
                return Pairing(a.form, b)
        raise AssertionError("rewrite called without a sharp slot")


class MetricSharpLinearityDefinition(Definition):
    """``C^∞``-linearity of the inverse metric (definitional):
    ``g⁻¹(fω + Υ) = f·g⁻¹(ω) + g⁻¹(Υ)``, ``g⁻¹(0) = 0``."""

    name = "inverse-metric linearity: g⁻¹(fω + Υ) = f·g⁻¹(ω) + g⁻¹(Υ)"
    anchor = MetricSharp

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MetricSharp):
            return False
        form = expr.form
        if isinstance(form, (Sum, Neg)) or form == Integer(0):
            return True
        return self._scalar_split(form) is not None

    def rewrite(self, expr: Expr) -> Expr:
        form = expr.form
        if form == Integer(0):
            return Integer(0)
        if isinstance(form, Sum):
            return Sum(*(expr.with_slots(c) for c in form.children))
        if isinstance(form, Neg):
            return Neg(expr.with_slots(form.arg))
        scalar, rest = self._scalar_split(form)
        return Product(scalar, expr.with_slots(rest))


class CoboundaryLinearityDefinition(Definition):
    """Linearity + Leibniz of the coboundary — THEOREMS from the
    defining pairing ``⟨Df, u⟩ = ρ(u)(f)``:

    * ``D(f + g) = Df + Dg``, ``D(−f) = −Df``,
    * ``D(c) = 0`` for numeric ``c`` (also ``D(c·f) = c·Df``),
    * ``D(f·g) = f·Dg + g·Df`` (both factors certainly scalar).

    Each follows by pairing both sides against a generic section and
    using ``ρ(u)``'s derivation properties; the proof builder derives
    exactly that, from an engine WITHOUT this rule (non-circular)."""

    anchor = CoboundaryForm

    def __init__(
        self, alg: Algebroid, registry: Optional[PropertyRegistry] = None
    ) -> None:
        if not isinstance(alg, Algebroid):
            raise TypeError(
                "CoboundaryLinearityDefinition expects an Algebroid"
            )
        self._alg = alg
        self._registry = registry
        self.name = (
            f"coboundary linearity + Leibniz ({alg.name}): "
            "D(f·g) = f·Dg + g·Df, D(f+g) = Df + Dg, D(c) = 0"
        )

    def _scalar_leibniz_split(self, expr: Expr):
        """``(x, rest)`` for the Leibniz case — both certainly scalar."""
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            rest = expr.children[1:]
            rest_expr = rest[0] if len(rest) == 1 else Product(*rest)
            if is_scalar_function(head, self._registry) and is_scalar_function(
                rest_expr, self._registry
            ):
                return head, rest_expr
        return None

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, CoboundaryForm)
            and expr.algebroid_name == self._alg.name
        ):
            return False
        f = expr.function
        if isinstance(f, (Sum, Neg, Integer, Rational)):
            return True
        if isinstance(f, Product) and isinstance(
            f.children[0], (Integer, Rational)
        ):
            return True
        return self._scalar_leibniz_split(f) is not None

    def rewrite(self, expr: Expr) -> Expr:
        D = self._alg.D
        f = expr.function
        if isinstance(f, (Integer, Rational)):
            return Integer(0)
        if isinstance(f, Sum):
            return Sum(*(D(c) for c in f.children))
        if isinstance(f, Neg):
            return Neg(D(f.arg))
        if isinstance(f, Product) and isinstance(
            f.children[0], (Integer, Rational)
        ):
            rest = f.children[1:]
            rest_expr = rest[0] if len(rest) == 1 else Product(*rest)
            return Product(f.children[0], D(rest_expr))
        head, rest = self._scalar_leibniz_split(f)
        return Sum(Product(head, D(rest)), Product(rest, D(head)))

    # -- theorem classification ------------------------------------- #

    def _bare_engine(self):
        """An engine with the defining rules only — crucially WITHOUT
        this linearity rule, so the derivation is non-circular."""
        from jacopy.proof.expansion import (
            ActOverSumOpDefinition,
            ExpansionEngine,
        )
        from jacopy.central.calculus import (
            HeadScalarDefinition,
            HeadSumDefinition,
            SlotNegDefinition,
            SlotZeroDefinition,
        )
        from jacopy.central.tangent.lie_bracket import (
            ScalarActAsMultiplicationDefinition,
        )

        return ExpansionEngine(
            [
                ActOverSumOpDefinition(),
                ScalarActAsMultiplicationDefinition(self._registry),
                HeadSumDefinition(),
                HeadScalarDefinition(self._registry),
                SlotNegDefinition(),
                SlotZeroDefinition(),
                AnchorLinearityDefinition(self._registry),
                CoboundaryPairingDefinition(self._alg),
            ]
        )

    def theorem_proof_builder(self):
        alg = self._alg
        registry = self._registry

        def _builder(matched: Expr) -> ProofChain:
            from jacopy.proof.strategies import ExpandAndSimplify

            after = self.rewrite(matched)
            taken = matched._repr_inner()
            name = next(
                n for n in ("s", "r", "q", "z", "w") if n not in taken
            )
            (s,) = alg.sections(name)
            sub = ExpandAndSimplify().prove(
                Pairing(matched, s),
                Pairing(after, s),
                registry=registry,
                engine=self._bare_engine(),
            )
            step = ProofStep(
                matched,
                after,
                rule="agreement on generators (pairing section)",
                justification=(
                    "both sides pair equally against the generic "
                    f"section {name}; E-1-forms agreeing on all "
                    "sections are equal"
                ),
            )
            for st in sub:
                step.add_child(st)
            return ProofChain([step])

        return _builder
