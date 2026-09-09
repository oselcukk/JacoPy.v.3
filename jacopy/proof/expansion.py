"""
Definition-based expansion.

A :class:`Definition` is a local rewrite rule keyed to a predicate on
an expression subtree: *if this shape appears, replace it with this*.
The :class:`ExpansionEngine` walks an expression bottom-up, fires
registered definitions at the first matching site, and records each
rewrite as a :class:`ProofStep` so the caller can reconstruct the
sequence of unfolds.

The engine is deliberately narrow: it only expands definitions. Koszul
signs, Leibniz distribution, and collecting like terms are *not* its
job, those live in :mod:`jacopy.algorithms` and the strategies layer
runs them afterward. Keeping the two concerns separate matches the
plan's "expand definitions first, then simplify" proof shape, and it
means a strategy can choose when to interleave or defer each pass.

Built-in definitions live in this module. More can be plugged in via
:meth:`ExpansionEngine.register`. Future phases will add
:class:`DerivedBracket` and operator-equation definitions here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, compose, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.step import ProofStep



# --------------------------------------------------------------------- #
# Definition base class                                                  #
# --------------------------------------------------------------------- #


class Definition(ABC):
    """A single rewrite rule keyed to a local pattern.

    Subclasses provide :meth:`matches` (does ``expr`` fit the left-hand
    side?) and :meth:`rewrite` (produce the right-hand side). The
    :attr:`name` is used verbatim in the :class:`ProofStep` rule field.

    A definition is classified as an **axiom** by default, the engine
    tags its :class:`ProofStep` with ``provenance_tag="axiom"``. A
    subclass that represents a *theorem* overrides
    :meth:`theorem_proof_builder` to return a callable producing the
    sub-proof; in foundational mode the engine then attaches that
    sub-proof under the step as children. Efficient mode still tags the
    step ``"theorem"`` but skips sub-proof construction, so the same
    definition serves both proof modes.
    """

    name: str = "definition"

    #: Optional dispatch anchor: the Expr class (or tuple of classes)
    #: this definition's LHS is rooted at. When set, the engine only
    #: consults the rule on nodes of that type (via an MRO-indexed
    #: lookup) instead of trying every rule at every node. ``None``
    #: keeps the rule in the always-consulted fallback list.
    anchor = None

    @abstractmethod
    def matches(self, expr: Expr) -> bool:
        """True when ``expr`` is an instance of this definition's LHS."""

    @abstractmethod
    def rewrite(self, expr: Expr) -> Expr:
        """Return the RHS. Caller guarantees ``matches(expr)`` is True."""

    def theorem_proof_builder(self):
        """Return a callable producing this definition's sub-proof, or ``None``.

        The returned callable has signature
        ``(matched_expr: Expr) -> ProofChain`` and is only invoked by the
        engine when ``mode="foundational"``. ``None`` (the default)
        marks the definition as a pure axiom.
        """
        return None

    @property
    def is_theorem(self) -> bool:
        """True when this definition carries a sub-proof builder."""
        return self.theorem_proof_builder() is not None


# --------------------------------------------------------------------- #
# Built-in definitions                                                   #
# --------------------------------------------------------------------- #


def _is_degree_zero(expr: Expr, registry: Optional[PropertyRegistry]) -> bool:
    """Safe degree-zero check: returns False on any undecidable case."""
    try:
        return degree_of(expr, registry) == Degree.const(0)
    except ValueError:
        return False


class ActOverSumOpDefinition(Definition):
    """``Act(A + B, x) → Act(A, x) + Act(B, x)``, linearity in the operator.

    ``product_rule`` distributes an operator across a sum in its
    operand (``D(a+b) → D(a) + D(b)``) but not across a sum in its
    operator position. The agreement-on-generators strategy routinely
    builds :class:`Act` nodes whose operator is a :class:`Sum` of
    compositions (the Cartan magic formula RHS is the canonical
    example), so this rewrite is what lets such proofs close via a
    structural cancellation.
    """

    name = "Act linearity: (A + B)(x) = A(x) + B(x)"
    anchor = Act

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Act) and isinstance(expr.op, Sum)

    def rewrite(self, expr: Expr) -> Expr:
        ops = expr.op.children
        return Sum(*(Act(op, expr.arg) for op in ops))


#: The two classifications a :class:`DSquaredZeroDefinition` can carry.
D_SQUARED_CLASSIFICATIONS = ("axiom", "theorem")


















# --------------------------------------------------------------------- #
# Engine                                                                 #
# --------------------------------------------------------------------- #


#: The two proof modes recognised by :class:`ExpansionEngine`.
MODES = ("efficient", "foundational")


class ExpansionEngine:
    """Apply registered :class:`Definition` rules bottom-up to fix-point.

    On each iteration :meth:`expand_once` finds the leftmost-innermost
    matching site and rewrites it, recording a :class:`ProofStep`.
    :meth:`expand` loops until no definition fires. A ``max_steps``
    guard protects against a misbehaving rule that produces an
    ever-growing expression.

    The engine's :attr:`mode` controls how theorem-classified
    definitions are recorded:

    * ``"efficient"``, theorems fire like axioms; the resulting step
      is tagged ``"theorem"`` but carries no sub-proof, matching the
      "property taken as given" mode of the plan.
    * ``"foundational"``, theorems fire and their
      :meth:`Definition.theorem_proof_builder` is invoked; the
      resulting sub-proof is attached under the step as children,
      exposing the derivation down to axioms.
    """

    __slots__ = (
        "_definitions",
        "_mode",
        "_index",
        "_unanchored",
        "assembly_report",
    )

    def __init__(
        self,
        definitions: Optional[List[Definition]] = None,
        *,
        mode: str = "efficient",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._definitions: List[Definition] = []
        self._mode = mode
        # Filled by jacopy.research.engine_assembly.assemble_engine:
        # one line per detected family / added rule. Lives ON the
        # engine so no side registry keeps discarded engines alive
        # (2026-09-09 audit, finding F4).
        self.assembly_report: List[str] = []
        # Dispatch index: Expr class -> [(registration order, rule)].
        # Rules without an anchor stay in the always-consulted fallback
        # list. Registration order is preserved across both so the
        # first-registered-wins semantics is unchanged.
        self._index: dict = {}
        self._unanchored: List[tuple] = []
        for d in definitions or []:
            self.register(d)

    def register(self, definition: Definition) -> None:
        """Append ``definition`` to the rule list (and dispatch index)."""
        if not isinstance(definition, Definition):
            raise TypeError("ExpansionEngine.register expects a Definition")
        order = len(self._definitions)
        self._definitions.append(definition)
        anchor = definition.anchor
        if anchor is None:
            self._unanchored.append((order, definition))
            return
        anchors = anchor if isinstance(anchor, tuple) else (anchor,)
        for cls in anchors:
            self._index.setdefault(cls, []).append((order, definition))

    @property
    def definitions(self) -> Tuple[Definition, ...]:
        return tuple(self._definitions)

    @property
    def mode(self) -> str:
        return self._mode

    def with_mode(self, mode: str) -> "ExpansionEngine":
        """Return a new engine with the same definitions but a different mode."""
        return ExpansionEngine(list(self._definitions), mode=mode)

    # -- single-step ----------------------------------------------------- #

    def _match(self, expr: Expr) -> Optional[Definition]:
        # Gather anchored candidates via the expression's MRO so a rule
        # anchored at a base class also fires on subclasses, then merge
        # with the unanchored fallback list in registration order.
        candidates = list(self._unanchored)
        for cls in type(expr).__mro__:
            candidates.extend(self._index.get(cls, ()))
        candidates.sort(key=lambda pair: pair[0])
        for _, d in candidates:
            if d.matches(expr):
                return d
        return None

    def _build_step(self, expr: Expr, after: Expr, d: Definition) -> ProofStep:
        """Create the :class:`ProofStep` for a fired rewrite, with mode-dependent children."""
        tag = "theorem" if d.is_theorem else "axiom"
        step = ProofStep(
            expr,
            after,
            rule=d.name,
            justification=f"apply {tag}: {d.name}",
            provenance_tag=tag,
        )
        if self._mode == "foundational" and d.is_theorem:
            builder = d.theorem_proof_builder()
            if builder is not None:
                sub_chain = builder(expr)
                for sub_step in sub_chain:
                    step.add_child(sub_step)
        return step

    def expand_once(self, expr: Expr) -> Tuple[Expr, Optional[ProofStep]]:
        """Apply a single rewrite at the leftmost-innermost matching site.

        Children are visited first, so the rule that fires is the one
        closest to the leaves. Returns the (possibly unchanged)
        expression and the :class:`ProofStep` produced by the fired
        rule, or ``None`` when no definition matches anywhere.

        **Slot protocol**: operator-like atoms that carry expressions
        in private slots (an algebroid bracket's sections, a mapped
        section's argument, …) may expose them via a
        ``rewritable_slots`` tuple plus a ``with_slots(*subs)``
        rebuilder. The engine recurses into those slots exactly like
        children, so rules fire *inside* opaque atoms — the structural
        fix for the slot-opacity limitation v2 worked around with
        targeted pre-passes.
        """
        if not expr.is_atom:
            children = list(expr.children)
            for i, c in enumerate(children):
                new_c, step = self.expand_once(c)
                if step is not None:
                    children[i] = new_c
                    return expr._rebuild(tuple(children)), step
        else:
            slots = getattr(expr, "rewritable_slots", None)
            if slots:
                slots = tuple(slots)
                for i, sub in enumerate(slots):
                    new_sub, step = self.expand_once(sub)
                    if step is not None:
                        new_slots = slots[:i] + (new_sub,) + slots[i + 1:]
                        return expr.with_slots(*new_slots), step
        d = self._match(expr)
        if d is not None:
            after = d.rewrite(expr)
            return after, self._build_step(expr, after, d)
        return expr, None

    # -- driver ---------------------------------------------------------- #

    def expand(
        self, expr: Expr, *, max_steps: int = 1024
    ) -> Tuple[Expr, List[ProofStep]]:
        """Apply definitions repeatedly until no rule fires.

        Returns the fully expanded expression and the list of steps
        taken. Raises :class:`RuntimeError` if the fix-point isn't
        reached within ``max_steps`` iterations, that would indicate a
        cyclic or divergent rule set. (Raised 256 → 1024 in the
        higher-degree audit: a ``d²`` Palais double-expansion of a
        5-form on 7 slots legitimately exceeds 256 rewrites; a true
        cycle still trips the bound, just later.)"""
        steps: List[ProofStep] = []
        current = expr
        for _ in range(max_steps):
            nxt, step = self.expand_once(current)
            if step is None:
                return current, steps
            steps.append(step)
            current = nxt
        raise RuntimeError(
            f"ExpansionEngine did not converge within {max_steps} steps; "
            "check for cyclic definitions"
        )




def default_engine(
    *,
    registry: Optional[PropertyRegistry] = None,
    mode: str = "efficient",
) -> ExpansionEngine:
    """Minimal, calculus-independent default engine.

    Contains only the degree-independent structural rule
    ``ActOverSumOpDefinition`` (``Act(op, Sum(...)) -> Sum(Act(op, ...))``).
    In the v3 architecture the Cartan-calculus rules (``d² = 0``,
    ``ι² = 0``, Cartan magic, ``ι(df) = X(f)`` …) are supplied by the
    **central code** (Phase 2, ``jacopy.central.tangent``); each
    package adds its own rule set via ``ExpansionEngine.register``.
    That way the proof engine stays independent of any geometry
    package.

    ``registry`` is kept for signature compatibility for now (it will
    be forwarded to registry-aware rules when the calculus rules
    arrive).
    """
    return ExpansionEngine([ActOverSumOpDefinition()], mode=mode)
