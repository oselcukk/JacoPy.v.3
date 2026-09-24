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

    #: The structure this rule was built for (Faz 8 step 2c): the
    #: algebroid of a declared axiom, the theorem's owner for a cited
    #: theorem, … ``None`` for structure-free rules. The engine refuses
    #: two rules whose owners share a tree-visible name but differ
    #: (:mod:`jacopy.proof.ownership`).
    owner = None

    #: What the rule IS, logically (Faz 8 step 3a): ``"definition"`` —
    #: a definitional unfolding or a structural law (no assumption);
    #: ``"assumption"`` — a DECLARED axiom of a structure (the
    #: ``*Declaration`` rules), i.e. a hypothesis the proof depends on;
    #: ``"theorem"`` — a cited proven result. ``ProofResult.from_chain``
    #: reads it from the fired steps; a step whose rule recorded no
    #: role is kept as a LEGACY assumption, never dropped.
    role = "definition"

    #: Explicitly-unverified citation (a TheoremBook record marked
    #: ``unverified``): its steps are classified as legacy assumptions.
    unverified = False

    #: Structural IDENTITY of the rule (Faz 8 step 1d contract):
    #: ``None`` by default — the rule is never deduplicated by identity
    #: and its :attr:`key` falls back to ``(class name, display name)``.
    #: A rule that gives one returns ``(class, structure identity,
    #: mathematical parameters)``: two rules with equal identity are the
    #: same rule (an explicit assumption instance, a declared axiom of
    #: a structure with its parameters …).
    identity = None

    @property
    def key(self):
        """The rule's structural key: :attr:`identity` when given, else
        ``(class name, display name)``. Recorded on every fired
        :class:`ProofStep` and on the resulting requirement records, so a
        citation can check that the SAME licensing rule is registered in
        the citing engine (Faz 8 step 3b, audit dc44f79 F2)."""
        ident = self.identity
        return ident if ident is not None else (type(self).__name__, self.name)

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


class ExplicitAssumption(Definition):
    """An EXPLICIT assumption instance: ``instance → target`` (default
    ``0``), declared by the caller for one concrete expression. Role
    ``assumption``; identity ``("ExplicitAssumption", owner name,
    instance, target)`` so that the same declaration in two engines is
    the same licensing rule — a theorem that used it is citable only
    where it is registered again. The seed of ``assume_zero`` (Faz 8
    step 5b): an exact-instance hypothesis, never a general identity."""

    role = "assumption"
    anchor = None

    def __init__(self, instance: Expr, target: Optional[Expr] = None, *, owner=None, name: Optional[str] = None) -> None:
        if not isinstance(instance, Expr):
            raise TypeError("ExplicitAssumption expects an Expr instance")
        from jacopy.core.expr import Integer as _Int

        self._instance = instance
        self._target = target if target is not None else _Int(0)
        if not isinstance(self._target, Expr):
            raise TypeError("ExplicitAssumption target must be an Expr")
        self.owner = owner
        self.anchor = type(instance)
        self.name = name or f"explicit assumption: {instance._repr_inner()} = {self._target._repr_inner()}"

    @property
    def instance(self) -> Expr:
        return self._instance

    @property
    def target(self) -> Expr:
        return self._target

    @property
    def identity(self):
        from jacopy.proof.ownership import owner_name

        return ("ExplicitAssumption", owner_name(self.owner) if self.owner is not None else None, self._instance, self._target)

    def matches(self, expr: Expr) -> bool:
        return expr == self._instance and expr != self._target

    def rewrite(self, expr: Expr) -> Expr:
        return self._target


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


def _step_role(d: Definition, tag: str) -> str:
    role = getattr(d, "role", None)
    if role == "assumption":
        return "assumption"
    if getattr(d, "unverified", False):
        return "unverified-citation"
    return "theorem" if tag == "theorem" else role


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
        "_owners",
        "_version",
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
        # tree-visible name -> owning structure of the registered rules
        # (Faz 8 step 2c; see jacopy.proof.ownership)
        self._owners: dict = {}
        # bumped on every registration: bound citations re-check their
        # verdict when it changes (audit dc44f79 F4)
        self._version = 0
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
        # A theorem citation is bound to THIS engine (a bound copy: the
        # caller's object stays reusable elsewhere) and checked against
        # the engine's structures; a blocked citation is registered
        # inert, reported, and claims no owner (Faz 8 step 3b).
        bind = getattr(definition, "bound_to", None)
        citable = True
        if bind is not None:
            definition = bind(self)
            citable = definition.citable
        owner = getattr(definition, "owner", None) if citable else None
        if owner is not None:
            from jacopy.proof.ownership import check_owners

            # refuses a second, different structure under a name already
            # owned in this engine (ambiguous: the tree cannot tell them
            # apart); a sub-structure's rule is covered by the owner
            # already present, a super-structure's rule STRENGTHENS the
            # engine and becomes the name's owner
            self._owners = check_owners([*self._owners.values(), definition])
        order = len(self._definitions)
        self._definitions.append(definition)
        self._version += 1
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
    def version(self) -> int:
        """Registration counter (changes whenever a rule is added)."""
        return self._version

    @property
    def owners(self) -> dict:
        """``{tree-visible name: structure}`` for the owned rules of
        this engine (Faz 8 step 2c)."""
        return dict(self._owners)

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
            owner=getattr(d, "owner", None),
            # an assumption stays an assumption even when theorem-tagged
            # (a theorem-classified consequence of a declared axiom); an
            # explicitly-unverified citation is marked as such
            role=_step_role(d, tag),
            cites=(getattr(d, "theorem", None) if tag == "theorem" else None),
            key=d.key,
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
