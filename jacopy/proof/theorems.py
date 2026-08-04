"""
Theorem Book — registry of proven results + the derived-rule bridge
(Engine pass E.3; ported from v2 and extended for the Phase 2
definition policy).

A :class:`Theorem` stores, besides the human-readable record, the
**equation itself** (``lhs``/``rhs`` Exprs) and a **generality tag**
declaring at which level the proof closed (an operator-level identity
holds for every degree; a ``p=2`` unroll only witnesses that instance).

The bridge to the engine is :class:`TheoremDefinition`: a *derived*
:class:`~jacopy.proof.expansion.Definition` built from a proven
theorem. It rewrites ``lhs → rhs``, carries the stored proof as its
``theorem_proof_builder`` (so ``mode="foundational"`` inlines the full
derivation under the step), and is tagged ``"theorem"`` in the
resulting :class:`ProofStep`. Per the definition policy, such rules are
NEVER part of the always-on definitional engine — they enter a proof
only through explicit citation::

    eng = tangent_engine(registry=reg)
    cite(eng, book, "cartan_magic", "d_squared_zero")
    ExpandAndSimplify().prove(lhs, rhs, engine=eng, ...)

Matching is structural in v1: the rule fires on nodes structurally
equal to ``lhs`` (or, when ``both_directions`` is set, to ``rhs``,
rewriting the other way). Wildcard/pattern lhs matching is a planned
extension once the first genuinely schematic theorem needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

from jacopy.core.expr import Expr, Integer, Sum
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine


def _sum_multiset_contains(
    haystack: Tuple[Expr, ...], needles: Tuple[Expr, ...]
) -> bool:
    """Is every term of ``needles`` present in ``haystack`` (with
    multiplicity)?"""
    pool = list(haystack)
    for n in needles:
        try:
            pool.remove(n)
        except ValueError:
            return False
    return True


#: Recognized proof-generality tags (definition policy §3).
GENERALITY_TAGS = (
    "operator-level",   # holds for all degrees, proved on operators
    "generic-function", # proved via action on a generic scalar function
    "symbolic-p",       # proved with symbolic degree (IndexedSum route)
    "instance",         # proved for a concrete instance (e.g. p=2)
)


@dataclass(frozen=True)
class Theorem:
    """A named, proven equation.

    Parameters
    ----------
    name
        Unique short handle (``"cartan_magic"``), the Book lookup key.
    statement
        Human-readable statement, e.g. ``"L_X = d ι_X + ι_X d"``.
    lhs, rhs
        The two sides of the proven equation, as Exprs. ``lhs`` is the
        side a citation rewrites *from* by default.
    proof
        The :class:`ProofChain` establishing ``lhs = rhs``.
    generality
        One of :data:`GENERALITY_TAGS` — at which level the proof
        closed.
    from_axioms
        Descriptive tuple of the definitional-rule names the proof
        relies on.
    notes
        Optional exposition.
    """

    name: str
    statement: str
    lhs: Expr
    rhs: Expr
    proof: ProofChain
    generality: str = "generic-function"
    from_axioms: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Theorem.name must be a non-empty string")
        if not isinstance(self.statement, str):
            raise TypeError("Theorem.statement must be a string")
        if not isinstance(self.lhs, Expr) or not isinstance(self.rhs, Expr):
            raise TypeError("Theorem.lhs / Theorem.rhs must be Exprs")
        if not isinstance(self.proof, ProofChain):
            raise TypeError("Theorem.proof must be a ProofChain")
        if self.generality not in GENERALITY_TAGS:
            raise ValueError(
                f"Theorem.generality must be one of {GENERALITY_TAGS}, "
                f"got {self.generality!r}"
            )
        if not isinstance(self.from_axioms, tuple) or not all(
            isinstance(a, str) for a in self.from_axioms
        ):
            raise TypeError("Theorem.from_axioms must be a tuple of strings")
        if not isinstance(self.notes, str):
            raise TypeError("Theorem.notes must be a string")


class TheoremBook:
    """Ordered registry of :class:`Theorem` records keyed by name.

    Insertion order is preserved (a rendered "table of results" reads
    in registration order). Duplicate names raise on :meth:`add`; use
    :meth:`replace` for a deliberate override.
    """

    __slots__ = ("_theorems",)

    def __init__(self) -> None:
        self._theorems: dict = {}

    def add(self, theorem: Theorem) -> None:
        """Register ``theorem``; raise :class:`KeyError` on name clash."""
        if not isinstance(theorem, Theorem):
            raise TypeError("TheoremBook.add expects a Theorem")
        if theorem.name in self._theorems:
            raise KeyError(
                f"Theorem {theorem.name!r} is already registered; "
                f"use replace() to override"
            )
        self._theorems[theorem.name] = theorem

    def replace(self, theorem: Theorem) -> None:
        """Register ``theorem``, overwriting an existing same-name entry."""
        if not isinstance(theorem, Theorem):
            raise TypeError("TheoremBook.replace expects a Theorem")
        self._theorems[theorem.name] = theorem

    def get(self, name: str) -> Theorem:
        if name not in self._theorems:
            raise KeyError(f"no theorem named {name!r}; known: {self.names()}")
        return self._theorems[name]

    def names(self) -> Tuple[str, ...]:
        return tuple(self._theorems)

    def remove(self, name: str) -> None:
        if name not in self._theorems:
            raise KeyError(f"no theorem named {name!r}")
        del self._theorems[name]

    def clear(self) -> None:
        self._theorems.clear()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._theorems

    def __len__(self) -> int:
        return len(self._theorems)

    def __iter__(self) -> Iterator[Theorem]:
        return iter(self._theorems.values())

    def __repr__(self) -> str:
        return f"TheoremBook({len(self)} theorems: {', '.join(self.names())})"


# --------------------------------------------------------------------- #
# Derived-rule bridge                                                    #
# --------------------------------------------------------------------- #


class TheoremDefinition(Definition):
    """A derived rewrite rule backed by a proven :class:`Theorem`.

    Rewrites nodes structurally equal to ``theorem.lhs`` into
    ``theorem.rhs``. With ``reverse=True`` the orientation flips
    (``rhs → lhs``) — some citations close a proof only in the other
    direction. ``theorem_proof_builder`` returns the stored chain, so
    the engine tags the step ``"theorem"`` and, in foundational mode,
    inlines the derivation.
    """

    def __init__(self, theorem: Theorem, *, reverse: bool = False) -> None:
        if not isinstance(theorem, Theorem):
            raise TypeError("TheoremDefinition expects a Theorem")
        self._theorem = theorem
        self._reverse = bool(reverse)
        direction = "⇐" if self._reverse else "⇒"
        self.name = f"theorem {theorem.name} {direction}: {theorem.statement}"
        src = theorem.rhs if self._reverse else theorem.lhs
        self.anchor = type(src)

    @property
    def theorem(self) -> Theorem:
        return self._theorem

    def _source(self) -> Expr:
        return self._theorem.rhs if self._reverse else self._theorem.lhs

    def _target(self) -> Expr:
        return self._theorem.lhs if self._reverse else self._theorem.rhs

    def matches(self, expr: Expr) -> bool:
        # An identity rewrite (lhs == rhs — degenerate instances can
        # normalize to 0 = 0) would fire forever; stay inert instead
        # (caught by the Phase 5 edge-case audit).
        if self._source() == self._target():
            return False
        if expr == self._source():
            return True
        return self._matches_sum_subset(expr)

    @staticmethod
    def _term_count(expr: Expr) -> int:
        if expr == Integer(0):
            return 0
        if isinstance(expr, Sum):
            return len(expr.children)
        return 1

    def _matches_sum_subset(self, expr: Expr) -> bool:
        # Sum-subset citation (Phase 5.E.2b): a proven Sum instance
        # covers a subset of a larger Sum's terms —
        # Σ(A ∪ B) = Σ(A) + Σ(B) and Σ(A) = rhs replaces A by rhs.
        # Admitted only when the rhs has STRICTLY FEWER terms than
        # the covered subset (rhs = 0, or e.g. a 2→1 bracket
        # rearrangement), so the total term count strictly decreases
        # and the rewrite terminates.
        src = self._source()
        return (
            isinstance(src, Sum)
            and isinstance(expr, Sum)
            and self._term_count(self._target())
            < len(src.children)
            and len(expr.children) > len(src.children)
            and _sum_multiset_contains(expr.children, src.children)
        )

    def rewrite(self, expr: Expr) -> Expr:
        if expr == self._source():
            return self._target()
        remaining = list(expr.children)
        for t in self._source().children:
            remaining.remove(t)
        target = self._target()
        if target == Integer(0):
            return Sum.make(*remaining)
        extra = (
            list(target.children)
            if isinstance(target, Sum)
            else [target]
        )
        return Sum.make(*remaining, *extra)

    def theorem_proof_builder(self):
        chain = self._theorem.proof

        def _builder(_matched: Expr) -> ProofChain:
            return chain

        return _builder


def cite(
    engine: ExpansionEngine,
    book: TheoremBook,
    *names: str,
    reverse: bool = False,
) -> ExpansionEngine:
    """Register the named theorems on ``engine`` as derived rules.

    Mutates and returns ``engine`` for chaining. Citation is the ONLY
    sanctioned way a proven result enters a proof (definition policy
    §2) — derived rules are never part of the always-on engines.
    """
    for n in names:
        engine.register(TheoremDefinition(book.get(n), reverse=reverse))
    return engine


#: Process-wide default book. Central-code modules register their
#: proven theorems here; proofs cite them by name.
theorem_book = TheoremBook()
