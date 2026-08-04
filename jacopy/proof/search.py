"""
Proof search over the rewrite state graph (Engine pass E.5).

The deterministic :class:`~jacopy.proof.strategies.ExpandAndSimplify`
commits to one rule order (leftmost-innermost, first match).
:class:`SearchStrategy` instead explores the graph whose **nodes are
expressions** (memoized by structural equality) and whose **edges are
single moves** — one definition firing at one position, one
``product_rule`` sweep, or one ``simplify`` sweep. Different rule
orders that reconverge to the same expression collapse into one node,
which is what keeps the search tractable where the *path* space would
explode combinatorially.

Two engines under one interface:

* **Bounded exhaustive BFS** — finds *every* proof up to
  ``max_depth`` within the state budget, scores them with the chosen
  :data:`SCORERS` entry, returns the best as an ordinary
  :class:`ProofChain` (deterministic replay, display-ready).
* **Beam fallback** — when the exhaustive budget is exceeded with no
  proof found, a greedy beam (width ``beam_width``, node-count
  heuristic) takes over.

The user-facing knob is ``prefer``:

* ``"shortest"`` — fewest steps;
* ``"readable"`` — smallest peak intermediate expression (proofs that
  never blow up mid-way);
* ``"elementary"`` — fewest theorem citations (most definitional),
  step count as tie-break.

Soundness is untouched by any of this: search only chooses the *order*
in which sound rules fire, and the returned object is a plain
:class:`ProofChain`.

A later e-graph / equality-saturation backend is planned behind this
same interface (ROADMAP E.6) — callers depend on ``prefer``, not on
the engine that satisfied it.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

from jacopy.algorithms.product_rule import product_rule
from jacopy.algorithms.simplify import simplify
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine, default_engine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure, Strategy


def _node_count(expr: Expr) -> int:
    return sum(1 for _ in expr.walk())


# --------------------------------------------------------------------- #
# Scorers — lower is better                                              #
# --------------------------------------------------------------------- #


def _score_shortest(steps: List[ProofStep]) -> Tuple:
    return (len(steps),)


def _score_readable(steps: List[ProofStep]) -> Tuple:
    peak = max(
        (max(_node_count(s.before), _node_count(s.after)) for s in steps),
        default=0,
    )
    return (peak, len(steps))


def _score_elementary(steps: List[ProofStep]) -> Tuple:
    theorem_steps = sum(
        1 for s in steps if getattr(s, "provenance_tag", None) == "theorem"
    )
    return (theorem_steps, len(steps))


SCORERS: Dict[str, Callable[[List[ProofStep]], Tuple]] = {
    "shortest": _score_shortest,
    "readable": _score_readable,
    "elementary": _score_elementary,
}


# --------------------------------------------------------------------- #
# Move generation                                                        #
# --------------------------------------------------------------------- #


def _rewrite_at(root: Expr, path: Tuple[int, ...], new_sub: Expr) -> Expr:
    """Rebuild ``root`` with the subexpression at ``path`` replaced."""
    if not path:
        return new_sub
    i = path[0]
    children = list(root.children)
    children[i] = _rewrite_at(children[i], path[1:], new_sub)
    return root._rebuild(tuple(children))


def _definition_moves(
    engine: ExpansionEngine, root: Expr
) -> List[Tuple[Expr, ProofStep]]:
    """Every single-position definition firing available on ``root``.

    Unlike :meth:`ExpansionEngine.expand_once` (leftmost-innermost,
    first match only) this enumerates *all* (rule, position) pairs —
    that freedom of order is the whole point of searching.
    """
    moves: List[Tuple[Expr, ProofStep]] = []

    def visit(sub: Expr, path: Tuple[int, ...]) -> None:
        if not sub.is_atom:
            for i, c in enumerate(sub.children):
                visit(c, path + (i,))
        d = engine._match(sub)
        if d is not None:
            after_sub = d.rewrite(sub)
            after_root = _rewrite_at(root, path, after_sub)
            tag = "theorem" if d.is_theorem else "axiom"
            step = ProofStep(
                root,
                after_root,
                rule=d.name,
                justification=f"apply {tag}: {d.name}",
                provenance_tag=tag,
            )
            moves.append((after_root, step))

    visit(root, ())
    return moves


def _sweep_moves(
    root: Expr, registry: Optional[PropertyRegistry]
) -> List[Tuple[Expr, ProofStep]]:
    """The two composite moves: one Leibniz sweep, one simplify sweep."""
    moves: List[Tuple[Expr, ProofStep]] = []
    pr = product_rule(root, registry)
    if pr != root:
        moves.append(
            (
                pr,
                ProofStep(
                    root,
                    pr,
                    rule="product-rule",
                    justification="graded Leibniz + linearity",
                ),
            )
        )
    sp = simplify(root, registry)
    if sp != root:
        moves.append(
            (
                sp,
                ProofStep(
                    root,
                    sp,
                    rule="simplify",
                    justification="canonical-form pipeline",
                ),
            )
        )
    return moves


# --------------------------------------------------------------------- #
# The strategy                                                           #
# --------------------------------------------------------------------- #


class SearchStrategy(Strategy):
    """Find a proof by graph search and return the best one found.

    Parameters
    ----------
    prefer
        Scorer name from :data:`SCORERS` (default ``"shortest"``).
    max_depth
        Maximum number of moves in a proof (BFS horizon).
    max_states
        Budget of distinct expressions to explore before falling back
        to beam mode.
    beam_width
        Frontier width of the beam fallback.
    max_solutions
        Stop collecting proofs after this many are found (they are
        scored and the best is returned).
    """

    name = "Search"

    def __init__(
        self,
        *,
        prefer: str = "shortest",
        max_depth: int = 12,
        max_states: int = 5000,
        beam_width: int = 32,
        max_solutions: int = 16,
    ) -> None:
        if prefer not in SCORERS:
            raise ValueError(
                f"prefer must be one of {tuple(SCORERS)}, got {prefer!r}"
            )
        self._prefer = prefer
        self._max_depth = max_depth
        self._max_states = max_states
        self._beam_width = beam_width
        self._max_solutions = max_solutions
        #: Observation hook: statistics of the most recent prove() call
        #: (states explored, proofs found, which engine closed, the
        #: candidate proof lengths). Purely informational.
        self.last_stats: Dict[str, object] = {}
        #: Observation hook: every proof found by the most recent
        #: prove() call, as lists of ProofSteps (BFS solutions in
        #: discovery order; in beam mode the single found proof).
        self.last_solutions: List[List[ProofStep]] = []

    # ---- helpers ---------------------------------------------------- #

    def _moves(
        self,
        engine: ExpansionEngine,
        state: Expr,
        registry: Optional[PropertyRegistry],
    ) -> List[Tuple[Expr, ProofStep]]:
        return _definition_moves(engine, state) + _sweep_moves(state, registry)

    @staticmethod
    def _reconstruct(
        parents: Dict[Expr, Tuple[Optional[Expr], Optional[ProofStep]]],
        final_state: Expr,
        final_step: Optional[ProofStep],
    ) -> List[ProofStep]:
        steps: List[ProofStep] = [] if final_step is None else [final_step]
        cur = final_state
        while True:
            parent, step = parents[cur]
            if parent is None:
                break
            steps.append(step)
            cur = parent
        steps.reverse()
        return steps

    # ---- driver ----------------------------------------------------- #

    def prove(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> ProofChain:
        eng = engine if engine is not None else default_engine()
        goal = Integer(0)
        start: Expr = Sum(lhs, Neg(rhs))
        if lhs == rhs:
            chain = ProofChain()
            chain.append(
                ProofStep(
                    lhs,
                    rhs,
                    rule="reflexive",
                    justification="lhs and rhs are syntactically identical",
                )
            )
            return chain

        scorer = SCORERS[self._prefer]
        parents: Dict[Expr, Tuple[Optional[Expr], Optional[ProofStep]]] = {
            start: (None, None)
        }
        depth_of: Dict[Expr, int] = {start: 0}
        solutions: List[List[ProofStep]] = []
        frontier = deque([start])
        exhausted = False

        while frontier:
            state = frontier.popleft()
            depth = depth_of[state]
            if depth >= self._max_depth:
                continue
            for nxt, step in self._moves(eng, state, registry):
                if nxt == goal:
                    solutions.append(
                        self._reconstruct(parents, state, step)
                    )
                    if len(solutions) >= self._max_solutions:
                        frontier.clear()
                        break
                    continue
                if nxt in parents:
                    continue
                parents[nxt] = (state, step)
                depth_of[nxt] = depth + 1
                frontier.append(nxt)
                if len(parents) > self._max_states:
                    exhausted = True
                    frontier.clear()
                    break
            if exhausted:
                break

        self.last_stats = {
            "prefer": self._prefer,
            "states_explored": len(parents),
            "solutions_found": len(solutions),
            "solution_lengths": sorted(len(s) for s in solutions),
            "engine": "bfs",
            "budget_exhausted": exhausted,
        }
        self.last_solutions = list(solutions)

        if solutions:
            best = min(solutions, key=scorer)
            self.last_stats["chosen_length"] = len(best)
            chain = ProofChain()
            chain.extend(best)
            return chain

        if exhausted:
            beam_chain = self._beam(eng, start, goal, registry)
            if beam_chain is not None:
                self.last_stats["engine"] = "beam"
                self.last_stats["chosen_length"] = len(beam_chain.steps)
                self.last_solutions = [list(beam_chain.steps)]
                return beam_chain

        raise ProofFailure(
            f"SearchStrategy found no proof of "
            f"{lhs._repr_inner()} == {rhs._repr_inner()} within "
            f"depth {self._max_depth} / {self._max_states} states"
            + (" (beam fallback also failed)" if exhausted else "")
        )

    # ---- beam fallback ---------------------------------------------- #

    def _beam(
        self,
        engine: ExpansionEngine,
        start: Expr,
        goal: Expr,
        registry: Optional[PropertyRegistry],
    ) -> Optional[ProofChain]:
        """Greedy beam search: keep the ``beam_width`` smallest states."""
        frontier: List[Tuple[Expr, List[ProofStep]]] = [(start, [])]
        seen = {start}
        for _ in range(self._max_depth * 2):
            candidates: List[Tuple[Expr, List[ProofStep]]] = []
            for state, steps in frontier:
                for nxt, step in self._moves(engine, state, registry):
                    if nxt == goal:
                        chain = ProofChain()
                        chain.extend(steps + [step])
                        return chain
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    candidates.append((nxt, steps + [step]))
            if not candidates:
                return None
            candidates.sort(key=lambda pair: _node_count(pair[0]))
            frontier = candidates[: self._beam_width]
        return None


# --------------------------------------------------------------------- #
# User-facing convenience                                                #
# --------------------------------------------------------------------- #


def prove(
    lhs: Expr,
    rhs: Expr,
    *,
    prefer: Optional[str] = None,
    registry: Optional[PropertyRegistry] = None,
    engine: Optional[ExpansionEngine] = None,
    **search_options,
) -> ProofChain:
    """Prove ``lhs == rhs``, optionally choosing among found proofs.

    ``prefer=None`` (default) uses the deterministic
    :class:`ExpandAndSimplify` — fast, single path. Passing
    ``prefer="shortest" | "readable" | "elementary"`` switches to
    :class:`SearchStrategy`, which explores alternative rule orders
    and returns the best proof under the chosen scorer. Extra keyword
    options are forwarded to :class:`SearchStrategy`.
    """
    if prefer is None:
        return ExpandAndSimplify().prove(
            lhs, rhs, registry=registry, engine=engine
        )
    return SearchStrategy(prefer=prefer, **search_options).prove(
        lhs, rhs, registry=registry, engine=engine
    )
