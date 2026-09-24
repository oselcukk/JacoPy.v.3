"""
The common normalization contract (Faz 8 step 4b).

Every proof in the package drives an expression to an engine
"normal form" with the same loop — definitional expansion, the
graded product rule, the algebraic simplifier, repeated — and fifteen
private copies of that loop had grown, each with its own round and
step defaults and its own idea of what happens when a budget runs
out (a silent return, a ``RuntimeError``, a ``ProofFailure``).
This module is the ONE loop; the old helpers are adapters that keep
their own defaults (documented at each) and delegate here.

Two layers:

* :func:`normalize` never raises for a budget: it returns a
  :class:`NormalForm` that says what it reached and WHY it stopped —
  ``"fixpoint"`` (one more round changes nothing), ``"rounds"`` (the
  round budget ran out while the expression was still changing),
  ``"steps"`` (an expansion pass ran out of rewrite steps), or
  ``"target"`` (the caller's early-stop predicate held). ``converged``
  means a FIXPOINT OF THIS PIPELINE — not a unique canonical form and
  not a proof that nothing else would reduce it.
* :func:`require_normal_form` is the raising layer: a result that is
  not a fixpoint (or the target) raises
  :class:`NormalizationBudgetExceeded`, which carries the partial
  :class:`NormalForm`.

Two loop shapes cover the package: the FLAT loop (each round is one
expansion pass, then the passes), and the NESTED loop
(``inner_rounds``: expansion + product rule to their own fixpoint,
then the simplifier, repeated) used by the strategies and the
algebroid provers. Steps are recorded on request (``record=True``) in
the engine's own LOCAL-step convention, so a recorded normalization
can extend a :class:`~jacopy.proof.chain.ProofChain` exactly as the
old loops did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from jacopy.core.expr import Expr
from jacopy.proof.step import ProofStep

STOP_REASONS = ("fixpoint", "rounds", "steps", "target")

Pass = Callable[[Expr, object], Expr]


def _product_rule_pass(expr: Expr, registry) -> Expr:
    from jacopy.algorithms.product_rule import product_rule

    return product_rule(expr, registry)


def _simplify_pass(expr: Expr, registry) -> Expr:
    from jacopy.algorithms.simplify import simplify

    return simplify(expr, registry)


#: The two standard passes by name.
PASSES = {"product_rule": _product_rule_pass, "simplify": _simplify_pass}
PASS_RULES = {"product_rule": ("product-rule", "graded Leibniz + linearity"), "simplify": ("simplify", "canonical-form pipeline")}


@dataclass(frozen=True)
class NormalForm:
    """What a normalization reached and why it stopped."""

    expr: Expr
    stopped_by: str
    rounds: int
    steps: int
    chain: Tuple[ProofStep, ...] = field(default=(), compare=False)
    detail: str = ""

    def __post_init__(self) -> None:
        if self.stopped_by not in STOP_REASONS:
            raise ValueError(f"stopped_by must be one of {STOP_REASONS}, got {self.stopped_by!r}")

    @property
    def converged(self) -> bool:
        """A fixpoint of the pipeline (or the caller's target) — NOT a
        claim of a unique canonical form."""
        return self.stopped_by in ("fixpoint", "target")

    @property
    def exhausted(self) -> bool:
        return self.stopped_by in ("rounds", "steps")

    def __str__(self) -> str:
        why = {"fixpoint": "fixpoint of the pipeline", "rounds": "round budget exhausted", "steps": "step budget exhausted", "target": "target reached"}[self.stopped_by]
        extra = f" ({self.detail})" if self.detail else ""
        return f"{self.expr._repr_inner()}  [{why}{extra}; {self.rounds} rounds, {self.steps} steps]"


class NormalizationBudgetExceeded(RuntimeError):
    """The raising layer's error: the normalization did not reach a
    fixpoint within its budget. ``normal_form`` is the partial
    result (the expression where it stopped, and why)."""

    def __init__(self, normal_form: NormalForm, what: str = "") -> None:
        self.normal_form = normal_form
        head = f"{what}: " if what else ""
        super().__init__(f"{head}normalization stopped by {normal_form.stopped_by} at {normal_form.expr._repr_inner()[:200]}")


def _expand(engine, expr: Expr, max_steps: int, record: bool):
    """One expansion pass keeping its partial steps: ``(expr, steps,
    exhausted)``. Unlike ``engine.expand`` it does not raise on the
    step budget — the caller decides."""
    steps: List[ProofStep] = []
    cur = expr
    n = 0
    for _ in range(max_steps):
        nxt, step = engine.expand_once(cur)
        if step is None:
            return cur, steps, n, False
        n += 1
        if record:
            steps.append(step)
        cur = nxt
    nxt, step = engine.expand_once(cur)
    if step is None:
        return cur, steps, n, False
    return cur, steps, n, True


def normalize(
    engine,
    expr: Expr,
    registry=None,
    *,
    rounds: int = 10,
    max_steps: int = 1024,
    passes: Sequence = ("product_rule", "simplify"),
    inner_rounds: Optional[int] = None,
    record: bool = False,
    stop_when: Optional[Callable[[Expr], bool]] = None,
) -> NormalForm:
    """Drive ``expr`` to a fixpoint of ``engine`` + ``passes``.

    ``rounds`` — outer rounds; ``max_steps`` — rewrite steps per
    expansion pass; ``passes`` — names from :data:`PASSES` or
    callables ``(expr, registry) -> expr`` applied after each expansion
    (flat loop) or after the inner fixpoint (nested loop);
    ``inner_rounds`` — when given, each outer round first iterates
    expansion + the FIRST pass to a fixpoint (at most ``inner_rounds``
    times) and then applies the remaining passes (the strategies'
    shape); ``record`` — keep the steps; ``stop_when`` — early stop
    (``stopped_by="target"``).

    Never raises for a budget; see :func:`require_normal_form`."""
    if not isinstance(expr, Expr):
        raise TypeError("normalize expects an Expr")
    if rounds < 1 or max_steps < 1 or (inner_rounds is not None and inner_rounds < 1):
        raise ValueError("rounds, max_steps and inner_rounds must be positive")
    fns = []
    for p in passes:
        if isinstance(p, str):
            if p not in PASSES:
                raise ValueError(f"unknown pass {p!r}; known: {tuple(PASSES)}")
            fns.append((p, PASSES[p]))
        elif callable(p):
            fns.append((getattr(p, "__name__", "pass"), p))
        else:
            raise TypeError("passes must be names or callables")

    chain: List[ProofStep] = []
    total_steps = 0
    cur = expr

    def apply(name, fn, before):
        after = fn(before, registry)
        if record and after != before:
            rule, why = PASS_RULES.get(name, (name, "normalization pass"))
            chain.append(ProofStep(before, after, rule=rule, justification=why))
        return after

    def done(stopped_by, used_rounds, detail=""):
        return NormalForm(cur, stopped_by, used_rounds, total_steps, tuple(chain), detail)

    for r in range(1, rounds + 1):
        if inner_rounds is None:
            expanded, steps, n, exhausted = _expand(engine, cur, max_steps, record)
            chain.extend(steps)
            total_steps += n
            if exhausted:
                cur = expanded
                return done("steps", r, f"expansion pass of round {r} used all {max_steps} steps")
            after = expanded
            for name, fn in fns:
                after = apply(name, fn, after)
        else:
            inner = cur
            settled = False
            for i in range(inner_rounds):
                expanded, steps, n, exhausted = _expand(engine, inner, max_steps, record)
                chain.extend(steps)
                total_steps += n
                if exhausted:
                    cur = expanded
                    return done("steps", r, f"expansion pass of round {r}.{i + 1} used all {max_steps} steps")
                after_first = apply(*fns[0], expanded) if fns else expanded
                if after_first == inner:
                    settled = True
                    break
                inner = after_first
            if not settled:
                cur = inner
                return done("rounds", r, f"inner loop of round {r} did not settle in {inner_rounds} rounds")
            after = inner
            for name, fn in fns[1:]:
                after = apply(name, fn, after)
        if stop_when is not None and stop_when(after):
            cur = after
            return done("target", r)
        if after == cur:
            return done("fixpoint", r)
        cur = after
    return done("rounds", rounds, f"still changing after {rounds} rounds")


def require_normal_form(engine, expr: Expr, registry=None, *, what: str = "", **kw) -> NormalForm:
    """:func:`normalize`, raising :class:`NormalizationBudgetExceeded`
    unless the result converged (a fixpoint, or the target)."""
    nf = normalize(engine, expr, registry, **kw)
    if not nf.converged:
        raise NormalizationBudgetExceeded(nf, what)
    return nf


__all__ = [
    "PASSES",
    "STOP_REASONS",
    "NormalForm",
    "NormalizationBudgetExceeded",
    "normalize",
    "require_normal_form",
]
