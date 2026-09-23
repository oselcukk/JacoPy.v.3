"""
The result / assumption / provenance model (Faz 8 step 3a).

A proof attempt answers three different questions, and this module
keeps the answers apart:

* WHAT was attempted — :class:`Goal` (``lhs = rhs``) and the
  :class:`ProofResult.status`: ``CLOSED`` (the two sides met),
  ``RESIDUAL`` (the engine reached a fix-point with a surviving
  difference — NOT a disproof, see below), ``BUDGET`` (the step budget
  ran out before a fix-point) or ``INVALID`` (the goal is ill-typed:
  a 1-form equated to a 2-form, a vector to a function …; nothing was
  run). There is no ``DISPROVED``: an unreduced residual is never a
  certificate of non-zero (audit 2026-09-23, 0058df0).
* WHAT IT DEPENDS ON — ``requires``: the :class:`Assumption` records.
  ``declared`` — a declared axiom of a structure that actually fired
  (with its owner and the number of instances); ``structure`` — the
  structure objects those axioms belong to; ``scope`` — the proof's
  generality when it is an instance proof (an instance certificate is
  never upgraded to a general one); ``legacy`` — an assumption that is
  RECORDED but not structurally verified: a step whose rule carries no
  role, an unverified citation. ``textual`` — a ``from_axioms`` prose
  entry no step accounts for: kept visible, NOT a verification input
  (prose is not a trust boundary; the structural record is). Also a
  theorem record whose chain does not literally run ``lhs → rhs``.
  Legacy entries are kept, never dropped (step 3b decides what may
  cite them).
* HOW it was derived — ``provenance``: cited theorems, definitional
  unfoldings, canonicalizations, plain engine steps, computations.
  These are not assumptions: a definition assumes nothing.

``CLOSED`` is always shown "under" its assumptions (:meth:`summary`).
The result carries the build version from the package metadata (no
``.git`` needed) and the step :class:`Budget`.

The old ``(chain, theorem)`` returns of the provers are untouched:
:meth:`ProofResult.from_theorem` and :meth:`ProofResult.from_chain`
wrap them, and :func:`prove` is the small driver that produces a
result directly from an engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from jacopy.core.expr import Expr, Integer
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep

STATUSES = ("CLOSED", "RESIDUAL", "BUDGET", "INVALID")
ASSUMPTION_KINDS = ("declared", "structure", "scope", "legacy", "textual")
PROVENANCE_KINDS = ("theorem", "definition", "canonicalization", "engine-step", "computation")

_CANONICAL_RULES = ("simplify", "canonicalize", "normalize", "collect", "sort", "flatten", "distribute")


def package_version() -> str:
    """The installed package version from the build metadata (the
    ``jacopy.__version__`` fallback when not installed)."""
    try:
        from importlib.metadata import version

        return version("jacopy")
    except Exception:  # pragma: no cover - not installed as a distribution
        import jacopy

        return getattr(jacopy, "__version__", "unknown")


@dataclass(frozen=True)
class Goal:
    lhs: Expr
    rhs: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.lhs, Expr) or not isinstance(self.rhs, Expr):
            raise TypeError("Goal sides must be Exprs")

    def __str__(self) -> str:
        return f"{self.lhs._repr_inner()} = {self.rhs._repr_inner()}"


@dataclass(frozen=True)
class Assumption:
    """One logical dependency of a result (module docstring)."""

    kind: str
    name: str
    owner: object = field(default=None, compare=False)
    instances: int = field(default=0, compare=False)
    example: Optional[Expr] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in ASSUMPTION_KINDS:
            raise ValueError(f"Assumption.kind must be one of {ASSUMPTION_KINDS}, got {self.kind!r}")

    @property
    def legacy(self) -> bool:
        return self.kind == "legacy"

    def __str__(self) -> str:
        who = f" [{self.owner!r}]" if self.owner is not None else ""
        times = f" ×{self.instances}" if self.instances > 1 else ""
        return f"{self.kind}: {self.name}{who}{times}"


@dataclass(frozen=True)
class Provenance:
    """One derivation ingredient (not an assumption)."""

    kind: str
    name: str
    owner: object = field(default=None, compare=False)
    count: int = field(default=1, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in PROVENANCE_KINDS:
            raise ValueError(f"Provenance.kind must be one of {PROVENANCE_KINDS}, got {self.kind!r}")

    def __str__(self) -> str:
        times = f" ×{self.count}" if self.count > 1 else ""
        return f"{self.kind}: {self.name}{times}"


@dataclass(frozen=True)
class Budget:
    steps: int
    max_steps: Optional[int] = None
    exhausted: bool = False

    def __str__(self) -> str:
        cap = f"/{self.max_steps}" if self.max_steps is not None else ""
        return f"{self.steps}{cap} steps" + (" (exhausted)" if self.exhausted else "")


# ------------------------------------------------------------------ #
# classification of a chain                                          #
# ------------------------------------------------------------------ #


def _dominant(owner, candidates):
    """The structure among ``candidates`` that ``owner`` is a
    SUB-STRUCTURE of — same tree-visible name and bundle, declarations
    a subset of the candidate's — with the most declarations; a proof
    run under fewer axioms of the same structure (the provers' reduced
    twins, e.g. "E without jacobi") depends on that structure, not on a
    second same-named owner. ``owner`` itself when none dominates."""
    from jacopy.proof.ownership import owner_name

    decl = getattr(owner, "declarations", None)
    if decl is None:
        return owner
    best, best_size = owner, len(decl)
    for c in candidates:
        cdecl = getattr(c, "declarations", None)
        if cdecl is None or c is owner:
            continue
        if (
            owner_name(c) == owner_name(owner)
            and getattr(c, "bundle", None) == getattr(owner, "bundle", None)
            and set(decl) <= set(cdecl)
            and len(cdecl) >= best_size
        ):
            best, best_size = c, len(cdecl)
    return best


def _classify(steps: Iterable[ProofStep], principal=None):
    """Walk the steps (children included) and bucket them into
    assumptions and provenance, aggregated by (kind, name, owner).
    Owners that are sub-structures of another owner seen in the proof
    (or of ``principal``) are folded into it (:func:`_dominant`)."""
    assumptions: dict = {}
    provenance: dict = {}
    owners_seen: list = [] if principal is None else [principal]

    def collect(seq):
        for s in seq:
            o = getattr(s, "owner", None)
            if o is not None and not any(o is x for x in owners_seen):
                owners_seen.append(o)
            collect(s.children)

    collect(steps)
    canon = {id(o): _dominant(o, owners_seen) for o in owners_seen}

    def bump(table, kind, name, owner, example=None):
        owner = canon.get(id(owner), owner) if owner is not None else None
        key = (kind, name, id(owner) if owner is not None else None)
        if key in table:
            k, n, o, c, ex = table[key]
            table[key] = (k, n, o, c + 1, ex)
        else:
            table[key] = (kind, name, owner, 1, example)

    def visit(seq):
        for s in seq:
            tag, role = s.provenance_tag, getattr(s, "role", None)
            owner = getattr(s, "owner", None)
            if tag == "theorem":
                bump(provenance, "theorem", s.rule, owner)
                cited = getattr(s, "cites", None)
                if cited is not None:
                    # TRANSITIVE requirements (Faz 8 step 3b): a verified
                    # citation contributes its own requirements; an
                    # unverified (legacy) one is itself a legacy assumption
                    if getattr(cited, "legacy", True):
                        bump(assumptions, "legacy", f"unverified citation: theorem {cited.name}", getattr(cited, "owner", None), s.before)
                    else:
                        for a in cited.requires:
                            if a.kind in ("declared", "legacy", "scope"):
                                bump(assumptions, a.kind, a.name, a.owner, a.example)
            elif tag == "axiom":
                if role == "assumption":
                    bump(assumptions, "declared", s.rule, owner, s.before)
                elif role == "definition":
                    bump(provenance, "definition", s.rule, owner)
                else:
                    # the rule recorded no role: a hand-built or unmigrated
                    # step — kept as an assumption, marked legacy
                    bump(assumptions, "legacy", s.rule, owner, s.before)
            elif tag == "computation":
                bump(provenance, "computation", s.rule, owner)
            else:
                low = s.rule.lower()
                kind = "canonicalization" if any(low.startswith(c) or f" {c}" in low for c in _CANONICAL_RULES) else "engine-step"
                bump(provenance, kind, s.rule, owner)
            # a whole-expression wrapper carries its local step as the only
            # child with the same rule: that child is the same event
            kids = [c for c in s.children if not (len(s.children) == 1 and c.rule == s.rule and c.provenance_tag == tag)]
            visit(kids)

    visit(steps)
    req = [Assumption(k, n, o, c, ex) for (k, n, o, c, ex) in assumptions.values()]
    prov = [Provenance(k, n, o, c) for (k, n, o, c, _) in provenance.values()]
    # the structures behind the declared axioms are assumptions too
    seen_owners: list = []
    for a in req:
        if a.kind == "declared" and a.owner is not None and not any(a.owner is o or a.owner == o for o in seen_owners):
            seen_owners.append(a.owner)
    for o in seen_owners:
        decl = getattr(o, "declarations", None)
        text = f"{o!r}" + (f" declaring {sorted(decl)}" if decl is not None else "")
        req.append(Assumption("structure", text, o))
    return tuple(req), tuple(prov)


def _whole_step(before: Expr, after: Expr, local: ProofStep) -> ProofStep:
    """A whole-expression step carrying the engine's local step (a
    subterm rewrite) as its child; the same rule, tag, owner, role and
    citation, so classification is unchanged."""
    if local.before == before and local.after == after:
        return local
    return ProofStep(
        before, after, rule=local.rule, justification=local.justification, children=[local],
        provenance_tag=local.provenance_tag, owner=local.owner, role=getattr(local, "role", None),
        cites=getattr(local, "cites", None),
    )


def _reversed_step(s: ProofStep) -> ProofStep:
    return ProofStep(
        s.after, s.before, rule=s.rule, justification=(s.justification + " (reversed)").strip(),
        children=list(s.children), provenance_tag=s.provenance_tag, owner=s.owner, role=getattr(s, "role", None),
    )


# ------------------------------------------------------------------ #
# the result                                                         #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ProofResult:
    goal: Goal
    status: str
    requires: Tuple[Assumption, ...] = ()
    provenance: Tuple[Provenance, ...] = ()
    chain: Optional[ProofChain] = field(default=None, compare=False)
    residual: Optional[Expr] = None
    budget: Budget = field(default_factory=lambda: Budget(0))
    generality: str = "generic-function"
    owner: object = field(default=None, compare=False)
    reason: str = ""
    version: str = field(default_factory=package_version, compare=False)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"ProofResult.status must be one of {STATUSES}, got {self.status!r}")
        from jacopy.proof.theorems import GENERALITY_TAGS

        if self.generality not in GENERALITY_TAGS:
            raise ValueError(f"ProofResult.generality must be one of {GENERALITY_TAGS}, got {self.generality!r}")
        if self.status == "CLOSED" and self.residual is not None:
            raise ValueError("a CLOSED result has no residual")
        if self.status in ("RESIDUAL", "BUDGET") and self.residual is None:
            raise ValueError(f"a {self.status} result must record its residual")

    # ---- views ------------------------------------------------------- #

    @property
    def closed(self) -> bool:
        return self.status == "CLOSED"

    @property
    def declared(self) -> Tuple[Assumption, ...]:
        return tuple(a for a in self.requires if a.kind == "declared")

    @property
    def legacy(self) -> Tuple[Assumption, ...]:
        """The recorded-but-unverified assumptions (never dropped)."""
        return tuple(a for a in self.requires if a.legacy)

    @property
    def textual(self) -> Tuple[Assumption, ...]:
        """The record's ``from_axioms`` prose no step accounts for:
        visible, never dropped, NOT a verification input (prose is not
        a trust boundary; the chain's structural record is)."""
        return tuple(a for a in self.requires if a.kind == "textual")

    @property
    def theorems_cited(self) -> Tuple[Provenance, ...]:
        return tuple(p for p in self.provenance if p.kind == "theorem")

    def summary(self) -> str:
        """``CLOSED`` always reads "under" its assumptions; an instance
        proof says so; a residual is shown as a residual, not as a
        counterexample."""
        lines = [f"{self.status}: {self.goal}"]
        if self.status == "CLOSED":
            lines[0] += "  — under the assumptions below" if self.requires else "  — assumption-free"
        if self.status in ("RESIDUAL", "BUDGET"):
            lines.append(f"  residual (unreduced, NOT a disproof): {self.residual._repr_inner()}")
        if self.status == "INVALID":
            lines.append(f"  reason: {self.reason}")
        if self.generality == "instance":
            lines.append("  generality: instance — this certificate is not a general theorem")
        else:
            lines.append(f"  generality: {self.generality}")
        for a in self.requires:
            lines.append(f"  requires {a}")
        for p in self.provenance:
            lines.append(f"  via {p}")
        lines.append(f"  budget: {self.budget}; jacopy {self.version}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    # ---- constructors ------------------------------------------------ #

    @classmethod
    def invalid(cls, goal: Goal, reason: str, **kw) -> "ProofResult":
        return cls(goal=goal, status="INVALID", reason=reason, **kw)

    @classmethod
    def from_chain(
        cls,
        chain: ProofChain,
        goal: Goal,
        *,
        status: Optional[str] = None,
        max_steps: Optional[int] = None,
        generality: str = "generic-function",
        owner=None,
        extra: Sequence[Assumption] = (),
        reason: str = "",
    ) -> "ProofResult":
        """Classify ``chain`` against ``goal``. The chain must start at
        ``goal.lhs``; its final expression decides ``CLOSED`` (equal to
        ``goal.rhs``) or ``RESIDUAL``; ``status="BUDGET"`` records an
        exhausted budget. Assumptions the chain does not record
        structurally are kept as ``legacy``, never ignored."""
        if not isinstance(chain, ProofChain):
            raise TypeError("from_chain expects a ProofChain")
        if not isinstance(goal, Goal):
            raise TypeError("from_chain expects a Goal")
        if len(chain) and chain.initial != goal.lhs:
            raise ValueError(
                f"the chain starts at {chain.initial._repr_inner()}, not at the goal's "
                f"left-hand side {goal.lhs._repr_inner()}"
            )
        final = chain.final if len(chain) else goal.lhs
        req, prov = _classify(chain.steps, principal=owner)
        req = req + tuple(extra)
        if generality == "instance":
            req = req + (Assumption("scope", "instance proof — the certificate is not generalized"),)
        exhausted = status == "BUDGET"
        if status is None:
            status = "CLOSED" if final == goal.rhs else "RESIDUAL"
        residual = None if status == "CLOSED" else final
        return cls(
            goal=goal, status=status, requires=req, provenance=prov, chain=chain, residual=residual,
            budget=Budget(len(chain), max_steps, exhausted), generality=generality, owner=owner, reason=reason,
        )

    @classmethod
    def from_theorem(cls, theorem) -> "ProofResult":
        """Wrap a :class:`~jacopy.proof.theorems.Theorem` (the old
        ``(chain, theorem)`` return shape). The record's closure is
        taken as CLOSED; textual ``from_axioms`` entries no step
        accounts for, and a chain that does not literally run
        ``lhs → rhs``, are kept as ``legacy`` assumptions."""
        from jacopy.proof.theorems import Theorem

        if not isinstance(theorem, Theorem):
            raise TypeError("from_theorem expects a Theorem")
        goal = Goal(theorem.lhs, theorem.rhs)
        chain = theorem.proof
        req, prov = _classify(chain.steps, principal=theorem.owner)
        known = {a.name for a in req} | {p.name for p in prov}
        legacy: List[Assumption] = []
        for text in theorem.from_axioms:
            if text not in known and not any(text in k for k in known):
                legacy.append(Assumption("textual", f"from_axioms: {text}", theorem.owner))
        # (a Theorem's chain is the engine's transcript of LOCAL rewrites;
        # its endpoints are not the whole sides, so the chain's shape is
        # not judged here — the record asserts the closure, the
        # requirements say what it depends on)
        req = req + tuple(legacy)
        if theorem.generality == "instance":
            req = req + (Assumption("scope", "instance proof — the certificate is not generalized"),)
        return cls(
            goal=goal, status="CLOSED", requires=req, provenance=prov, chain=chain, residual=None,
            budget=Budget(len(chain)), generality=theorem.generality, owner=theorem.owner,
        )


def structural_requires(theorem) -> Tuple[Assumption, ...]:
    """The requirement records of a theorem derived from its OWN chain
    (Faz 8 step 3b): declared axiom instances, the structures behind
    them, the transitive requirements of cited theorems, a scope entry
    for an instance proof, and a legacy entry for every ``from_axioms``
    text no step accounts for. The chain's endpoint shape is not a
    requirement and is not judged here (``ProofResult.from_theorem``
    flags it)."""
    req, prov = _classify(theorem.proof.steps, principal=theorem.owner)
    known = {a.name for a in req} | {p.name for p in prov}
    out = list(req)
    for text in theorem.from_axioms:
        if text not in known and not any(text in k for k in known):
            out.append(Assumption("textual", f"from_axioms: {text}", theorem.owner))
    if theorem.generality == "instance" and not any(a.kind == "scope" for a in out):
        out.append(Assumption("scope", "instance proof — the certificate is not generalized"))
    return tuple(out)


# ------------------------------------------------------------------ #
# the driver                                                         #
# ------------------------------------------------------------------ #


def check_goal(lhs: Expr, rhs: Expr, registry=None) -> Optional[str]:
    """The reason a goal is ill-typed, or ``None`` when its two sides
    have compatible kinds (:func:`~jacopy.central.objects.kind.kind_of`;
    zero fits every kind, unknown is not judged)."""
    from jacopy.central.objects.kind import kind_of

    a, b = kind_of(lhs, registry), kind_of(rhs, registry)
    for side, k in (("left", a), ("right", b)):
        if k.kind == "mismatch":
            return f"the {side}-hand side is a sum of different known types"
    if a.kind == "zero" or b.kind == "zero" or not (a.known and b.known):
        return None
    if a.kind != b.kind:
        return f"the sides have different kinds: {a.kind} vs {b.kind}"
    ca, cb = a.concrete_degree, b.concrete_degree
    if ca is not None and cb is not None and ca != cb:
        return f"the sides have different degrees: {ca} vs {cb}"
    if a.signature is not None and b.signature is not None and a.signature != b.signature:
        return f"the sides have different tensor signatures: {a.signature} vs {b.signature}"
    return None


def prove(
    engine,
    lhs: Expr,
    rhs: Expr = Integer(0),
    *,
    registry=None,
    max_steps: int = 1024,
    generality: str = "generic-function",
    owner=None,
    simplify_normal_forms: bool = True,
) -> "ProofResult":
    """Normalize both sides with ``engine`` (optionally followed by the
    algebraic simplifier) and compare. Returns a :class:`ProofResult`:
    ``INVALID`` without running when the goal is ill-typed, ``BUDGET``
    when ``max_steps`` runs out, ``CLOSED`` when the normal forms
    agree, else ``RESIDUAL`` with the surviving difference. The chain
    runs ``lhs → normal form ← rhs`` (the right side's steps reversed)."""
    goal = Goal(lhs, rhs)
    reason = check_goal(lhs, rhs, registry)
    if reason is not None:
        return ProofResult.invalid(goal, reason, generality=generality, owner=owner)

    def run(expr: Expr, cap: int):
        steps: List[ProofStep] = []
        cur = expr
        for _ in range(cap):
            nxt, step = engine.expand_once(cur)
            if step is None:
                break
            # the engine's step is LOCAL (the rewritten subterm); the
            # result's chain is a whole-expression transcript, so wrap it
            # — the local step rides along as the child that names the site
            steps.append(_whole_step(cur, nxt, step))
            cur = nxt
        else:
            nxt, step = engine.expand_once(cur)
            if step is not None:
                return cur, steps, True
        if simplify_normal_forms:
            from jacopy.algorithms.simplify import simplify

            simplified = simplify(cur, registry)
            if simplified != cur:
                steps.append(ProofStep(cur, simplified, rule="simplify", justification="canonical form"))
                cur = simplified
        return cur, steps, False

    left, lsteps, lexh = run(lhs, max_steps)
    if lexh:
        chain = ProofChain(lsteps)
        return ProofResult.from_chain(chain, goal, status="BUDGET", max_steps=max_steps, generality=generality, owner=owner)
    remaining = max(max_steps - len(lsteps), 1)
    right, rsteps, rexh = run(rhs, remaining)
    if rexh:
        chain = ProofChain(lsteps + [_reversed_step(s) for s in reversed(rsteps)])
        return ProofResult.from_chain(chain, Goal(lhs, chain.final if len(chain) else lhs), status="BUDGET", max_steps=max_steps, generality=generality, owner=owner)
    if left == right:
        chain = ProofChain(lsteps + [_reversed_step(s) for s in reversed(rsteps)])
        return ProofResult.from_chain(chain, goal, status="CLOSED", max_steps=max_steps, generality=generality, owner=owner)
    from jacopy.core.expr import Neg, Sum

    diff = Sum(left, Neg(right)) if right != Integer(0) else left
    if simplify_normal_forms:
        from jacopy.algorithms.simplify import simplify

        diff = simplify(diff, registry)
    steps = list(lsteps)
    if diff != left:
        steps.append(ProofStep(left, diff, rule="residual: lhs − rhs", justification="normal forms differ; the difference is the residual"))
    chain = ProofChain(steps)
    return ProofResult.from_chain(chain, goal, status="RESIDUAL", max_steps=max_steps, generality=generality, owner=owner)


__all__ = [
    "ASSUMPTION_KINDS",
    "Assumption",
    "Budget",
    "Goal",
    "PROVENANCE_KINDS",
    "ProofResult",
    "Provenance",
    "STATUSES",
    "check_goal",
    "package_version",
    "prove",
    "structural_requires",
]
