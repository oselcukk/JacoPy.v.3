"""
LWX Courant-algebroid axiomatics and the Uchino redundancy theorems
(Phase 7.A) [uchino.pdf, math/0204010].

The Liu-Weinstein-Xu definition lists five axioms [C1]-[C5] plus the
defining relation ``D = ½β⁻¹ρ*d``. Uchino's remark: **[C3] (the
Leibniz rule with pairing defect), [C4] (ρ∘D = 0) and the D-formula
are REDUNDANT** — they follow from [C2] (anchor morphism), [C5]
(invariance of the pairing) and the bare Leibniz rule (L) for the
otherwise-arbitrary map ``D : C^∞(M) → ΓE``.

Mechanization strategy (definition policy §1 — nothing derivable is
assumed):

* the pairing is the Phase 3 :class:`EMetric` — symmetric and
  non-degenerate BY DEFINITION [MC Def 3.1], matching LWX's
  "nondegenerate symmetric bilinear form";
* the bracket is the opaque :class:`AlgebroidBracket` — ℝ-bilinear by
  definition, SKEW by the LWX definition (declared, since it is part
  of the definition, not a derived axiom);
* ``D`` is the opaque section-valued :class:`CourantD` whose ONLY
  rule is (L), :class:`DLeibnizDefinition`. In particular the Phase 3
  coboundary pairing ``⟨Df,u⟩ = ρ(u)(f)`` is NOT available here —
  the value ``(Df, y) = ½ρ(y)(f)`` is exactly what Proposition 2.2
  proves;
* [C5] is the declared rewrite :class:`CourantInvarianceDeclaration`
  (the D-term-carrying analogue of the metric-invariance rule C1);
* [C2] is the Phase 3 ``anchor-morphism`` declaration.

The proofs are the [B 4.11] two-leg tactic (Phase 3.E): expand the
same anchored action of a pairing two ways, subtract, strip the
generic probe section by non-degeneracy, solve. Honest-fail: without
the [C5] declaration each proof stalls with a residual.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.algebroid.context import (
    Algebroid,
    AlgebroidBracket,
    AnchoredVF,
    EMetric,
    algebroid,
)
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.theorems import (
    _fresh_section,
    _metric_coefficient_of,
    _normalize,
    _strip_sign,
    _terms,
)
from jacopy.algorithms.simplify import simplify


class CourantD(Derivation):
    """``D(f)`` — the abstract map ``D : C^∞(M) → ΓE`` of the LWX
    definition, an OPAQUE section-valued atom.

    Its only structure at this layer is the Leibniz rule (L)
    (:class:`DLeibnizDefinition`). Everything else about it —
    ``(Df, y) = ½ρ(y)(f)``, ``ρ∘D = 0`` — is a THEOREM (Uchino
    Prop 2.1-2.2), never a defining rule.
    """

    __slots__ = ("_algebroid_name", "_function")

    def __init__(
        self,
        algebroid_name: str,
        f: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(f, Expr):
            raise TypeError("CourantD requires an Expr function")
        display = (
            name
            if name is not None
            else f"D({f._repr_inner()})"
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._function = f

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def wedge_degree(self) -> Degree:
        """``Df`` is a section (1-vector)."""
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._function,)

    def with_slots(self, f: Expr) -> "CourantD":
        return CourantD(self._algebroid_name, f)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._algebroid_name,
            self._function,
        )


class DLeibnizDefinition(Definition):
    """(L) — the ONLY assumed structure of ``D``:
    ``D(f·g) = f·D(g) + g·D(f)`` (both factors certainly scalar).

    Deliberately NOTHING else: no ℝ-linearity, no ``D(c) = 0`` — the
    Uchino derivations use the bare (L) and this module tracks that
    assumption budget exactly.
    """

    anchor = CourantD

    def __init__(
        self,
        alg: Algebroid,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._alg = alg
        self._registry = registry
        self.name = (
            f"(L) Leibniz rule for D ({alg.name}): "
            "D(f·g) = f·D(g) + g·D(f)"
        )

    def _split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            rest = expr.children[1:]
            rest_expr = (
                rest[0] if len(rest) == 1 else Product(*rest)
            )
            if is_scalar_function(
                head, self._registry
            ) and is_scalar_function(rest_expr, self._registry):
                return head, rest_expr
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, CourantD)
            and expr.algebroid_name == self._alg.name
            and self._split(expr.function) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        f, g = self._split(expr.function)
        return Sum(
            Product(f, expr.with_slots(g)),
            Product(g, expr.with_slots(f)),
        )


class CourantInvarianceDeclaration(Definition):
    """[C5] — the declared invariance axiom of the LWX definition:

    ``ρ(x)(y, z) → ([x,y] + D(x,y), z) + (y, [x,z] + D(x,z))``

    where ``D(x,y)`` abbreviates ``D`` applied to the pairing
    function ``(x, y)``. The D-term-carrying analogue of the Phase 3
    metric-invariance rule C1 [B 4.10].
    """

    anchor = Act

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"[C5] invariance ({alg.name}): ρ(x)(y,z) = "
            "([x,y]+D(x,y), z) + (y, [x,z]+D(x,z))"
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
        x = expr.op.section
        y, z = expr.arg.u, expr.arg.v
        g = self._alg.metric
        br = self._alg.bracket
        D = lambda u, v: CourantD(self._alg.name, g(u, v))
        return Sum(
            g(br(x, y), z),
            g(D(x, y), z),
            g(y, br(x, z)),
            g(y, D(x, z)),
        )


class C3IdentityDeclaration(Definition):
    """The Prop 2.1(i) IDENTITY assumed as a rewrite (for Rem 2.1,
    where it is the hypothesis):

    ``[x, f·R] → f·[x,R] + ρ(x)(f)·R − (x, R)·D(f)``

    ``peel="one"`` strips the leading scalar factor only (iterated
    application); ``peel="all"`` strips the FULL scalar prefix in a
    single application (``f`` above is then the whole product). The
    two orders are both sound readings of the same identity — their
    difference is exactly the Leibniz defect of ``D`` (Rem 2.1).
    """

    anchor = AlgebroidBracket

    def __init__(
        self,
        alg: Algebroid,
        registry: Optional[PropertyRegistry] = None,
        *,
        peel: str = "one",
    ) -> None:
        if peel not in ("one", "all"):
            raise ValueError("peel must be 'one' or 'all'")
        self._alg = alg
        self._registry = registry
        self._peel = peel
        self.name = (
            f"assumed identity (i) ({alg.name}, peel-{peel}): "
            "[x, f·R] = f·[x,R] + ρ(x)(f)·R − (x,R)·Df"
        )

    def _split(self, slot: Expr):
        if not (
            isinstance(slot, Product)
            and len(slot.children) >= 2
        ):
            return None
        scalars: List[Expr] = []
        rest = list(slot.children)
        while rest and is_scalar_function(
            rest[0], self._registry
        ):
            scalars.append(rest.pop(0))
            if self._peel == "one":
                break
        if not scalars or not rest:
            return None
        f: Expr = (
            scalars[0]
            if len(scalars) == 1
            else Product(*scalars)
        )
        R: Expr = rest[0] if len(rest) == 1 else Product(*rest)
        return f, R

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._alg.name
            and self._split(expr.v) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        x = expr.u
        f, R = self._split(expr.v)
        return Sum(
            Product(f, self._alg.bracket(x, R)),
            Product(Act(self._alg.anchor(x), f), R),
            Neg(
                Product(
                    self._alg.metric(x, R),
                    CourantD(self._alg.name, f),
                )
            ),
        )


class SkewSwapCompositeDefinition(Definition):
    """The declared skew-symmetry applied at a COMPOSITE left slot:
    ``[f·x, y] → −[y, f·x]`` (scalar-product left, plain right).

    The Phase 3 :class:`AntisymmetryDeclaration` deliberately fires
    on plain slots only (linearity splits keep precedence there);
    the Uchino 2.2 derivation needs the swap exactly where the LEFT
    slot is composite, so the right-slot Leibniz theorem (Prop
    2.1(i)) can open the result. Same axiom, different site — the
    sign rule holds for ALL sections.
    """

    anchor = AlgebroidBracket

    def __init__(
        self,
        alg: Algebroid,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._alg = alg
        self._registry = registry
        self.name = (
            f"skew swap at composite slot ({alg.name}): "
            "[f·x, y] = −[y, f·x]"
        )

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._alg.name
            and self._alg.declares("antisymmetric")
        ):
            return False
        u, v = expr.u, expr.v
        if isinstance(v, (Sum, Neg, Product)):
            return False
        return (
            isinstance(u, Product)
            and len(u.children) >= 2
            and is_scalar_function(
                u.children[0], self._registry
            )
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Neg(
            AlgebroidBracket(
                expr.algebroid_name, expr.v, expr.u
            )
        )


class DPairingUnfoldDefinition(Definition):
    """THEOREM-classified rule citing the proven Prop 2.2:
    ``(Df, u) → ½ρ(u)(f)`` (unfold direction). Registered only in
    engines that do NOT carry [C5], so the rewrite terminates (the
    produced action has no pairing argument left)."""

    anchor = EMetric

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"cite Prop 2.2 ({alg.name}): (Df, u) = ½ρ(u)(f)"
        )

    def _split(self, expr: EMetric):
        for a, b in ((expr.u, expr.v), (expr.v, expr.u)):
            if (
                isinstance(a, CourantD)
                and a.algebroid_name == self._alg.name
                and not isinstance(b, CourantD)
            ):
                return a, b
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, EMetric)
            and expr.algebroid_name == self._alg.name
            and self._split(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        Dnode, u = self._split(expr)
        return Product(
            Rational(1, 2),
            Act(self._alg.anchor(u), Dnode.function),
        )


class DDZeroDefinition(Definition):
    """THEOREM-classified rule citing the proven Prop 2.1(ii) +
    Prop 2.2: ``ρ∘D = 0``, hence ``(Da, Db) → 0`` and
    ``ρ(D(a))(h) → 0``."""

    anchor = (EMetric, Act)

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"cite [C4] theorem ({alg.name}): (Da, Db) = 0, "
            "ρ(Da)(h) = 0"
        )

    def matches(self, expr: Expr) -> bool:
        if (
            isinstance(expr, EMetric)
            and expr.algebroid_name == self._alg.name
            and isinstance(expr.u, CourantD)
            and isinstance(expr.v, CourantD)
        ):
            return True
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, AnchoredVF)
            and expr.op.algebroid_name == self._alg.name
            and isinstance(expr.op.section, CourantD)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


class DNumericHomogeneityDeclaration(Definition):
    """THEOREM-classified rule: ``D(c·h) = c·D(h)`` and ``D(c) = 0``
    for NUMERIC ``c``.

    The bare (L) alone does not provide this — but under the same
    assumptions as Prop 2.2 it is DERIVED, not declared (2026-09-08
    audit, compliance finding 7):

        ⟨D(c·f) − c·D(f), y⟩ = ½ρ(y)(c·f) − c·½ρ(y)(f) = 0

    for every generic ``y`` (Prop 2.2 twice + ℝ-linearity of the
    vector-field action), hence ``D(c·f) = c·D(f)`` by
    non-degeneracy; ``D(c) = 0`` is the ``f = 1`` case combined with
    ``D(1) = 0`` (from (L): ``D(1·1) = 2·D(1)``). The mechanical
    derivation is :func:`prove_d_numeric_homogeneity`; the classical
    Rem 2.2 corollary therefore needs NO extra axiom. (The class
    name is kept for API stability; ``Declaration`` is historical —
    the rule cites a theorem.)"""

    anchor = CourantD

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"ℝ-homogeneity of D ({alg.name}) — DERIVED from "
            "Prop 2.2 + non-degeneracy: D(c·h) = c·D(h), D(c) = 0"
        )

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, CourantD)
            and expr.algebroid_name == self._alg.name
        ):
            return False
        if isinstance(expr.function, (Integer, Rational)):
            # D(c) = c·D(1) (homogeneity) and D(1) = 0 (from (L):
            # D(1·1) = 2·D(1)).
            return True
        return isinstance(expr.function, Product) and isinstance(
            expr.function.children[0], (Integer, Rational)
        )

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr.function, (Integer, Rational)):
            return Integer(0)
        c = expr.function.children[0]
        rest = expr.function.children[1:]
        return Product(
            c,
            expr.with_slots(
                rest[0] if len(rest) == 1 else Product(*rest)
            ),
        )


def courant_context(name: str = "E") -> Algebroid:
    """The LWX Courant DATA as a context: opaque skew bracket
    (``antisymmetric`` is part of the LWX definition), anchor with
    [C2], definitionally symmetric non-degenerate pairing. [C3],
    [C4], [C1] and the D-formula are deliberately ABSENT — the first
    three of these are what Phase 7.A proves or declares separately.
    """
    return algebroid(
        name, declare=("anchor-morphism", "antisymmetric")
    )


def courant_engine(
    alg: Algebroid,
    registry: Optional[PropertyRegistry] = None,
    *,
    invariance: bool = True,
    theorems: Tuple = (),
) -> ExpansionEngine:
    """The Phase 3 declaration engine + [C5] + (L) (+ optionally
    cited theorems, e.g. the proven 2.1(i) instances). Registration
    order puts the Courant rules first (match precedence).
    ``invariance=False`` drops [C5] — the honest-fail engine."""
    from jacopy.proof.theorems import TheoremDefinition

    rules: List[Definition] = []
    if invariance:
        rules.append(CourantInvarianceDeclaration(alg))
    rules.append(DLeibnizDefinition(alg, registry))
    for t in theorems:
        rules.append(TheoremDefinition(t))
    if theorems and alg.declares("antisymmetric"):
        # The composite-slot swap is useful exactly when a cited
        # right-Leibniz theorem can open the swapped bracket.
        rules.append(SkewSwapCompositeDefinition(alg, registry))
    eng = ExpansionEngine(rules)
    for d in algebroid_engine(alg, registry=registry).definitions:
        eng.register(d)
    return eng


def _strip_generic_scalar_factor(
    expr: Expr, factor: Expr
) -> Expr:
    """Divide every term of ``expr`` by the common scalar ``factor``
    (a pairing of GENERIC sections — non-degeneracy makes it
    generically invertible: choose sections with ``(x,y) = 1``).
    Raises :class:`ProofFailure` when a term does not carry it —
    the step would then be unsound."""
    out: List[Expr] = []
    for term in _terms(expr):
        sign, core = _strip_sign(term)
        factors = (
            list(core.children)
            if isinstance(core, Product)
            else [core]
        )
        if factor not in factors:
            raise ProofFailure(
                "generic-factor strip: the term "
                f"{term._repr_inner()} does not carry the factor "
                f"{factor._repr_inner()}"
            )
        factors.remove(factor)
        stripped: Expr = (
            Integer(1)
            if not factors
            else (
                factors[0]
                if len(factors) == 1
                else Product(*factors)
            )
        )
        out.append(Neg(stripped) if sign < 0 else stripped)
    return (
        Integer(0)
        if not out
        else (out[0] if len(out) == 1 else Sum(*out))
    )


def prove_left_leibniz_c3(
    alg: Algebroid,
    x: Expr,
    y: Expr,
    f: Expr,
    *,
    w: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
    invariance: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """Uchino Proposition 2.1(i): the LWX axiom [C3]

    ``[x, f·y] = f·[x,y] + ρ(x)(f)·y − (x,y)·Df``

    is REDUNDANT — it follows from [C5] + (L) + non-degeneracy.
    The two-leg tactic at the composite site ``ρ(x)(f·y, w)``:

    1. leg A — apply [C5] with the composite section ``f·y``, then
       normalize (bilinearity opens the pairings; (L) splits
       ``D((x, f·y)) = D(f·(x,y))``),
    2. leg B — normalize the same start engine-first (bilinearity
       pulls the scalar out, the product rule splits, [C5] fires on
       ``ρ(x)(y, w)``),
    3. subtract, strip the generic probe ``w`` (non-degeneracy),
    4. solve for ``[x, f·y]`` and recognize the [C3] form.
    """
    if not isinstance(alg, Algebroid):
        raise TypeError("prove_left_leibniz_c3 expects an Algebroid")
    if alg.declares("right-leibniz"):
        raise ValueError(
            "right-leibniz is DECLARED on this context — [C3] would "
            "be an axiom, not a theorem; use a bare courant_context()"
        )

    if w is None:
        w = _fresh_section(alg, x, y, f)
    engine = courant_engine(alg, registry, invariance=invariance)

    start = Act(alg.anchor(x), alg.metric(Product(f, y), w))

    if not invariance:
        # Honest-fail: without [C5] the bracket terms never appear —
        # the two legs agree and there is no identity to strip.
        n0, _ = _normalize(start, engine, registry)
        raise ProofFailure(
            "Uchino 2.1(i) needs the [C5] invariance declaration; "
            "without it the probe normalizes inertly to "
            + n0._repr_inner()[:120]
        )

    c5 = CourantInvarianceDeclaration(alg)
    if not c5.matches(start):
        raise ProofFailure(  # pragma: no cover - shape fixed above
            "[C5] does not apply to " + start._repr_inner()
        )
    after_c5 = c5.rewrite(start)
    step_c5 = ProofStep(
        start,
        after_c5,
        rule=c5.name,
        justification="declared axiom applied at the composite site",
        provenance_tag="axiom",
    )
    n_a, steps_a = _normalize(after_c5, engine, registry)
    step_leg_a = ProofStep(
        after_c5,
        n_a,
        rule="normalize leg A (bilinearity, (L) on D, canonical form)",
        justification="engine normal form of the [C5] expansion",
    )
    for s in steps_a:
        step_leg_a.add_child(s)

    n_b, steps_b = _normalize(start, engine, registry)
    step_leg_b = ProofStep(
        start,
        n_b,
        rule=(
            "normalize leg B (scalar pull-out, product rule, [C5] "
            "on ρ(x)(y,w))"
        ),
        justification="engine normal form of the same expression",
    )
    for s in steps_b:
        step_leg_b.add_child(s)

    diff, diff_steps = _normalize(
        Sum(n_a, Neg(n_b)), engine, registry
    )
    step_diff = ProofStep(
        Sum(n_a, Neg(n_b)),
        diff,
        rule="collect: leg A − leg B = g(X, w)",
        justification=(
            "common terms cancel; bilinearity groups the rest "
            "against the probe section w"
        ),
    )
    for s in diff_steps:
        step_diff.add_child(s)

    contributions = _metric_coefficient_of(diff, w, alg)
    X = simplify(
        contributions[0]
        if len(contributions) == 1
        else Sum(*contributions),
        registry,
    )
    step_nondeg = ProofStep(
        diff,
        Integer(0),
        rule="non-degeneracy of ( , ) (generic probe section w)",
        justification=(
            "g(X, w) = 0 for the generic w; the pairing is "
            "non-degenerate by definition, hence X = 0"
        ),
    )

    target = alg.bracket(x, Product(f, y))
    solved: Optional[Expr] = None
    x_terms = list(_terms(X))
    for i, term in enumerate(x_terms):
        sign, core = _strip_sign(term)
        if core == target:
            rest = x_terms[:i] + x_terms[i + 1 :]
            rest_sum: Expr = (
                Integer(0)
                if not rest
                else (rest[0] if len(rest) == 1 else Sum(*rest))
            )
            solved = simplify(
                Neg(rest_sum) if sign > 0 else rest_sum, registry
            )
            break
    if solved is None:
        raise ProofFailure(
            "Uchino 2.1(i): the section identity "
            f"{X._repr_inner()} = 0 does not contain "
            f"{target._repr_inner()}"
        )
    step_solve = ProofStep(
        target,
        solved,
        rule="solve X = 0 for [x, f·y]",
        justification="rearrange the vanishing section identity",
    )

    rhs = Sum(
        Product(f, alg.bracket(x, y)),
        Product(Act(alg.anchor(x), f), y),
        Neg(Product(alg.metric(x, y), CourantD(alg.name, f))),
    )
    recog = ExpandAndSimplify().prove(
        solved,
        rhs,
        registry=registry,
        engine=courant_engine(alg, registry),
    )
    step_recog = ProofStep(
        target,
        rhs,
        rule="[C3] form recognized",
        justification="canonical-form comparison of the solved side",
    )
    for s in recog:
        step_recog.add_child(s)

    chain = ProofChain(
        [
            step_c5,
            step_leg_a,
            step_leg_b,
            step_diff,
            step_nondeg,
            step_solve,
            step_recog,
        ]
    )
    xn, yn, fn = (
        x._repr_inner(),
        y._repr_inner(),
        f._repr_inner(),
    )
    theorem = Theorem(
        name=f"uchino_c3_{alg.name}_{xn}_{yn}_{fn}",
        statement=(
            f"[{xn}, {fn}·{yn}] = {fn}·[{xn},{yn}] + "
            f"ρ({xn})({fn})·{yn} − ({xn},{yn})·D{fn} on {alg.name} "
            "(LWX [C3] is redundant — Uchino Prop 2.1(i))"
        ),
        lhs=target,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"[C5] invariance ({alg.name})",
            "(L) Leibniz rule for D",
            "non-degeneracy of the pairing (definitional)",
        ),
        notes=(
            "uchino.pdf Prop 2.1(i): ρ(x)(f·y, w) expanded two ways "
            "against a generic probe section w"
        ),
    )
    return chain, theorem


def _two_leg_difference(
    start: Expr,
    engine_a: ExpansionEngine,
    engine_b: ExpansionEngine,
    registry,
    *,
    label_a: str,
    label_b: str,
    diff_engine: Optional[ExpansionEngine] = None,
) -> Tuple[Expr, List[ProofStep]]:
    """The 6.B two-normal-form recipe as a step builder: normalize
    the SAME start under two sound rule sets and collect the
    difference (a proven zero rearranged into an identity)."""
    n_a, steps_a = _normalize(start, engine_a, registry)
    step_a = ProofStep(
        start, n_a, rule=label_a,
        justification="engine normal form (rule set A)",
    )
    for s in steps_a:
        step_a.add_child(s)
    n_b, steps_b = _normalize(start, engine_b, registry)
    step_b = ProofStep(
        start, n_b, rule=label_b,
        justification="engine normal form (rule set B)",
    )
    for s in steps_b:
        step_b.add_child(s)
    diff, diff_steps = _normalize(
        Sum(n_a, Neg(n_b)),
        diff_engine if diff_engine is not None else engine_b,
        registry,
    )
    step_d = ProofStep(
        Sum(n_a, Neg(n_b)), diff,
        rule="collect: leg A − leg B",
        justification=(
            "the same expression normalized twice — the difference "
            "is a proven zero, rearranged"
        ),
    )
    for s in diff_steps:
        step_d.add_child(s)
    return diff, [step_a, step_b, step_d]


def prove_anchor_annihilates_d(
    alg: Algebroid,
    x: Expr,
    y: Expr,
    f: Expr,
    h: Expr,
    *,
    c3: Optional[Theorem] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Uchino Proposition 2.1(ii): [C4] is redundant —

    ``ρ(Df) = 0``  (probed on the generic scalar ``h``)

    from [C2] + Prop 2.1(i). The two-leg tactic on
    ``ρ([x, f·y])(h)``: leg A goes anchor-morphism-first ([C2] on
    the composite bracket), leg B rewrites ``[x, f·y]`` by the
    cited 2.1(i) theorem first; the difference is
    ``(x,y)·ρ(Df)(h)`` and the generic pairing factor strips.
    """
    if c3 is None:
        _, c3 = prove_left_leibniz_c3(
            alg, x, y, f, registry=registry
        )
    eng_a = courant_engine(alg, registry)
    eng_b = courant_engine(alg, registry, theorems=(c3,))

    start = Act(
        alg.anchor(alg.bracket(x, Product(f, y))), h
    )
    diff, steps = _two_leg_difference(
        start, eng_a, eng_b, registry,
        label_a=(
            "normalize leg A ([C2] anchor morphism on the "
            "composite bracket, Lie action expansion)"
        ),
        label_b=(
            "normalize leg B (cite Prop 2.1(i) on [x, f·y], "
            "anchor linearity, [C2])"
        ),
    )
    if diff == Integer(0):
        raise ProofFailure(
            "Uchino 2.1(ii): the two legs coincide — no identity "
            "to extract (rule sets degenerate)"
        )

    gxy, _ = _normalize(alg.metric(x, y), eng_a, registry)
    stripped = simplify(
        _strip_generic_scalar_factor(diff, gxy), registry
    )
    step_strip = ProofStep(
        diff,
        stripped,
        rule="strip the generic pairing factor (x,y)",
        justification=(
            "x, y are generic sections and the pairing is "
            "non-degenerate: (x,y) is generically invertible "
            "(choose y with (x,y) = 1)"
        ),
    )

    target = Act(alg.anchor(CourantD(alg.name, f)), h)
    sign, core = _strip_sign(stripped)
    if core != target:
        raise ProofFailure(
            "Uchino 2.1(ii): expected the stripped identity to be "
            f"±ρ(Df)(h), got {stripped._repr_inner()[:140]}"
        )
    step_conclude = ProofStep(
        target,
        Integer(0),
        rule="ρ(Df)(h) = 0 for the generic scalar h ⟹ ρ∘D = 0",
        justification=(
            "the vanishing identity holds for every f and every "
            "probe h; hence the vector field ρ(Df) is zero"
        ),
    )

    chain = ProofChain(steps + [step_strip, step_conclude])
    fn = f._repr_inner()
    theorem = Theorem(
        name=f"uchino_c4_{alg.name}_{fn}",
        statement=(
            f"ρ(D{fn}) = 0 on {alg.name} "
            "(LWX [C4] is redundant — Uchino Prop 2.1(ii))"
        ),
        lhs=target,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"[C2] anchor morphism ({alg.name})",
            "Prop 2.1(i) (itself from [C5] + (L))",
            "non-degeneracy of the pairing (definitional)",
        ),
        notes=(
            "uchino.pdf Prop 2.1(ii): ρ([x, f·y])(h) expanded two "
            "ways; the defect is (x,y)·ρ(Df)(h)"
        ),
    )
    return chain, theorem


def prove_d_pairing_formula(
    alg: Algebroid,
    x: Expr,
    y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Uchino Proposition 2.2: the LWX defining relation
    ``D = ½β⁻¹ρ*d`` is redundant — any map ``D`` satisfying [C5]
    and (L) obeys

    ``(Df, y) = ½·ρ(y)(f)``.

    The two-leg tactic on ``ρ(f·x)(y, y)``: leg A applies [C5] at
    the composite site ``f·x`` (the skew swap and the cited 2.1(i)
    theorem open ``[f·x, y]``), leg B pulls the scalar out first.
    The difference carries the generic factor ``(x,y)``; stripping
    it leaves the scalar identity.
    """
    _, c3_yx = prove_left_leibniz_c3(
        alg, y, x, f, registry=registry
    )
    engine = courant_engine(alg, registry, theorems=(c3_yx,))

    start = Act(
        alg.anchor(Product(f, x)), alg.metric(y, y)
    )

    c5 = CourantInvarianceDeclaration(alg)
    if not c5.matches(start):
        raise ProofFailure(  # pragma: no cover - shape fixed above
            "[C5] does not apply to " + start._repr_inner()
        )
    after_c5 = c5.rewrite(start)
    step_c5 = ProofStep(
        start,
        after_c5,
        rule=c5.name,
        justification=(
            "declared axiom applied at the composite site f·x"
        ),
        provenance_tag="axiom",
    )
    n_a, steps_a = _normalize(after_c5, engine, registry)
    step_leg_a = ProofStep(
        after_c5, n_a,
        rule=(
            "normalize leg A (skew swap, cite Prop 2.1(i) on "
            "[y, f·x], (L) on D, bilinearity)"
        ),
        justification="engine normal form of the [C5] expansion",
    )
    for s in steps_a:
        step_leg_a.add_child(s)

    n_b, steps_b = _normalize(start, engine, registry)
    step_leg_b = ProofStep(
        start, n_b,
        rule=(
            "normalize leg B (anchor linearity pulls f out, [C5] "
            "on ρ(x)(y,y))"
        ),
        justification="engine normal form of the same expression",
    )
    for s in steps_b:
        step_leg_b.add_child(s)

    diff, diff_steps = _normalize(
        Sum(n_a, Neg(n_b)), engine, registry
    )
    step_diff = ProofStep(
        Sum(n_a, Neg(n_b)), diff,
        rule="collect: leg A − leg B",
        justification=(
            "common terms cancel; the defect carries the generic "
            "factor (x,y)"
        ),
    )
    for s in diff_steps:
        step_diff.add_child(s)
    if diff == Integer(0):
        raise ProofFailure(
            "Uchino 2.2: the two legs coincide — no identity to "
            "extract"
        )

    gxy, _ = _normalize(alg.metric(x, y), engine, registry)
    stripped = simplify(
        _strip_generic_scalar_factor(diff, gxy), registry
    )
    step_strip = ProofStep(
        diff, stripped,
        rule="strip the generic pairing factor (x,y)",
        justification=(
            "x, y generic + non-degeneracy: (x,y) is generically "
            "invertible"
        ),
    )

    # The stripped identity must be ±(c·(Df,y) − c/2·ρ(y)(f)) = 0;
    # solve it for (Df, y).
    target = alg.metric(CourantD(alg.name, f), y)
    rhs = Product(
        Rational(1, 2), Act(alg.anchor(y), f)
    )
    zero_check, _ = _normalize(
        Sum(stripped), engine, registry
    )
    coeff = None
    rest_terms: List[Expr] = []
    for term in _terms(zero_check):
        sign, core = _strip_sign(term)
        factors = (
            list(core.children)
            if isinstance(core, Product)
            else [core]
        )
        if target in factors:
            num = [
                c for c in factors
                if isinstance(c, (Integer, Rational))
            ]
            if len(factors) - len(num) == 1:
                from fractions import Fraction

                val = Fraction(1)
                for c in num:
                    val *= Fraction(c._repr_inner())
                coeff = val if sign > 0 else -val
                continue
        rest_terms.append(term)
    if coeff is None or coeff == 0:
        raise ProofFailure(
            "Uchino 2.2: the stripped identity does not contain "
            f"(Df, y): {stripped._repr_inner()[:140]}"
        )
    rest: Expr = (
        Integer(0)
        if not rest_terms
        else (
            rest_terms[0]
            if len(rest_terms) == 1
            else Sum(*rest_terms)
        )
    )
    solved_minus_rhs, _ = _normalize(
        Sum(
            Product(
                Rational(-1, int(coeff)) if coeff != 1
                else Integer(-1),
                rest,
            )
            if coeff != -1
            else rest,
            Neg(rhs),
        ),
        engine,
        registry,
    )
    if solved_minus_rhs != Integer(0):
        raise ProofFailure(
            "Uchino 2.2: solving for (Df, y) does not yield "
            "½ρ(y)(f); got residual "
            + solved_minus_rhs._repr_inner()[:140]
        )
    step_solve = ProofStep(
        target,
        rhs,
        rule="solve the stripped identity for (Df, y)",
        justification=(
            "divide by the coefficient of (Df, y); the remainder "
            "is ½ρ(y)(f) (engine-checked)"
        ),
    )

    chain = ProofChain(
        [
            step_c5,
            step_leg_a,
            step_leg_b,
            step_diff,
            step_strip,
            step_solve,
        ]
    )
    fn, yn = f._repr_inner(), y._repr_inner()
    theorem = Theorem(
        name=f"uchino_d_formula_{alg.name}_{fn}_{yn}",
        statement=(
            f"(D{fn}, {yn}) = ½ρ({yn})({fn}) on {alg.name} "
            "(the LWX defining relation D = ½β⁻¹ρ*d is redundant "
            "— Uchino Prop 2.2)"
        ),
        lhs=target,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"[C5] invariance ({alg.name})",
            "(L) Leibniz rule for D",
            "skew-symmetry of the bracket (LWX definitional)",
            "non-degeneracy of the pairing (definitional)",
        ),
        notes=(
            "uchino.pdf Prop 2.2: ρ(f·x)(y,y) expanded two ways; "
            "the defect is (x,y)·(2(Df,y) − ρ(y)f)"
        ),
    )
    return chain, theorem


def prove_d_numeric_homogeneity(
    alg: Algebroid,
    y: Expr,
    f: Expr,
    c: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """DERIVE the ℝ-homogeneity ``D(c·f) = c·D(f)`` (numeric ``c``)
    from Prop 2.2 + non-degeneracy — no extra axiom (2026-09-08
    audit, compliance finding 7):

    1. probe ``⟨D(c·f) − c·D(f), y⟩`` against the generic ``y``,
    2. unfold both pairings by the cited Prop 2.2
       (``(Dh, y) = ½ρ(y)(h)``),
    3. the vector-field action is ℝ-linear, the difference
       normalizes to literal 0,
    4. strip ``y`` by non-degeneracy.
    """
    if not isinstance(c, (Integer, Rational)):
        raise TypeError(
            "prove_d_numeric_homogeneity expects a numeric c"
        )
    D = lambda h: CourantD(alg.name, h)
    eng = ExpansionEngine(
        [DPairingUnfoldDefinition(alg)]
    )
    for d_ in algebroid_engine(
        alg, registry=registry
    ).definitions:
        eng.register(d_)

    probe = alg.metric(
        Sum(D(Product(c, f)), Neg(Product(c, D(f)))), y
    )
    nf, steps = _normalize(probe, eng, registry)
    step_probe = ProofStep(
        probe,
        nf,
        rule=(
            "unfold both pairings by the cited Prop 2.2; the "
            "vector-field action is ℝ-linear"
        ),
        justification=(
            "⟨D(c·f), y⟩ = ½ρ(y)(c·f) = c·½ρ(y)(f) = "
            "c·⟨D(f), y⟩"
        ),
        provenance_tag="theorem",
    )
    for s in steps:
        step_probe.add_child(s)
    if nf != Integer(0):
        raise ProofFailure(
            "D-homogeneity derivation: the probe does not vanish "
            "— residual " + nf._repr_inner()[:140]
        )
    step_nondeg = ProofStep(
        probe,
        Integer(0),
        rule="non-degeneracy of ( , ) (generic probe section y)",
        justification=(
            "the probe vanishes for generic y, hence "
            "D(c·f) = c·D(f)"
        ),
    )
    chain = ProofChain([step_probe, step_nondeg])
    cn, fn = c._repr_inner(), f._repr_inner()
    theorem = Theorem(
        name=f"d_homogeneity_{alg.name}_{cn}_{fn}",
        statement=(
            f"D({cn}·{fn}) = {cn}·D({fn}) on {alg.name} "
            "(ℝ-homogeneity of D — DERIVED from Prop 2.2 + "
            "non-degeneracy; no extra axiom)"
        ),
        lhs=D(Product(c, f)),
        rhs=Product(c, D(f)),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "Prop 2.2 (itself from [C5] + (L) + skew + "
            "non-degeneracy)",
            "ℝ-linearity of the vector-field action "
            "(definitional)",
            "non-degeneracy of the pairing (definitional)",
        ),
        notes=(
            "2026-09-08 audit, compliance finding 7: the "
            "homogeneity previously gated behind a declaration is "
            "derivable — ⟨D(cf) − cDf, y⟩ = 0 for generic y"
        ),
    )
    return chain, theorem


def prove_leibniz_from_c3(
    alg: Algebroid,
    x: Expr,
    y: Expr,
    f: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Uchino Remark 2.1 — the CONVERSE of Prop 2.1(i): a map ``D``
    satisfying the identity (i) automatically obeys the Leibniz rule

    ``D(f·g) = f·D(g) + g·D(f)``.

    Mechanization: the identity (i) is assumed as a rewrite and the
    same start ``[x, f·g·y]`` is normalized under its two sound
    application orders — peel the scalars ONE at a time
    (``[x, f·(g·y)]``) versus peel the full prefix AT ONCE
    (``[x, (f·g)·y]``). Every bracket/action term cancels in the
    difference; what survives is ``(x,y)·(D(fg) − fD(g) − gD(f))``,
    and the generic pairing factor strips. NO (L) rule anywhere in
    either engine — ``D`` is fully inert here.
    """
    base = algebroid_engine(alg, registry=registry).definitions
    eng_one = ExpansionEngine(
        [C3IdentityDeclaration(alg, registry, peel="one")]
    )
    eng_all = ExpansionEngine(
        [C3IdentityDeclaration(alg, registry, peel="all")]
    )
    for d in base:
        eng_one.register(d)
        eng_all.register(d)

    start = alg.bracket(x, Product(f, g, y))
    diff, steps = _two_leg_difference(
        start, eng_one, eng_all, registry,
        label_a=(
            "normalize leg A (identity (i) peeling one scalar at a "
            "time: [x, f·(g·y)])"
        ),
        label_b=(
            "normalize leg B (identity (i) peeling the full scalar "
            "prefix: [x, (f·g)·y])"
        ),
    )
    if diff == Integer(0):
        raise ProofFailure(
            "Rem 2.1: the two peel orders coincide — no Leibniz "
            "defect to extract"
        )

    gxy, _ = _normalize(alg.metric(x, y), eng_all, registry)
    stripped = simplify(
        _strip_generic_scalar_factor(diff, gxy), registry
    )
    step_strip = ProofStep(
        diff, stripped,
        rule="strip the generic pairing factor (x,y)",
        justification=(
            "x, y generic + non-degeneracy: (x,y) is generically "
            "invertible"
        ),
    )

    target = CourantD(alg.name, Product(f, g))
    rhs = Sum(
        Product(f, CourantD(alg.name, g)),
        Product(g, CourantD(alg.name, f)),
    )
    check, _ = _normalize(
        Sum(stripped, Neg(Sum(target, Neg(rhs)))),
        eng_all,
        registry,
    )
    sign_flip = False
    if check != Integer(0):
        check, _ = _normalize(
            Sum(stripped, Sum(target, Neg(rhs))),
            eng_all,
            registry,
        )
        sign_flip = True
    if check != Integer(0):
        raise ProofFailure(
            "Rem 2.1: the stripped defect is not the Leibniz "
            f"defect of D: {stripped._repr_inner()[:140]}"
        )
    step_solve = ProofStep(
        target,
        rhs,
        rule="the stripped identity IS the Leibniz defect = 0",
        justification=(
            "±(D(f·g) − f·D(g) − g·D(f)) = 0 "
            f"(sign {'−' if sign_flip else '+'}; engine-checked)"
        ),
    )

    chain = ProofChain(steps + [step_strip, step_solve])
    fn, gn = f._repr_inner(), g._repr_inner()
    theorem = Theorem(
        name=f"uchino_rem21_{alg.name}_{fn}_{gn}",
        statement=(
            f"D({fn}·{gn}) = {fn}·D({gn}) + {gn}·D({fn}) on "
            f"{alg.name} (the Leibniz rule (L) FOLLOWS from the "
            "identity (i) alone — Uchino Rem 2.1, the converse of "
            "Prop 2.1(i))"
        ),
        lhs=target,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "the identity (i) (assumed as hypothesis)",
            "non-degeneracy of the pairing (definitional)",
        ),
        notes=(
            "uchino.pdf Rem 2.1: [x, f·g·y] peeled two ways; no "
            "(L) rule is registered in either engine"
        ),
    )
    return chain, theorem


def prove_bracket_with_d(
    alg: Algebroid,
    x: Expr,
    y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    classical: bool = False,
) -> Tuple[ProofChain, Theorem]:
    """Uchino Rem 2.2, mechanized precisely: under the full Courant
    axioms ([C2] + [C5] + (L)),

    ``[x, Df] + D((x, Df)) = D(ρ(x)(f))``.

    This is the raw pre-collapse form. The classical Courant
    property ``[x, Df] = D((x, Df))`` [Roytenberg] follows by the
    ℝ-homogeneity ``D(c·h) = c·D(h)`` — which is itself DERIVED
    from Prop 2.2 + non-degeneracy
    (:func:`prove_d_numeric_homogeneity`; 2026-09-08 audit,
    compliance finding 7 — an earlier revision wrongly gated it
    behind a declaration). Pass ``classical=True`` for that
    corollary; both forms need no axiom beyond the LWX data.
    The paper's identity (A) is stated loosely; this derivation
    replaces it:

    1. two-NF on ``ρ(x)((Df, y))``: leg A = [C5] + the [C4]/DD-zero
       theorem; leg B = unfold (Df,y) by the cited Prop 2.2 — their
       difference I1 is a proven zero,
    2. cite Prop 2.2 twice more, at the COMPOSITE section ``[x,y]``
       and at the DERIVED function ``ρ(x)(f)``,
    3. the probe of the target against generic ``y`` reduces, via
       these three zeros + [C2], to literal 0,
    4. strip ``y`` (non-degeneracy) and read off the section
       identity.
    """
    D = lambda h: CourantD(alg.name, h)

    # Cited Prop 2.2 instances: at the composite section [x,y] and
    # at the derived scalar ρ(x)(f). Each is re-PROVEN at its
    # instance (the tactic is generic in both slots). The instance
    # proofs need no [C2], and at a composite section the declared
    # anchor-morphism would derail the probe (ρ([x,y]) opens into
    # Lie brackets) — so they run on the [C2]-free twin context
    # (same name and bundle: identical atoms).
    alg_no_c2 = Algebroid(
        alg.name,
        alg.bundle,
        anchor_name=alg.anchor_name,
        declare=tuple(
            d for d in alg.declarations if d != "anchor-morphism"
        ),
    )
    aux = _fresh_section(alg, x, y, f)
    prove_d_pairing_formula(
        alg_no_c2,
        aux,
        alg.bracket(x, y),
        f,
        registry=registry,
    )
    prove_d_pairing_formula(
        alg_no_c2,
        aux,
        y,
        Act(alg.anchor(x), f),
        registry=registry,
    )

    eng_a = courant_engine(alg, registry)
    eng_a.register(DDZeroDefinition(alg))
    eng_b = ExpansionEngine(
        [DPairingUnfoldDefinition(alg), DDZeroDefinition(alg)]
    )
    for d_ in algebroid_engine(
        alg, registry=registry
    ).definitions:
        eng_b.register(d_)

    start = Act(alg.anchor(x), alg.metric(D(f), y))
    diff, steps = _two_leg_difference(
        start, eng_a, eng_b, registry,
        label_a=(
            "normalize leg A ([C5] on ρ(x)(Df, y); (Da,Db) = 0 "
            "by the [C4] theorem)"
        ),
        label_b=(
            "normalize leg B (cite Prop 2.2: (Df,y) = ½ρ(y)f)"
        ),
        diff_engine=eng_a,
    )

    # Assemble the probe of the target identity against y and
    # reduce it with the proven zeros.
    target_section = Sum(
        alg.bracket(x, D(f)),
        D(alg.metric(x, D(f))),
        Neg(D(Act(alg.anchor(x), f))),
    )
    probe = alg.metric(target_section, y)
    zeros = [
        ("cite I1 (the two-NF difference above)", diff),
        (
            "cite Prop 2.2 at the composite section [x,y]",
            Sum(
                alg.metric(D(f), alg.bracket(x, y)),
                Neg(
                    Product(
                        Rational(1, 2),
                        Act(
                            alg.anchor(alg.bracket(x, y)), f
                        ),
                    )
                ),
            ),
        ),
        (
            "cite Prop 2.2 at the derived scalar ρ(x)(f)",
            Sum(
                alg.metric(D(Act(alg.anchor(x), f)), y),
                Neg(
                    Product(
                        Rational(1, 2),
                        Act(
                            alg.anchor(y),
                            Act(alg.anchor(x), f),
                        ),
                    )
                ),
            ),
        ),
    ]
    # A signed combination of the three cited zeros must reduce
    # the probe to literal 0 — a zero remains a zero under either
    # sign, so trying both signs per cite is sound; the engine
    # verifies the winning combination end-to-end.
    residual0, _ = _normalize(probe, eng_a, registry)
    red_steps: List[ProofStep] = []
    winning = None
    from itertools import product as _iproduct

    for signs in _iproduct((1, -1), repeat=len(zeros)):
        combo = Sum(
            residual0,
            *(
                (zero if s > 0 else Neg(zero))
                for s, (_lbl, zero) in zip(signs, zeros)
            ),
        )
        check, _ = _normalize(combo, eng_a, registry)
        if check == Integer(0):
            winning = signs
            break
    if winning is None:
        raise ProofFailure(
            "Rem 2.2: the probe does not reduce to 0 — residual "
            + residual0._repr_inner()[:160]
        )
    residual = residual0
    for s, (label, zero) in zip(winning, zeros):
        cand, csteps = _normalize(
            Sum(residual, zero if s > 0 else Neg(zero)),
            eng_a,
            registry,
        )
        st = ProofStep(
            residual, cand,
            rule=label + (" (+)" if s > 0 else " (−)"),
            justification=(
                "adding a proven zero (theorem-classified)"
            ),
            provenance_tag="theorem",
        )
        for cs in csteps:
            st.add_child(cs)
        red_steps.append(st)
        residual = cand
    if residual != Integer(0):
        raise ProofFailure(  # pragma: no cover - combo pre-checked
            "Rem 2.2: staged reduction disagrees with the "
            "verified combination"
        )

    step_nondeg = ProofStep(
        probe,
        Integer(0),
        rule="non-degeneracy of ( , ) (generic probe section y)",
        justification=(
            "the probe vanishes for generic y, hence the section "
            "identity holds"
        ),
    )

    lhs = Sum(
        alg.bracket(x, D(f)), D(alg.metric(x, D(f)))
    )
    rhs: Expr = D(Act(alg.anchor(x), f))
    chain_steps = (
        steps + red_steps + [step_nondeg]
    )

    if classical:
        # ℝ-homogeneity: (x,Df) = ½ρ(x)f collapses the identity to
        # the classical [x,Df] = D(x,Df). The homogeneity is DERIVED
        # first (Prop 2.2 + non-degeneracy — 2026-09-08 audit,
        # compliance finding 7), never declared.
        hom_chain, hom_thm = prove_d_numeric_homogeneity(
            alg, y, f, Rational(1, 2), registry=registry
        )
        hom = DNumericHomogeneityDeclaration(alg)
        eng_c = courant_engine(alg, registry)
        eng_c.register(hom)
        eng_c.register(DPairingUnfoldDefinition(alg))
        lhs_c = alg.bracket(x, D(f))
        rhs_c = D(alg.metric(x, D(f)))
        check, csteps = _normalize(
            Sum(lhs_c, Neg(rhs_c), Neg(Sum(lhs, Neg(rhs)))),
            eng_c,
            registry,
        )
        if check != Integer(0):
            raise ProofFailure(
                "Rem 2.2 classical: homogeneity did not collapse "
                "the identity — residual "
                + check._repr_inner()[:140]
            )
        st = ProofStep(
            lhs_c, rhs_c,
            rule=(
                "derived ℝ-homogeneity of D collapses the "
                "identity: D(ρ(x)f) = 2·D((x,Df))"
            ),
            justification=(
                "(x,Df) = ½ρ(x)f (Prop 2.2) + D(c·h) = c·D(h) "
                f"(cited theorem: {hom_thm.name}); engine-checked"
            ),
            provenance_tag="theorem",
        )
        for s in hom_chain:
            st.add_child(s)
        for s in csteps:
            st.add_child(s)
        chain_steps = chain_steps + [st]
        lhs, rhs = lhs_c, rhs_c

    chain = ProofChain(chain_steps)
    xn, fn = x._repr_inner(), f._repr_inner()
    theorem = Theorem(
        name=(
            f"uchino_rem22_{'classical_' if classical else ''}"
            f"{alg.name}_{xn}_{fn}"
        ),
        statement=(
            (
                f"[{xn}, D{fn}] = D(({xn}, D{fn})) on {alg.name} "
                "(classical Courant property — Uchino Rem 2.2, "
                "mechanized; the ℝ-homogeneity of D is DERIVED "
                "from Prop 2.2 + non-degeneracy)"
            )
            if classical
            else (
                f"[{xn}, D{fn}] + D(({xn}, D{fn})) = "
                f"D(ρ({xn})({fn})) on {alg.name} "
                "(the raw pre-collapse form — Uchino Rem 2.2, "
                "mechanized; the paper's (A) is loose)"
            )
        ),
        lhs=lhs,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"[C2] anchor morphism ({alg.name})",
            f"[C5] invariance ({alg.name})",
            "(L) Leibniz rule for D",
            "Prop 2.2 / [C4] theorems (cited)",
            "non-degeneracy of the pairing (definitional)",
        )
        + (
            (
                "ℝ-homogeneity of D (DERIVED — Prop 2.2 + "
                "non-degeneracy)",
            )
            if classical
            else ()
        ),
        notes=(
            "2026-09-08 audit, compliance finding 7: the "
            "homogeneity D(c·h) = c·D(h) follows from Prop 2.2 + "
            "non-degeneracy under the same assumptions — both "
            "forms of Rem 2.2 close with NO extra axiom."
        ),
    )
    return chain, theorem
