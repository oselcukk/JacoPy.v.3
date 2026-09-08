"""
The (almost-/metric-)Bourbaki STRUCTURE layer (Phase 7.B.2f.2), from
the PRIMARY paper [arXiv:2210.00548] §6-7 — the R-VALUED metric
side of the hierarchy, mechanized without ever collapsing R to
functions (the audit's 10h/10j.iii-iv scope point):

* an R-valued E-metric ``g : 𝔛(E) × 𝔛(E) → 𝔛(R)`` (symmetric,
  C∞-bilinear, non-degenerate by declaration of the context);
* ``𝔻 : 𝔛(R) → 𝔛(E)`` first-order with symbol ``𝕃``
  (``𝔻(f·r) = f·𝔻r + 𝕃_{df} r``, eq (6.8));
* ``ℒ^R_{ρ(u)} : 𝔛(R) → 𝔛(R)`` a DERIVATION with symbol the anchor
  (``ℒ^R_{ρu}(f·r) = f·ℒ^R_{ρu}r + ρ(u)(f)·r``, eq (7.3)).

Theorems mechanized here:

* **Corollary 6.1** — every almost-Bourbaki algebroid is local
  almost-Leibniz with locality operator ``L(df,u,v) =
  𝕃_{df}(g(u,v))``: the C∞-defect of ``S = 𝔻g`` is exactly the
  symbol term (pure ``𝔻``-first-order + ``g``-bilinearity);
* **(7.2)** — the R-VALUED generalization of the B 4.11 tactic:
  metric invariance (7.8) + the derivation law (7.3) + the
  non-degeneracy of ``g`` imply the RIGHT-LEIBNIZ rule
  ``[u, f·v] = f[u,v] + ρ(u)(f)·v`` (defect-free);
* **left-Leibniz corollary** — from the declared symmetric part
  ``[u,v] + [v,u] = 𝔻g(u,v)`` (6.11) + the proven right-Leibniz +
  Cor 6.1: ``[f·u, v] = f[u,v] − ρ(v)(f)·u + 𝕃_{df}(g(u,v))``
  (the ℚ-linear citation of four proven/declared zeros,
  engine-verified).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Product,
    Sum,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.algebroid.context import Algebroid, algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.theorems import _fresh_section
from jacopy.algorithms.simplify import simplify
from jacopy.packages.generalized.bourbaki_precalculus import (
    _SlotAtom,
    _split_scalar,
)


# ------------------------------------------------------------------ #
# Nodes                                                              #
# ------------------------------------------------------------------ #


class BMetric(_SlotAtom):
    """``g(u, v) ∈ 𝔛(R)`` — the R-VALUED E-metric (Def 6.2): NOT a
    scalar; symmetric and C∞-bilinear; non-degenerate by the
    context's declaration (licensing the generic-probe strip)."""

    def __init__(self, u: Expr, v: Expr) -> None:
        super().__init__(
            f"g({u._repr_inner()},{v._repr_inner()})", u, v
        )

    @property
    def u(self) -> Expr:
        return self._slots[0]

    @property
    def v(self) -> Expr:
        return self._slots[1]

    def with_slots(self, u: Expr, v: Expr) -> "BMetric":
        return BMetric(u, v)


class BDop(_SlotAtom):
    """``𝔻r ∈ 𝔛(E)`` — first-order with symbol ``𝕃`` (6.8)."""

    def __init__(self, r: Expr) -> None:
        super().__init__(f"𝔻({r._repr_inner()})", r)

    @property
    def r(self) -> Expr:
        return self._slots[0]

    def with_slots(self, r: Expr) -> "BDop":
        return BDop(r)


class BSymbolL(_SlotAtom):
    """``𝕃_{df} r ∈ 𝔛(E)`` — the symbol leg of ``𝔻`` (recorded with
    the generating scalar ``f``); C∞-bilinear in ``r``."""

    def __init__(self, f: Expr, r: Expr) -> None:
        super().__init__(
            f"𝕃_d{f._repr_inner()}({r._repr_inner()})", f, r
        )

    @property
    def f(self) -> Expr:
        return self._slots[0]

    @property
    def r(self) -> Expr:
        return self._slots[1]

    def with_slots(self, f: Expr, r: Expr) -> "BSymbolL":
        return BSymbolL(f, r)


class BLieRE(_SlotAtom):
    """``ℒ^R_{ρ(u)} r ∈ 𝔛(R)`` — the metric-invariance operator of
    Def 7.1, a derivation on R with symbol the anchor (7.3)."""

    def __init__(self, u: Expr, r: Expr) -> None:
        super().__init__(
            f"L^R_ρ({u._repr_inner()})({r._repr_inner()})",
            u,
            r,
        )

    @property
    def u(self) -> Expr:
        return self._slots[0]

    @property
    def r(self) -> Expr:
        return self._slots[1]

    def with_slots(self, u: Expr, r: Expr) -> "BLieRE":
        return BLieRE(u, r)


# ------------------------------------------------------------------ #
# Definitional rules                                                 #
# ------------------------------------------------------------------ #


class BMetricBilinearityDefinition(Definition):
    """C∞-bilinearity of the R-valued metric (both slots)."""

    name = "R-valued metric bilinearity: g(fu+w, v) = f·g(u,v) + g(w,v)"
    anchor = BMetric

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BMetric):
            return False
        return any(
            isinstance(s, (Sum, Neg))
            or s == Integer(0)
            or _split_scalar(s, self._registry) is not None
            for s in expr.rewritable_slots
        )

    def rewrite(self, expr: Expr) -> Expr:
        for i, s in enumerate(expr.rewritable_slots):
            def rebuilt(new_s: Expr) -> Expr:
                slots = list(expr.rewritable_slots)
                slots[i] = new_s
                return expr.with_slots(*slots)

            if s == Integer(0):
                return Integer(0)
            if isinstance(s, Sum):
                return Sum(*(rebuilt(c) for c in s.children))
            if isinstance(s, Neg):
                return Neg(rebuilt(s.arg))
            split = _split_scalar(s, self._registry)
            if split:
                f, rest = split
                return Product(f, rebuilt(rest))
        raise AssertionError  # pragma: no cover


class BMetricSymmetryDefinition(Definition):
    """``g(v, u) → g(u, v)`` (canonical slot order by repr)."""

    name = "R-valued metric symmetry: g(v,u) = g(u,v)"
    anchor = BMetric

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BMetric):
            return False
        u, v = expr.u, expr.v
        if isinstance(u, (Sum, Neg, Product)) or isinstance(
            v, (Sum, Neg, Product)
        ):
            return False
        return u._repr_inner() > v._repr_inner()

    def rewrite(self, expr: Expr) -> Expr:
        return BMetric(expr.v, expr.u)


class BDopFirstOrderDefinition(Definition):
    """``𝔻(f·r) = f·𝔻r + 𝕃_{df} r`` (+ ℝ-additivity) — eq (6.8)."""

    name = "𝔻 first-order with symbol 𝕃: 𝔻(fr) = f𝔻r + 𝕃_df r"
    anchor = BDop

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BDop):
            return False
        r = expr.r
        return (
            isinstance(r, (Sum, Neg))
            or r == Integer(0)
            or _split_scalar(r, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        r = expr.r
        if r == Integer(0):
            return Integer(0)
        if isinstance(r, Sum):
            return Sum(*(BDop(c) for c in r.children))
        if isinstance(r, Neg):
            return Neg(BDop(r.arg))
        f, rest = _split_scalar(r, self._registry)
        return Sum(Product(f, BDop(rest)), BSymbolL(f, rest))


class BSymbolLLinearityDefinition(Definition):
    """C∞-linearity of ``𝕃_{df}`` in its R-slot."""

    name = "𝕃 C∞-linearity: 𝕃_df(fr + s) = f·𝕃_df r + 𝕃_df s"
    anchor = BSymbolL

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BSymbolL):
            return False
        r = expr.r
        return (
            isinstance(r, (Sum, Neg))
            or r == Integer(0)
            or _split_scalar(r, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        r = expr.r
        if r == Integer(0):
            return Integer(0)
        if isinstance(r, Sum):
            return Sum(
                *(BSymbolL(expr.f, c) for c in r.children)
            )
        if isinstance(r, Neg):
            return Neg(BSymbolL(expr.f, r.arg))
        h, rest = _split_scalar(r, self._registry)
        return Product(h, BSymbolL(expr.f, rest))


class BLieREDerivationDefinition(Definition):
    """The derivation law (7.3) with anchor symbol:
    ``ℒ^R_{ρu}(f·r) = f·ℒ^R_{ρu} r + ρ(u)(f)·r`` (+ ℝ-additivity)."""

    name = (
        "ℒ^R derivation with anchor symbol (7.3): "
        "ℒ^R_ρu(fr) = fℒ^R_ρu r + ρ(u)(f)·r"
    )
    anchor = BLieRE

    def __init__(
        self,
        alg: Algebroid,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._alg = alg
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BLieRE):
            return False
        r = expr.r
        return (
            isinstance(r, (Sum, Neg))
            or r == Integer(0)
            or _split_scalar(r, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        r = expr.r
        if r == Integer(0):
            return Integer(0)
        if isinstance(r, Sum):
            return Sum(
                *(BLieRE(expr.u, c) for c in r.children)
            )
        if isinstance(r, Neg):
            return Neg(BLieRE(expr.u, r.arg))
        f, rest = _split_scalar(r, self._registry)
        return Sum(
            Product(f, BLieRE(expr.u, rest)),
            Product(
                Act(self._alg.anchor(expr.u), f), rest
            ),
        )


class BourbakiInvarianceDeclaration(Definition):
    """DECLARED metric invariance (7.8):
    ``ℒ^R_{ρu}(g(v, w)) → g([u,v], w) + g(v, [u,w])`` — terminating
    (the ℒ^R node disappears)."""

    anchor = BLieRE

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"(7.8) invariance ({alg.name}): "
            "ℒ^R_ρu g(v,w) = g([u,v],w) + g(v,[u,w])"
        )

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, BLieRE) and isinstance(
            expr.r, BMetric
        )

    def rewrite(self, expr: Expr) -> Expr:
        u = expr.u
        v, w = expr.r.u, expr.r.v
        br = self._alg.bracket
        return Sum(
            BMetric(br(u, v), w), BMetric(v, br(u, w))
        )


# ------------------------------------------------------------------ #
# Context + engine                                                   #
# ------------------------------------------------------------------ #


def bourbaki_structure_context(
    name: str = "E",
) -> Algebroid:
    """The almost-metric-Bourbaki DATA (Def 7.1) as a context: an
    anchored E with opaque ℝ-bilinear bracket; the R-valued g, 𝔻,
    𝕃, ℒ^R live in the nodes above. Nothing derivable declared."""
    return algebroid(name)


def bourbaki_structure_engine(
    alg: Algebroid,
    registry: Optional[PropertyRegistry] = None,
    *,
    invariance: bool = True,
) -> ExpansionEngine:
    rules: List[Definition] = []
    if invariance:
        rules.append(BourbakiInvarianceDeclaration(alg))
    rules.extend(
        [
            BMetricBilinearityDefinition(registry),
            BMetricSymmetryDefinition(),
            BDopFirstOrderDefinition(registry),
            BSymbolLLinearityDefinition(registry),
            BLieREDerivationDefinition(alg, registry),
        ]
    )
    eng = ExpansionEngine(rules)
    for d_ in algebroid_engine(
        alg, registry=registry
    ).definitions:
        eng.register(d_)
    return eng


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


# ------------------------------------------------------------------ #
# Corollary 6.1                                                      #
# ------------------------------------------------------------------ #


def prove_locality_is_symbol(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Corollary 6.1 — the C∞-defect of ``S = 𝔻g`` is the SYMBOL:

    ``𝔻g(f·u, v) − f·𝔻g(u,v) = 𝕃_{df}(g(u,v))``

    (pure ``g``-bilinearity + ``𝔻``-first-order; consequently every
    almost-Bourbaki algebroid is local almost-Leibniz with
    ``L(df,u,v) = 𝕃_{df}(g(u,v))``)."""
    engine = bourbaki_structure_engine(
        alg, registry, invariance=False
    )
    diff = Sum(
        BDop(BMetric(Product(f, u), v)),
        Neg(Product(f, BDop(BMetric(u, v)))),
        Neg(BSymbolL(f, BMetric(u, v))),
    )
    nf = _normalize(engine, diff, registry)
    if nf != Integer(0):
        raise ProofFailure(
            "Cor 6.1: the symbol defect does not vanish — "
            + nf._repr_inner()[:140]
        )
    chain = ProofChain(
        [
            ProofStep(
                diff,
                Integer(0),
                rule=(
                    "g-bilinearity + 𝔻-first-order collapse "
                    "the C∞ defect onto the symbol"
                ),
                justification="engine normal form",
            )
        ]
    )
    theorem = Theorem(
        name=f"bourbaki_cor61_{alg.name}",
        statement=(
            "𝔻g(f·u, v) = f·𝔻g(u,v) + 𝕃_df(g(u,v)) on "
            f"{alg.name} (Cor 6.1: every almost-Bourbaki "
            "algebroid is local almost-Leibniz with "
            "L(df,u,v) = 𝕃_df(g(u,v)))"
        ),
        lhs=diff,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "R-valued metric C∞-bilinearity (definitional)",
            "𝔻 first-order with symbol 𝕃 (6.8)",
        ),
        notes="pre-metric-bourbaki.pdf Cor 6.1",
    )
    return chain, theorem


# ------------------------------------------------------------------ #
# (7.2) — right-Leibniz from R-valued metric invariance             #
# ------------------------------------------------------------------ #


def _bmetric_coefficient_of(
    expr: Expr, w: Expr
) -> Tuple[Expr, ...]:
    """Split ``Σ ±cᵢ·g(xᵢ, w)`` into signed contributions ``±cᵢxᵢ``
    (R-valued bilinearity read backwards); raise when a term carries
    no g-factor against the probe ``w``."""
    from jacopy.central.algebroid.theorems import (
        _strip_sign,
        _terms,
    )

    out: List[Expr] = []
    for term in _terms(expr):
        sign, core = _strip_sign(term)
        factors = (
            list(core.children)
            if isinstance(core, Product)
            else [core]
        )
        hits = [
            (k, c)
            for k, c in enumerate(factors)
            if isinstance(c, BMetric)
            and (c.u == w or c.v == w)
        ]
        if len(hits) != 1:
            raise ProofFailure(
                "R-valued strip: the term "
                f"{term._repr_inner()} carries no g(·, w) factor"
            )
        k, gnode = hits[0]
        other = gnode.u if gnode.v == w else gnode.v
        cofactors = [
            c for m, c in enumerate(factors) if m != k
        ]
        piece = (
            other
            if not cofactors
            else Product(*cofactors, other)
        )
        out.append(Neg(piece) if sign < 0 else piece)
    return tuple(out)


def prove_right_leibniz_from_r_invariance(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    w: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """(7.2) — the R-VALUED generalization of B 4.11: the declared
    invariance (7.8) + the derivation law (7.3) + non-degeneracy of
    ``g`` imply the defect-free RIGHT-LEIBNIZ rule

    ``[u, f·v] = f·[u,v] + ρ(u)(f)·v``

    via the two-leg tactic on ``ℒ^R_{ρu}(g(f·v, w))``."""
    if w is None:
        w = _fresh_section(alg, u, v, f)
    engine = bourbaki_structure_engine(alg, registry)
    inv = BourbakiInvarianceDeclaration(alg)

    start = BLieRE(u, BMetric(Product(f, v), w))
    after = inv.rewrite(start)
    step_inv = ProofStep(
        start,
        after,
        rule=inv.name,
        justification="declared axiom at the composite site",
        provenance_tag="axiom",
    )
    from jacopy.central.algebroid.theorems import _normalize as _n

    n_a, steps_a = _n(after, engine, registry)
    step_a = ProofStep(
        after, n_a,
        rule="normalize leg A (R-valued bilinearity)",
        justification="engine normal form",
    )
    for s in steps_a:
        step_a.add_child(s)
    n_b, steps_b = _n(start, engine, registry)
    step_b = ProofStep(
        start, n_b,
        rule=(
            "normalize leg B (bilinearity pulls f, the "
            "derivation law (7.3), then (7.8))"
        ),
        justification="engine normal form of the same start",
    )
    for s in steps_b:
        step_b.add_child(s)
    diff, dsteps = _n(Sum(n_a, Neg(n_b)), engine, registry)
    step_d = ProofStep(
        Sum(n_a, Neg(n_b)), diff,
        rule="collect: leg A − leg B = g(X, w)",
        justification="common terms cancel",
    )
    for s in dsteps:
        step_d.add_child(s)

    contributions = _bmetric_coefficient_of(diff, w)
    X = simplify(
        contributions[0]
        if len(contributions) == 1
        else Sum(*contributions),
        registry,
    )
    step_nondeg = ProofStep(
        diff, Integer(0),
        rule=(
            "non-degeneracy of the R-VALUED metric (generic "
            "probe w)"
        ),
        justification="g(X, w) = 0 for generic w ⟹ X = 0",
    )
    target = alg.bracket(u, Product(f, v))
    rhs = Sum(
        Product(f, alg.bracket(u, v)),
        Product(Act(alg.anchor(u), f), v),
    )
    check = simplify(
        Sum(X, Neg(Sum(target, Neg(rhs)))), registry
    )
    sign_note = "+"
    if check != Integer(0):
        check = simplify(
            Sum(X, Sum(target, Neg(rhs))), registry
        )
        sign_note = "−"
    if check != Integer(0):
        raise ProofFailure(
            "(7.2): the stripped identity is not the "
            "right-Leibniz defect — X = "
            + X._repr_inner()[:140]
        )
    step_solve = ProofStep(
        target, rhs,
        rule="the vanishing section identity IS right-Leibniz",
        justification=(
            f"X = {sign_note}([u,fv] − f[u,v] − ρ(u)f·v) "
            "(engine-checked)"
        ),
    )
    chain = ProofChain(
        [step_inv, step_a, step_b, step_d, step_nondeg, step_solve]
    )
    theorem = Theorem(
        name=f"bourbaki_72_right_leibniz_{alg.name}",
        statement=(
            "[u, f·v] = f·[u,v] + ρ(u)(f)·v on "
            f"{alg.name} ((7.2): R-valued metric invariance + "
            "anchor-symbol derivation + non-degeneracy imply "
            "right-Leibniz — the generalized B 4.11)"
        ),
        lhs=target,
        rhs=rhs,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"(7.8) invariance ({alg.name}, declared)",
            "(7.3) derivation law with anchor symbol",
            "non-degeneracy of the R-valued metric "
            "(definitional)",
        ),
        notes="pre-metric-bourbaki.pdf eq (7.2)",
    )
    return chain, theorem


# ------------------------------------------------------------------ #
# Left-Leibniz corollary                                             #
# ------------------------------------------------------------------ #


def prove_left_leibniz_bourbaki(
    alg: Algebroid,
    u: Expr,
    v: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The left-Leibniz rule of the almost-Bourbaki hierarchy:

    ``[f·u, v] = f·[u,v] − ρ(v)(f)·u + 𝕃_{df}(g(u,v))``

    from FOUR proven/declared zeros — the declared symmetric part
    (6.11) at ``(fu,v)`` and at ``(u,v)``, the proven right-Leibniz
    (7.2), and the Cor 6.1 symbol identity — combined by the exact
    ℚ-linear citation phase and engine-verified."""
    from fractions import Fraction

    from jacopy.core.expr import Rational
    from jacopy.packages.poisson.koszul_jacobi import (
        _gauss_solve,
        _to_vec,
    )

    engine = bourbaki_structure_engine(alg, registry)
    br = alg.bracket
    target = Sum(
        br(Product(f, u), v),
        Neg(Product(f, br(u, v))),
        Product(Act(alg.anchor(v), f), u),
        Neg(BSymbolL(f, BMetric(u, v))),
    )
    zeros = [
        (
            "(6.11) symmetric part at (f·u, v)",
            Sum(
                br(Product(f, u), v),
                br(v, Product(f, u)),
                Neg(BDop(BMetric(Product(f, u), v))),
            ),
        ),
        (
            "(6.11) symmetric part at (u, v), scaled by f",
            Sum(
                Product(f, br(u, v)),
                Product(f, br(v, u)),
                Neg(Product(f, BDop(BMetric(u, v)))),
            ),
        ),
        (
            "(7.2) right-Leibniz at [v, f·u]",
            Sum(
                br(v, Product(f, u)),
                Neg(Product(f, br(v, u))),
                Neg(Product(Act(alg.anchor(v), f), u)),
            ),
        ),
        (
            "Cor 6.1 symbol identity",
            Sum(
                BDop(BMetric(Product(f, u), v)),
                Neg(Product(f, BDop(BMetric(u, v)))),
                Neg(BSymbolL(f, BMetric(u, v))),
            ),
        ),
    ]
    t_nf = _normalize(engine, target, registry)
    rows, labels, cites = [], [], []
    for label, z in zeros:
        nf = _normalize(engine, z, registry)
        # each cite must itself be certified: symmetric-part
        # instances are DECLARED; (7.2) and Cor 6.1 are proven
        # above, and their raw forms normalize to shapes the
        # engine can subtract.
        vec = _to_vec(nf)
        if not vec:
            continue
        rows.append(vec)
        labels.append(label)
        cites.append(nf)
    tv = _to_vec(t_nf)
    keys = sorted(set(tv) | {k for r in rows for k in r})
    coeffs = _gauss_solve(rows, tv, keys)
    if coeffs is None:
        raise ProofFailure(
            "left-Leibniz: no rational combination of the four "
            "cited zeros matches the target — residual "
            + t_nf._repr_inner()[:140]
        )

    def times(c: Fraction, e: Expr) -> Expr:
        if c == 1:
            return e
        if c == -1:
            return Neg(e)
        n_, d_ = c.numerator, c.denominator
        base = e if n_ > 0 else Neg(e)
        ce = (
            Integer(abs(n_))
            if d_ == 1
            else Rational(abs(n_), d_)
        )
        return Product(ce, base)

    nz = [(i, c) for i, c in enumerate(coeffs) if c != 0]
    combo = Sum(
        t_nf, *(times(-c, cites[i]) for i, c in nz)
    )
    check = _normalize(engine, combo, registry)
    if check != Integer(0):
        raise ProofFailure(
            "left-Leibniz: the ℚ-combination fails engine "
            "verification — " + check._repr_inner()[:140]
        )
    steps = [
        ProofStep(
            target,
            Integer(0),
            rule=(
                "cite ℚ-linear combination of "
                f"{len(nz)} proven/declared zeros "
                "(engine-verified): "
                + "; ".join(labels[i] for i, _ in nz)
            ),
            justification=(
                "declared (6.11) instances + proven (7.2) + "
                "proven Cor 6.1; the combination subtracts to "
                "literal 0"
            ),
            provenance_tag="theorem",
        )
    ]
    chain = ProofChain(steps)
    theorem = Theorem(
        name=f"bourbaki_left_leibniz_{alg.name}",
        statement=(
            "[f·u, v] = f·[u,v] − ρ(v)(f)·u + 𝕃_df(g(u,v)) on "
            f"{alg.name} (left-Leibniz of the almost-Bourbaki "
            "hierarchy — the locality operator is the symbol, "
            "Cor 6.1)"
        ),
        lhs=target,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "(6.11) symmetric part [u,v]+[v,u] = 𝔻g(u,v) "
            "(declared)",
            "(7.2) right-Leibniz (proven)",
            "Cor 6.1 (proven)",
        ),
        notes="pre-metric-bourbaki.pdf Prop 6.1 / (7.9) route",
    )
    return chain, theorem
