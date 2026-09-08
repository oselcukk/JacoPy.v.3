"""
Theorem 9.2 + Corollary 9.1 — the Ševera-lite CLASSIFICATION of
exact pre-metric-Bourbaki algebroids (Phase 7.B.2f.4; primary paper
§9): every exact pre-metric-Bourbaki algebroid with g-isotropic
``χ`` is the STANDARD one twisted by a pair ``(F, H)``,

    g((φ⊕χ)(U+z), (φ⊕χ)(V+y)) = g_S(U+z, V+y) + F(U,V),   (9.19)
    [(φ⊕χ)(U+z), (φ⊕χ)(V+y)]  = (φ⊕χ)[U+z, V+y]_S + χH(U,V), (9.20)

with ``F(U,V) = g(φU, φV)`` and ``H`` obeying the (9.21) laws. The
paper's proof decomposes the bracket into four leg-types; each step
is mechanized here as its own theorem:

* ``[χz, φV] = χ(−ℒ^Z_V z + dι_V z)`` — symmetric part + (9.7);
* ``[χz, χy] = 0`` — the invariance instance at ``(χz, χy, w)``
  with exactness + isotropy + the R-valued non-degeneracy strip;
* ``Δ(U,V) := [φU, φV] − φ[U,V]`` is TENSORIAL in the second slot
  (right-Leibniz (7.2), cited as a theorem rule) and picks up the
  ``χ l_{df} F`` anomaly in the first (left-Leibniz of 7.B.2f.2 +
  ``𝕃 = χl``) — exactly the (9.21) laws of ``H``;
* ``ρ(Δ(U,V)) = 0`` — ``im Δ ⊂ im χ``, so ``χH = Δ`` defines ``H``;
* **Cor 9.1**: ``Δ(U,V) + Δ(V,U) = χ(d F(U,V))`` — so for a
  g-ISOTROPIC ``φ`` (``F = 0``, declared) the twist ``H`` is
  antisymmetric: a Z-valued 2-form. The Ševera picture: exact
  structures are classified by ``(F, H)`` twists of the standard
  bracket.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
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
from jacopy.central.algebroid.context import (
    Algebroid,
    AlgebroidBracket,
)
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.generalized.bourbaki_exact import (
    ChiSec,
    PhiSec,
    _invariance_instance,
    exact_bourbaki_engine,
)
from jacopy.packages.generalized.bourbaki_precalculus import (
    BD,
    BIota,
    BLieZ,
    BSymbol,
    _split_scalar,
)
from jacopy.packages.generalized.bourbaki_structure import (
    BMetric,
    BSymbolL,
    _bmetric_coefficient_of,
)
from jacopy.algorithms.simplify import simplify


# ------------------------------------------------------------------ #
# Extra rules for the classification                                 #
# ------------------------------------------------------------------ #


class SplitLinearityDefinition(Definition):
    """C∞-linearity of the bundle morphisms ``φ`` and ``χ``:
    sums, negations, zeros and scalar factors pull out."""

    name = "φ/χ C∞-linearity (bundle morphisms)"
    anchor = (PhiSec, ChiSec)

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, (PhiSec, ChiSec)):
            return False
        (s,) = expr.rewritable_slots
        return (
            isinstance(s, (Sum, Neg))
            or s == Integer(0)
            or _split_scalar(s, self._registry) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        (s,) = expr.rewritable_slots
        if s == Integer(0):
            return Integer(0)
        if isinstance(s, Sum):
            return Sum(
                *(expr.with_slots(c) for c in s.children)
            )
        if isinstance(s, Neg):
            return Neg(expr.with_slots(s.arg))
        f, rest = _split_scalar(s, self._registry)
        return Product(f, expr.with_slots(rest))


class RightLeibnizTheoremRule(Definition):
    """THEOREM-classified right-Leibniz on E, citing the proven
    (7.2): ``[u, f·v]_E → f·[u,v]_E + ρ(u)(f)·v`` (composite
    second slot)."""

    anchor = AlgebroidBracket

    def __init__(
        self,
        alg: Algebroid,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._alg = alg
        self._registry = registry
        self.name = (
            "cite (7.2) right-Leibniz: [u, fv] = f[u,v] + ρ(u)f·v"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._alg.name
            and _split_scalar(expr.v, self._registry)
            is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        f, rest = _split_scalar(expr.v, self._registry)
        return Sum(
            Product(
                f,
                AlgebroidBracket(
                    expr.algebroid_name, expr.u, rest
                ),
            ),
            Product(
                Act(self._alg.anchor(expr.u), f), rest
            ),
        )


class LeftLeibnizTheoremRule(Definition):
    """THEOREM-classified left-Leibniz on E, citing the proven
    7.B.2f.2 corollary:
    ``[f·u, v]_E → f·[u,v]_E − ρ(v)(f)·u + 𝕃_{df}(g(u,v))``."""

    anchor = AlgebroidBracket

    def __init__(
        self,
        alg: Algebroid,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._alg = alg
        self._registry = registry
        self.name = (
            "cite left-Leibniz: [fu, v] = f[u,v] − ρ(v)f·u "
            "+ 𝕃_df g(u,v)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and expr.algebroid_name == self._alg.name
            and _split_scalar(expr.u, self._registry)
            is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        f, rest = _split_scalar(expr.u, self._registry)
        return Sum(
            Product(
                f,
                AlgebroidBracket(
                    expr.algebroid_name, rest, expr.v
                ),
            ),
            Neg(
                Product(
                    Act(self._alg.anchor(expr.v), f), rest
                )
            ),
            BSymbolL(f, BMetric(rest, expr.v)),
        )


class LSymbolViaChiDefinition(Definition):
    """(8.3): the symbol decomposes as ``𝕃_{df} r → χ(l_{df} r)``."""

    name = "(8.3): 𝕃 = χ∘l"
    anchor = BSymbolL

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, BSymbolL)

    def rewrite(self, expr: Expr) -> Expr:
        return ChiSec(BSymbol(expr.f, expr.r))


class SymPartCollectDeclaration(Definition):
    """DECLARED symmetric part (6.11), COLLECT orientation: a
    same-signed pair ``[a,b]_E … [b,a]_E`` inside a Sum is replaced
    by ``𝔻g(a,b)`` (the Faz 3 SymmetricPartDeclaration pattern —
    the single-swap orientation would not terminate on generic
    slots)."""

    name = "(6.11) collect: [a,b] + [b,a] → 𝔻g(a,b)"
    anchor = Sum

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg

    def _find_pair(self, expr: Sum):
        terms = list(expr.children)
        for i, t in enumerate(terms):
            sign_i, core_i = (
                (-1, t.arg) if isinstance(t, Neg) else (1, t)
            )
            if not isinstance(core_i, AlgebroidBracket):
                continue
            for j in range(i + 1, len(terms)):
                s = terms[j]
                sign_j, core_j = (
                    (-1, s.arg)
                    if isinstance(s, Neg)
                    else (1, s)
                )
                if (
                    isinstance(core_j, AlgebroidBracket)
                    and sign_i == sign_j
                    and core_j.u == core_i.v
                    and core_j.v == core_i.u
                    and core_i.u != core_i.v
                ):
                    return i, j, sign_i, core_i
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Sum)
            and self._find_pair(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.packages.generalized.bourbaki_structure import (
            BDop,
        )

        i, j, sign, core = self._find_pair(expr)
        replacement = BDop(BMetric(core.u, core.v))
        if sign < 0:
            replacement = Neg(replacement)
        rest = [
            t
            for k, t in enumerate(expr.children)
            if k not in (i, j)
        ]
        if not rest:
            return replacement
        return Sum(*rest, replacement)


class PhiIsotropyDeclaration(Definition):
    """DECLARED g-isotropy of the SPLITTING ``φ`` (Cor 9.1's extra
    hypothesis): ``g(φU, φV) → 0`` — i.e. ``F = 0``."""

    name = "declared g-isotropic splitting: F(U,V) = g(φU,φV) = 0"
    anchor = BMetric

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, BMetric)
            and isinstance(expr.u, PhiSec)
            and isinstance(expr.v, PhiSec)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


def severa_engine(
    alg: Algebroid,
    registry: Optional[PropertyRegistry] = None,
    *,
    isotropic_chi: bool = True,
    isotropic_phi: bool = False,
    leibniz_rules: bool = True,
) -> ExpansionEngine:
    from jacopy.central.tangent.lie_bracket import (
        BracketOrientationDefinition,
        LieBracketLeibnizDefinition,
    )

    rules: List[Definition] = [
        SplitLinearityDefinition(registry),
        LSymbolViaChiDefinition(),
        SymPartCollectDeclaration(alg),
        LieBracketLeibnizDefinition(registry),
        BracketOrientationDefinition(),
    ]
    if leibniz_rules:
        rules.append(RightLeibnizTheoremRule(alg, registry))
        rules.append(LeftLeibnizTheoremRule(alg, registry))
    if isotropic_phi:
        rules.append(PhiIsotropyDeclaration())
    eng = ExpansionEngine(rules)
    for d_ in exact_bourbaki_engine(
        alg, registry, isotropic=isotropic_chi
    ).definitions:
        eng.register(d_)
    return eng


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


def delta(alg: Algebroid, U: Expr, V: Expr) -> Expr:
    """``Δ(U,V) := [φU, φV]_E − φ([U,V])`` — the horizontal
    curvature of the splitting; ``χH = Δ``."""
    return Sum(
        alg.bracket(PhiSec(U), PhiSec(V)),
        Neg(PhiSec(lie_bracket(U, V))),
    )


def _zero_theorem(
    name,
    statement,
    diff,
    engine,
    registry,
    *,
    from_axioms,
    notes,
    label,
) -> Tuple[ProofChain, Theorem]:
    nf = _normalize(engine, diff, registry)
    if nf != Integer(0):
        raise ProofFailure(
            f"{name}: {label} FAILS — residual "
            + nf._repr_inner()[:160]
        )
    chain = ProofChain(
        [
            ProofStep(
                diff,
                Integer(0),
                rule=label,
                justification="engine normal form",
            )
        ]
    )
    theorem = Theorem(
        name=name,
        statement=statement,
        lhs=diff,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=from_axioms,
        notes=notes,
    )
    return chain, theorem


def prove_chi_phi_bracket(
    alg: Algebroid,
    V: Expr,
    z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Thm 9.2, leg 2: ``[χz, φV]_E = χ(−ℒ^Z_V z + dι_V z)`` —
    the symmetric part (6.11) + the (9.7)/(9.9)/(8.3)
    definitions."""
    engine = severa_engine(
        alg, registry, leibniz_rules=False
    )
    diff = Sum(
        alg.bracket(ChiSec(z), PhiSec(V)),
        Neg(
            ChiSec(
                Sum(Neg(BLieZ(V, z)), BD(BIota(V, z)))
            )
        ),
    )
    return _zero_theorem(
        f"severa_chi_phi_bracket_{alg.name}",
        "[χz, φV]_E = χ(−ℒ^Z_V z + dι_V z) (Thm 9.2, "
        "χ-φ leg)",
        diff,
        engine,
        registry,
        from_axioms=(
            "(6.11) symmetric part (declared, χ-φ orientation)",
            "(9.7)/(9.9)/(8.3) definitions",
        ),
        notes="pre-metric-bourbaki.pdf Thm 9.2 proof",
        label="χ-φ leg normalizes to 0",
    )


def prove_chi_chi_bracket_vanishes(
    alg: Algebroid,
    z: Expr,
    y: Expr,
    *,
    w: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Thm 9.2, leg 4: ``[χz, χy]_E = 0`` — the invariance
    instance at ``(χz, χy, w)``: the left side dies by exactness,
    the ``g(χy, [χz, w])`` term dies by isotropy + anchor
    morphism, and the R-valued non-degeneracy strips the probe."""
    from jacopy.central.algebroid.theorems import (
        _fresh_section,
    )

    if w is None:
        w = _fresh_section(alg, z, y)
    engine = severa_engine(
        alg, registry, leibniz_rules=False
    )
    instance = _invariance_instance(
        alg, ChiSec(z), ChiSec(y), w
    )
    step_cite = ProofStep(
        instance,
        instance,
        rule="cite the (7.8) invariance instance at (χz, χy, w)",
        justification="declared axiom",
        provenance_tag="axiom",
    )
    residual = _normalize(engine, instance, registry)
    step_norm = ProofStep(
        instance,
        residual,
        rule=(
            "normalize: exactness kills ℒ^R; isotropy + anchor "
            "morphism kill the [χz, w] leg"
        ),
        justification="engine normal form",
    )
    if residual == Integer(0):
        raise ProofFailure(
            "Thm 9.2 χ-χ leg: the instance normalized to 0 "
            "before the strip — nothing to extract"
        )
    contributions = _bmetric_coefficient_of(residual, w)
    X = simplify(
        contributions[0]
        if len(contributions) == 1
        else Sum(*contributions),
        registry,
    )
    target = alg.bracket(ChiSec(z), ChiSec(y))
    sign, core = (
        (1, X)
        if not isinstance(X, Neg)
        else (-1, X.arg)
    )
    if core != target:
        raise ProofFailure(
            "Thm 9.2 χ-χ leg: expected ±[χz, χy], got "
            + X._repr_inner()[:140]
        )
    step_strip = ProofStep(
        residual,
        Integer(0),
        rule=(
            "non-degeneracy of the R-valued metric (generic "
            "probe w): [χz, χy] = 0"
        ),
        justification="g(X, w) = 0 for generic w ⟹ X = 0",
    )
    chain = ProofChain([step_cite, step_norm, step_strip])
    theorem = Theorem(
        name=f"severa_chi_chi_vanishes_{alg.name}",
        statement=(
            "[χz, χy]_E = 0 (Thm 9.2, χ-χ leg: the Z-directions "
            "bracket to zero)"
        ),
        lhs=target,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "(7.8) invariance (declared instance)",
            "exactness + declared g-isotropy of χ",
            f"anchor morphism ({alg.name}, declared)",
            "non-degeneracy of the R-valued metric",
        ),
        notes="pre-metric-bourbaki.pdf Thm 9.2 proof",
    )
    return chain, theorem


def prove_delta_tensorial_second_slot(
    alg: Algebroid,
    U: Expr,
    V: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Thm 9.2 / (9.21) first law: ``Δ(U, f·V) = f·Δ(U,V)`` —
    hence ``H(U, fV) = fH(U,V)`` (right-Leibniz (7.2) cited)."""
    engine = severa_engine(alg, registry)
    diff = Sum(
        delta(alg, U, Product(f, V)),
        Neg(Product(f, delta(alg, U, V))),
    )
    return _zero_theorem(
        f"severa_delta_second_tensorial_{alg.name}",
        "Δ(U, f·V) = f·Δ(U,V) — H(U, fV) = fH(U,V) "
        "((9.21), first law)",
        diff,
        engine,
        registry,
        from_axioms=(
            "(7.2) right-Leibniz (proven, cited as a rule)",
            "φ C∞-linearity (bundle morphism)",
        ),
        notes="pre-metric-bourbaki.pdf Thm 9.2 proof / (9.21)",
        label="second-slot tensoriality normalizes to 0",
    )


def prove_delta_first_slot_anomaly(
    alg: Algebroid,
    U: Expr,
    V: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Thm 9.2 / (9.21) second law:
    ``Δ(f·U, V) = f·Δ(U,V) + χ(l_{df} F(U,V))`` with
    ``F(U,V) = g(φU, φV)`` — hence
    ``H(fU, V) = fH(U,V) + l_{df}F(U,V)`` (left-Leibniz cited;
    ``𝕃 = χl``)."""
    engine = severa_engine(alg, registry)
    F = BMetric(PhiSec(U), PhiSec(V))
    diff = Sum(
        delta(alg, Product(f, U), V),
        Neg(Product(f, delta(alg, U, V))),
        Neg(ChiSec(BSymbol(f, F))),
    )
    return _zero_theorem(
        f"severa_delta_first_anomaly_{alg.name}",
        "Δ(f·U, V) = f·Δ(U,V) + χ(l_df F(U,V)) — "
        "H(fU,V) = fH(U,V) + l_df F(U,V) ((9.21), second law)",
        diff,
        engine,
        registry,
        from_axioms=(
            "left-Leibniz (proven 7.B.2f.2, cited as a rule)",
            "(8.3) 𝕃 = χ∘l",
            "φ C∞-linearity (bundle morphism)",
        ),
        notes="pre-metric-bourbaki.pdf Thm 9.2 proof / (9.21)",
        label="first-slot anomaly normalizes to 0",
    )


def prove_delta_lands_in_chi(
    alg: Algebroid,
    U: Expr,
    V: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Thm 9.2: ``ρ(Δ(U,V)) = 0`` — ``im Δ ⊂ im χ``, so
    ``χH = Δ`` well-defines the twist ``H`` (probed on ``h``)."""
    engine = severa_engine(
        alg, registry, leibniz_rules=False
    )
    node = Act(alg.anchor(delta(alg, U, V)), h)
    return _zero_theorem(
        f"severa_delta_in_chi_{alg.name}",
        "ρ(Δ(U,V)) = 0 — im Δ ⊂ im χ, so χH = Δ defines the "
        "twist H (Thm 9.2)",
        node,
        engine,
        registry,
        from_axioms=(
            f"anchor morphism ({alg.name}, declared)",
            "exact splitting ρφ = id",
        ),
        notes="pre-metric-bourbaki.pdf Thm 9.2 proof",
        label="anchor of Δ normalizes to 0",
    )


def prove_h_antisymmetric_for_isotropic_phi(
    alg: Algebroid,
    U: Expr,
    V: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    isotropic_phi: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """Cor 9.1: ``Δ(U,V) + Δ(V,U) = χ(d F(U,V))`` — for a
    g-ISOTROPIC splitting ``φ`` (``F = 0``, declared) the twist is
    ANTISYMMETRIC: ``H`` is a Z-valued 2-form. Without the
    declaration the symmetric part survives as ``χ(dF)``
    (honest-fail shows it)."""
    engine = severa_engine(
        alg,
        registry,
        leibniz_rules=False,
        isotropic_phi=isotropic_phi,
    )
    F = BMetric(PhiSec(U), PhiSec(V))
    diff = (
        Sum(delta(alg, U, V), delta(alg, V, U))
        if isotropic_phi
        else Sum(
            delta(alg, U, V),
            delta(alg, V, U),
            Neg(ChiSec(BD(F))),
        )
    )
    label = (
        "H(U,V) + H(V,U) = 0 (F = 0 declared)"
        if isotropic_phi
        else "Δ(U,V) + Δ(V,U) = χ(dF(U,V))"
    )
    return _zero_theorem(
        f"severa_h_antisymmetric_{alg.name}",
        (
            "H(U,V) + H(V,U) = 0 — for a g-isotropic φ the "
            "Ševera twist H is a Z-valued 2-form (Cor 9.1)"
            if isotropic_phi
            else "Δ(U,V) + Δ(V,U) = χ(d F(U,V)) (Cor 9.1, "
            "general form)"
        ),
        diff,
        engine,
        registry,
        from_axioms=(
            "(6.11) symmetric part (declared)",
            "(8.3) 𝔻 = χd",
            "Lie antisymmetry",
        )
        + (
            ("declared g-isotropic splitting φ (F = 0)",)
            if isotropic_phi
            else ()
        ),
        notes="pre-metric-bourbaki.pdf Cor 9.1",
        label=label,
    )
