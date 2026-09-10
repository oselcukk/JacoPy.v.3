"""
The Nambu ↔ Poisson DIALECT BRIDGE (2026-09-10): the Poisson
package (``SharpVF``, ``KoszulBracket``, ``PoissonStructure``) and the
Nambu package (``NambuSharpVF``, the ``ℒ_{Πω}η − ι_{Πη}dω`` Koszul
formula, ``NambuPoissonStructure``) describe the SAME order-1
geometry with two node families that the engine never bridged; the
Watamura [C'1] proof therefore re-proved the Koszul Jacobi identity
in the Nambu dialect instead of citing the 5.E.2b theorem.

The bridge is one definitional rule,
:class:`~jacopy.packages.poisson.nambu.NambuToPoissonSharpDefinition`
(``Π(ω) → π♯(ω)`` on the same bivector), plus the alternating
canonicalisation inside Poisson sharp slots. With it:

* :func:`prove_koszul_dialects_agree` — the two Koszul brackets are
  the same 1-form (paired against a probe), declaration-free: the
  Nambu formula and the Poisson ``ℒ_{π♯α}β − ℒ_{π♯β}α − d π(α,β)``
  differ by a Cartan magic formula and the sharp's evaluation;
* :func:`prove_theta_form_jacobi_by_citation` — the FORM component of
  [C'1] for the Watamura double, obtained by bridging the Nambu-
  dialect Jacobiator to the Poisson-dialect one (declaration-free
  step) and then CITING the general Koszul Jacobi theorem
  (:func:`jacopy.packages.poisson.koszul_jacobi.prove_general_koszul_jacobi`,
  declared Poisson condition) — the citation route the [C'1] proof
  could not take before.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem


class PoissonSharpSlotAlternatingNormalizeDefinition(Definition):
    """Alternating canonicalisation INSIDE a Poisson sharp's form
    slot, ``π♯(…π(η,ω)…) → −π♯(…π(ω,η)…)`` — the ``SharpVF`` twin of
    :class:`jacopy.packages.generalized.r_twisted.SharpSlotAlternatingNormalizeDefinition`
    (``simplify`` walks children only; an operator atom's slot is
    reached through the engine's slot protocol)."""

    anchor = None  # SharpVF (set in __init__)

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        from jacopy.packages.poisson.core import SharpVF

        self._registry = registry
        self.anchor = SharpVF
        self.name = (
            "alternating canonicalisation inside a Poisson sharp slot"
        )

    def _normalized(self, form: Expr) -> Expr:
        from jacopy.algorithms.normalize_alternating import (
            normalize_alternating,
        )

        return normalize_alternating(form, self._registry)

    def matches(self, expr: Expr) -> bool:
        from jacopy.packages.poisson.core import SharpVF

        return (
            isinstance(expr, SharpVF)
            and self._normalized(expr.alpha) != expr.alpha
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.packages.poisson.core import SharpVF

        norm = self._normalized(expr.alpha)
        if isinstance(norm, Neg):
            return Neg(SharpVF(expr.pi, norm.arg))
        return SharpVF(expr.pi, norm)


def bridge_engine(N, registry: Optional[PropertyRegistry] = None):
    """The Nambu tilde engine (declaration-free) plus the bridge and
    the Poisson package's sharp / Koszul definitions — an engine in
    which Nambu-dialect and Poisson-dialect expressions of the same
    bivector normalise together."""
    from jacopy.packages.drinfeld.tilde_calculus import _tilde_engine
    from jacopy.packages.poisson.core import (
        SharpActionDefinition,
        SharpEvaluationDefinition,
    )
    from jacopy.packages.poisson.koszul import (
        KoszulBracketDefinition,
        SharpVFLinearityDefinition,
    )
    from jacopy.packages.poisson.nambu import (
        NambuToPoissonSharpDefinition,
        poisson_view,
    )

    P = poisson_view(N)
    eng = _tilde_engine(N, registry, declare_fi=False)
    eng.register(NambuToPoissonSharpDefinition(N))
    eng.register(KoszulBracketDefinition(P))
    eng.register(SharpEvaluationDefinition(P))
    eng.register(SharpActionDefinition(P, registry))
    eng.register(SharpVFLinearityDefinition(P, registry))
    eng.register(
        PoissonSharpSlotAlternatingNormalizeDefinition(registry)
    )
    return eng


def _normalize(
    engine, expr: Expr, registry, *, max_steps: int = 60000
) -> Expr:
    """Engine normal form with a larger per-pass expansion budget
    than the default 1024 (a bridged 3-form Jacobiator needs a few
    thousand definitional steps in its first pass — size, not a
    cycle)."""
    from jacopy.algorithms.product_rule import product_rule
    from jacopy.algorithms.simplify import simplify

    cur = expr
    for _ in range(12):
        expanded, _steps = engine.expand(cur, max_steps=max_steps)
        reduced = simplify(product_rule(expanded, registry), registry)
        if reduced == cur:
            break
        cur = reduced
    return cur


def _poisson_jacobiator(pi, a: Expr, b: Expr, c: Expr, X: Expr) -> Expr:
    from jacopy.packages.poisson.koszul import KoszulBracket

    def J(x, y, z):
        return Pairing(KoszulBracket(pi, x, KoszulBracket(pi, y, z)), X)

    return Sum(J(a, b, c), J(b, c, a), J(c, a, b))


def _nambu_jacobiator(N, a: Expr, b: Expr, c: Expr, X: Expr) -> Expr:
    from jacopy.packages.poisson.nambu import nambu_koszul_bracket

    def J(x, y, z):
        return Pairing(
            nambu_koszul_bracket(N, x, nambu_koszul_bracket(N, y, z)), X
        )

    return Sum(J(a, b, c), J(b, c, a), J(c, a, b))


def prove_koszul_dialects_agree(
    N,
    omega: Expr,
    eta: Expr,
    X: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``⟨[ω,η]_Π, X⟩ = ⟨[ω,η]_π, X⟩`` — the Nambu-dialect Koszul
    bracket (``ℒ_{Πω}η − ι_{Πη}dω``) and the Poisson-dialect atom
    (``ℒ_{π♯ω}η − ℒ_{π♯η}ω − dπ(ω,η)``) are the same 1-form, for ANY
    bivector (declaration-free; Cartan magic + sharp evaluation)."""
    from jacopy.packages.poisson.koszul import KoszulBracket
    from jacopy.packages.poisson.nambu import nambu_koszul_bracket

    engine = bridge_engine(N, registry)
    diff = Sum(
        Pairing(nambu_koszul_bracket(N, omega, eta), X),
        Neg(Pairing(KoszulBracket(N.pi, omega, eta), X)),
    )
    nf = _normalize(engine, diff, registry)
    if nf != Integer(0):
        raise ProofFailure(
            "koszul_dialects_agree FAILS — residual "
            + nf._repr_inner()[:160]
        )
    chain = ProofChain(
        [
            ProofStep(
                diff,
                Integer(0),
                rule="Nambu Koszul − Poisson Koszul, paired with X, "
                "normalises to 0 through the dialect bridge",
                justification="engine normal form",
            )
        ]
    )
    theorem = Theorem(
        name="koszul_dialects_agree",
        statement="[ω,η]_Π (Nambu dialect: ℒ_{Πω}η − ι_{Πη}dω) = "
        "[ω,η]_π (Poisson dialect: ℒ_{π♯ω}η − ℒ_{π♯η}ω − dπ(ω,η)) "
        "for any bivector — the two node families denote one bracket",
        lhs=diff,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "dialect bridge Π(ω) = π♯(ω)",
            "Koszul bracket definitions of both packages",
            "Cartan magic formula (engine rule)",
        ),
        notes="declaration-free",
    )
    return chain, theorem


def prove_theta_form_jacobi_by_citation(
    N,
    omega: Expr,
    eta: Expr,
    zeta: Expr,
    X: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """The FORM component of [C'1] for the Watamura double
    ``(TM)₀ ⊕ (T*M)_θ`` — the Koszul Jacobi identity on general
    1-forms — by CITATION:

    1. the Nambu-dialect Jacobiator equals the Poisson-dialect one
       (declaration-free, the bridge);
    2. the Poisson-dialect Jacobiator vanishes under the declared
       Poisson condition — the 5.E.2b theorem
       :func:`~jacopy.packages.poisson.koszul_jacobi.prove_general_koszul_jacobi`,
       run and attached as the cited sub-proof.

    ``declare_poisson=False`` fails honestly (the cited theorem does).
    About 45 s (the citation dominates)."""
    from jacopy.packages.poisson.koszul_jacobi import (
        prove_general_koszul_jacobi,
    )
    from jacopy.packages.poisson.nambu import poisson_view

    P = poisson_view(N)
    engine = bridge_engine(N, registry)
    JN = _nambu_jacobiator(N, omega, eta, zeta, X)
    JP = _poisson_jacobiator(N.pi, omega, eta, zeta, X)
    diff = Sum(JN, Neg(JP))
    nf = _normalize(engine, diff, registry)
    if nf != Integer(0):
        raise ProofFailure(
            "theta_form_jacobi_by_citation: the two dialects' "
            "Jacobiators do not agree — residual "
            + nf._repr_inner()[:160]
        )
    steps: List[ProofStep] = [
        ProofStep(
            diff,
            Integer(0),
            rule="Nambu-dialect Jacobiator − Poisson-dialect Jacobiator "
            "normalises to 0 (dialect bridge; declaration-free)",
            justification="engine normal form",
        )
    ]
    cited = prove_general_koszul_jacobi(
        P,
        omega,
        eta,
        zeta,
        X,
        f,
        registry=registry,
        declare_poisson=declare_poisson,
    )
    steps.append(
        ProofStep(
            JP,
            Integer(0),
            rule="Koszul Jacobi on general 1-forms under the declared "
            "Poisson condition (cited: prove_general_koszul_jacobi, "
            "5.E.2b)",
            justification="cited",
            children=list(cited.steps),
            provenance_tag="theorem",
        )
    )
    chain = ProofChain(steps)
    theorem = Theorem(
        name="theta_form_jacobi_by_citation",
        statement="⟨[ω,[η,ζ]_θ]_θ + cyclic, X⟩ = 0 — the form "
        "component of [C'1] for (TM)₀ ⊕ (T*M)_θ, by bridging the "
        "Nambu dialect to the Poisson dialect and citing the general "
        "Koszul Jacobi theorem (5.E.2b)",
        lhs=JN,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "dialect bridge Π(ω) = π♯(ω)",
            "declared Poisson condition (in the cited theorem)",
            "CITED: prove_general_koszul_jacobi",
        ),
        notes="the citation route the Nambu-dialect [C'1] proof lacked",
    )
    return chain, theorem
