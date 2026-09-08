"""
Poisson generalized geometry (Phase 7.B.2b; PDF item 14f, Watamura
et al. [arXiv:1408.2649 §2-3]): the Courant algebroid
``(TM)₀ ⊕ (T*M)_θ`` over a Poisson base — the TILDE-Dorfman bracket
whose form side carries the Koszul bracket and whose vector side
carries the θ-induced tilde calculus, its skew TILDE-Courant
companion, and their axiom suite, concrete on ``TM ⊕ T*M``.

Representation: the Phase 6 Nambu machinery at ``p = 1`` — a
``NambuPoissonStructure`` of order 1 IS a Poisson bivector ``θ``,
and :func:`~jacopy.packages.drinfeld.double.nambu_double` at
``p = 1`` is exactly the triangular/LWX double of Watamura's
structure:

    vector: [U,V] + ℒ̃_ω V − ℒ̃_η U + d̃ ι̃_η U,
    form:   [ω,η]_θ + ℒ_U η − ℒ_V ω + d ι_V ω,

with total anchor ``ρ(U+ω) = U + θ♯ω`` and coboundary
``D_θ(h) = ½·(d̃h, dh) = ½·(−θ♯dh, dh)`` (Uchino Rem 1
normalization, as in :mod:`standard_courant`). Inventory:

* [C'3] right-Leibniz and [C'4] symmetric part — Phase 6.D, any
  bivector (:mod:`jacopy.packages.drinfeld.double`);
* **[C'2]** — the total anchor is a bracket morphism: THE 6.I.4
  capstone (4.19), under the declared fundamental identity (= the
  Poisson condition at ``p = 1``); re-exported here with the
  Watamura framing;
* this module adds, concrete: **[C'5]** invariance of the canonical
  pairing, the **tilde-Courant bracket** with skewness and the
  Dorfman↔Courant relation, and the ``⟨D_θh, x⟩₊ = ½ρ(x)h``
  normalization cross-check against the 7.A axiomatics.

The audit note is honored: this is the Watamura STRUCTURE itself
(θ-twisted double on a Poisson base), not a renaming of Π-twist
results — the R-flux twist of THIS structure is the separate
7.B.2c slice.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Product,
    Rational,
    Sum,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.tangent.exterior import d
from jacopy.packages.drinfeld.double import (
    canonical_pairing,
    nambu_double,
)
from jacopy.packages.drinfeld.tilde_calculus import _tilde_engine
from jacopy.packages.poisson.nambu import (
    NambuPoissonStructure,
    nambu_structure,
)


def poisson_base(name: str = "θ") -> NambuPoissonStructure:
    """The Watamura base datum: a Poisson bivector ``θ`` as the
    order-1 Nambu structure (the fundamental identity declared per
    proof = the Poisson condition ``[θ,θ]_SN = 0``)."""
    return nambu_structure(name, p=1)


def _require_poisson(N: NambuPoissonStructure) -> None:
    if N.p != 1:
        raise ValueError(
            "Poisson generalized geometry needs an order-1 "
            f"structure (a bivector); got p = {N.p}"
        )


def theta_dorfman(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
) -> Tuple[Expr, Expr]:
    """The tilde-Dorfman bracket of ``(TM)₀ ⊕ (T*M)_θ`` — the
    triangular double [Watamura §2; drinfeld eq (4.17)]."""
    _require_poisson(N)
    return nambu_double(N, U, omega, V, eta)


def theta_anchor(
    N: NambuPoissonStructure, U: Expr, omega: Expr
) -> Expr:
    """``ρ(U+ω) = U + θ♯ω`` — the total anchor [eq (4.19)]."""
    _require_poisson(N)
    return Sum(U, N.sharp_vf(omega))


def d_operator_theta(
    N: NambuPoissonStructure, h: Expr
) -> Tuple[Expr, Expr]:
    """``D_θ(h) = ½·(−θ♯dh, dh)`` — the θ-coboundary, normalized so
    ``⟨D_θh, x⟩₊ = ½·ρ(x)(h)`` (see
    :func:`prove_theta_d_pairing_value`)."""
    _require_poisson(N)
    return (
        Product(Rational(1, 2), Neg(N.sharp_vf(d(h)))),
        Product(Rational(1, 2), d(h)),
    )


def theta_courant(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
) -> Tuple[Expr, Expr]:
    """The TILDE-Courant bracket — the skew-symmetrization:
    ``[x,y]_{C,θ} = [x,y]_{D,θ} − D_θ⟨x,y⟩₊`` component-wise
    (PDF 14f.ii)."""
    vec, form = theta_dorfman(N, U, omega, V, eta)
    D_vec, D_form = d_operator_theta(
        N, canonical_pairing(U, omega, V, eta)
    )
    return Sum(vec, Neg(D_vec)), Sum(form, Neg(D_form))


def prove_theta_anchor_morphism(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
) -> ProofChain:
    """[C'2] for the Poisson generalized double:
    ``ρ([x,y]_{D,θ}) = [ρ(x), ρ(y)]_Lie`` probed on ``h`` — THE
    6.I.4 capstone (4.19) at ``p = 1``, under the declared Poisson
    condition (honest-fail without)."""
    from jacopy.packages.drinfeld.bracket_morphism import (
        prove_total_anchor_is_bracket_morphism,
    )

    _require_poisson(N)
    return prove_total_anchor_is_bracket_morphism(
        N,
        U,
        omega,
        V,
        eta,
        h,
        registry=registry,
        declare_fi=declare_poisson,
    )


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


def _zero_theorem(
    name: str,
    statement: str,
    diffs,
    engine,
    registry,
    *,
    from_axioms,
    notes: str,
    labels,
) -> Tuple[ProofChain, Theorem]:
    steps: List[ProofStep] = []
    for label, diff in zip(labels, diffs):
        nf = _normalize(engine, diff, registry)
        if nf != Integer(0):
            raise ProofFailure(
                f"{name}: {label} FAILS — residual "
                + nf._repr_inner()[:160]
            )
        steps.append(
            ProofStep(
                diff,
                Integer(0),
                rule=label,
                justification=(
                    "engine normal form of the difference"
                ),
            )
        )
    chain = ProofChain(steps)
    theorem = Theorem(
        name=name,
        statement=statement,
        lhs=diffs[0],
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=from_axioms,
        notes=notes,
    )
    return chain, theorem


def prove_theta_courant_skew(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The tilde-Courant bracket is genuinely SKEW:
    ``[x,y]_{C,θ} + [y,x]_{C,θ} = 0`` component-wise — the 6.D
    symmetric part of the tilde-Dorfman is exactly ``2·D_θ⟨x,y⟩₊``
    and the subtraction removes it. Holds for ANY bivector (no
    Poisson declaration)."""
    _require_poisson(N)
    engine = _tilde_engine(N, registry, declare_fi=False)
    ab_vec, ab_form = theta_courant(N, U, omega, V, eta)
    ba_vec, ba_form = theta_courant(N, V, eta, U, omega)
    return _zero_theorem(
        "poisson_generalized_courant_skew",
        "[x,y]_{C,θ} + [y,x]_{C,θ} = 0 on (TM)₀ ⊕ (T*M)_θ "
        "(tilde-Courant skewness; equivalently the relation "
        "[x,y]_{C,θ} = [x,y]_{D,θ} − D_θ⟨x,y⟩₊ kills the 6.D "
        "symmetric part). Any bivector.",
        [Sum(ab_vec, ba_vec), Sum(ab_form, ba_form)],
        engine,
        registry,
        from_axioms=(
            "tilde-Dorfman + pairing + D_θ definitions",
            "6.D symmetric part (𝒟 = d + d̃)",
        ),
        notes="PDF 14f.ii / Watamura §2",
        labels=(
            "vector component normalizes to 0",
            "form component normalizes to 0",
        ),
    )


def prove_theta_d_pairing_value(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The Uchino Rem 1 normalization of the θ-coboundary:

    ``⟨D_θh, x⟩₊ = ½·ρ(x)(h)``

    — the dh-component pairs with the vector part, the −θ♯dh
    component with the form part (bivector antisymmetry), together
    reproducing half the total-anchor action. Ties the θ-double to
    the 7.A abstract axiomatics."""
    _require_poisson(N)
    engine = _tilde_engine(N, registry, declare_fi=False)
    D_vec, D_form = d_operator_theta(N, h)
    lhs = canonical_pairing(D_vec, D_form, U, omega)
    rhs = Product(
        Rational(1, 2), Act(theta_anchor(N, U, omega), h)
    )
    return _zero_theorem(
        "poisson_generalized_d_pairing",
        "⟨D_θh, x⟩₊ = ½ρ(x)(h) on (TM)₀ ⊕ (T*M)_θ (Uchino Rem 1 "
        "normalization; 7.A cross-check)",
        [Sum(lhs, Neg(rhs))],
        engine,
        registry,
        from_axioms=(
            "D_θ + pairing + total anchor definitions",
            "bivector antisymmetry",
        ),
        notes="PDF 14f",
        labels=(
            "pairing value normalizes to the anchor action",
        ),
    )



# ------------------------------------------------------------------ #
# Sharp-pairing canonicalization (the [C'5] closure, 2026-09-08)     #
# ------------------------------------------------------------------ #
#
# One scalar, three faces: θ(a, b) = ⟨b, θ♯a⟩ = −⟨a, θ♯b⟩ (the
# defining relation of the sharp plus bivector antisymmetry). The
# canonical representative, oriented so the three rules below are
# jointly TERMINATING (each strictly reduces in a well-founded
# order: MultiEval → Pairing for non-exact slots; Pairing →
# MultiEval when the sharp slot is exact; sharp always on the
# smaller-repr argument):
#
# * both slots non-exact  → ⟨larger, θ♯smaller⟩ (Pairing form),
# * a slot exact          → θ(dh, b) (MultiEval form — the sharp
#   ACTION's own canonical output, so no cycle with it).
#
# An earlier revision oriented the side-swap the WRONG way and
# cycled against the unfold; the fix is the orientation, not a
# deeper confluence obstacle.


class ThetaUnfoldDefinition(Definition):
    """``θ(a, b) → ⟨b, θ♯a⟩`` when BOTH slots are non-exact."""

    anchor = MultiEval

    def __init__(self, N: NambuPoissonStructure) -> None:
        self._N = N
        self.name = (
            "sharp-pairing canonical (unfold): "
            "θ(a,b) = ⟨b, θ♯a⟩ (both slots non-exact)"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.packages.poisson.showcase import _is_exact

        return (
            isinstance(expr, MultiEval)
            and expr.head == self._N.pi
            and expr.arity == 2
            and not _is_exact(expr.args[0])
            and not _is_exact(expr.args[1])
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.pairing import Pairing

        a, b = expr.args
        return Pairing(b, self._N.sharp_vf(a))


class SharpPairingExactFoldDefinition(Definition):
    """``⟨b, θ♯dh⟩ → θ(dh, b)`` — an EXACT sharp slot folds back to
    the MultiEval face (the sharp action's canonical form)."""

    def __init__(self, N: NambuPoissonStructure) -> None:
        from jacopy.packages.poisson.nambu import NambuSharpVF

        self._N = N
        self._sharp_cls = NambuSharpVF
        self.anchor = __import__(
            "jacopy.core.pairing", fromlist=["Pairing"]
        ).Pairing
        self.name = (
            "sharp-pairing canonical (exact fold): "
            "⟨b, θ♯dh⟩ = θ(dh, b)"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.core.pairing import Pairing
        from jacopy.packages.poisson.showcase import _is_exact

        return (
            isinstance(expr, Pairing)
            and isinstance(expr.X, self._sharp_cls)
            and expr.X.pi == self._N.pi
            and _is_exact(expr.X.omega)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return MultiEval(
            self._N.pi,
            expr.X.omega,
            expr.alpha,
            alternating=True,
            slot_kind="covector",
        )


class SharpPairingSideDefinition(Definition):
    """``⟨a, θ♯b⟩ → −⟨b, θ♯a⟩`` when the sharp carries the
    LARGER-repr argument (bivector antisymmetry; matches the unfold
    output, so jointly terminating)."""

    def __init__(self, N: NambuPoissonStructure) -> None:
        from jacopy.core.pairing import Pairing
        from jacopy.packages.poisson.nambu import NambuSharpVF

        self._N = N
        self._sharp_cls = NambuSharpVF
        self.anchor = Pairing
        self.name = (
            "sharp-pairing canonical (side): "
            "⟨a, θ♯b⟩ = −⟨b, θ♯a⟩ (sharp on the smaller)"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.core.pairing import Pairing
        from jacopy.packages.poisson.showcase import _is_exact

        return (
            isinstance(expr, Pairing)
            and isinstance(expr.X, self._sharp_cls)
            and expr.X.pi == self._N.pi
            and not _is_exact(expr.X.omega)
            and not _is_exact(expr.alpha)
            and expr.X.omega._repr_inner()
            > expr.alpha._repr_inner()
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.pairing import Pairing

        return Neg(
            Pairing(
                expr.X.omega,
                self._N.sharp_vf(expr.alpha),
            )
        )


def _invariance_engine(N, registry, *, declare_poisson: bool):
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )
    from jacopy.packages.drinfeld.tilde_calculus import (
        InteriorAnticommuteDefinition,
        LieIotaCommutatorDefinition,
        NambuMorphismDeclaration,
    )
    from jacopy.packages.drinfeld.twist import _twist_engine

    engine = _twist_engine(N, registry)
    engine.register(LieIotaCommutatorDefinition())
    engine.register(InteriorAnticommuteDefinition())
    if declare_poisson:
        engine.register(NambuMorphismDeclaration(N))
    engine.register(ActExpansionDefinition(registry))
    engine.register(MultiEvalArgLinearityDefinition(registry))
    engine.register(LieBracketLeibnizDefinition(registry))
    engine.register(ThetaUnfoldDefinition(N))
    engine.register(SharpPairingExactFoldDefinition(N))
    engine.register(SharpPairingSideDefinition(N))
    return engine


def prove_theta_invariance(
    N: NambuPoissonStructure,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    zeta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """[C'5] for the Poisson generalized double:

    ``ρ(x)⟨y, z⟩₊ = ⟨x∘_θ y, z⟩₊ + ⟨y, x∘_θ z⟩₊``

    with the TOTAL anchor acting on the scalar pairing — CLOSED
    (2026-09-08, second pass): the missing ingredient was a
    consistent canonical representative for the three faces
    ``θ(a,b) = ⟨b,θ♯a⟩ = −⟨a,θ♯b⟩`` of the sharp-pairing scalar.
    With the exact-oriented canonicalization triple above the
    invariance defect normalizes to literal 0 outright (the earlier
    'diagnosed open' status came from a wrongly-oriented side rule
    that cycled — the orientation, not confluence, was the
    obstacle)."""
    _require_poisson(N)
    engine = _invariance_engine(
        N, registry, declare_poisson=declare_poisson
    )
    lhs = Act(
        theta_anchor(N, U, omega),
        canonical_pairing(V, eta, W, zeta),
    )
    xy_vec, xy_form = theta_dorfman(N, U, omega, V, eta)
    xz_vec, xz_form = theta_dorfman(N, U, omega, W, zeta)
    rhs = Sum(
        canonical_pairing(xy_vec, xy_form, W, zeta),
        canonical_pairing(V, eta, xz_vec, xz_form),
    )
    return _zero_theorem(
        "poisson_generalized_invariance",
        "ρ(x)⟨y,z⟩₊ = ⟨x∘_θ y, z⟩₊ + ⟨y, x∘_θ z⟩₊ on "
        "(TM)₀ ⊕ (T*M)_θ ([C'5], total anchor)",
        [Sum(lhs, Neg(rhs))],
        engine,
        registry,
        from_axioms=(
            "tilde-Dorfman + canonical pairing definitions",
            "Cartan + Koszul/tilde calculus",
            "sharp-pairing canonicalization (definitional: "
            "θ(a,b) = ⟨b,θ♯a⟩ + bivector antisymmetry)",
        )
        + (
            ("declared Poisson condition",)
            if declare_poisson
            else ()
        ),
        notes=(
            "PDF 14f.i / Watamura §2 — the 14f.i deferral CLOSES"
        ),
        labels=("invariance defect normalizes to 0",),
    )
