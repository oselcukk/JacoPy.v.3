"""
EXACT pre-metric-Bourbaki algebroids and the INDUCED pre-calculus
(Phase 7.B.2f.3), from the PRIMARY paper §8-9: the splitting data
``φ : T(M) → E``, ``χ : Z → E`` with

    ρ∘φ = id,   ρ∘χ = 0   (exactness, 8.5),
    𝔻 = χ∘d,               (8.3)
    χ ℒ^Z_V z = [φV, χz]_E (9.7 — the DEFINITION of ℒ^Z),
    ι_V z = g(χz, φV)      (9.9 — the DEFINITION of ι),

mechanized as engine rules on top of the 7.B.2f.2 structure layer
(R-valued ``g``, 𝔻/𝕃, ℒ^R) and the 7.B.2f.1 quintet nodes
(``BIota``, ``BD``, ``BLieZ``). Theorems:

* **Lemma 9.1** — ``[u, χz]_E`` is in the image of ``χ``:
  mechanized as ``ρ([u, χz]) = 0`` (anchor morphism + exactness),
  probed on a generic scalar;
* **Prop 9.1 (9.11)** — the induced quintet satisfies the axiom
  (9.15): the metric-invariance instance at ``(χz, φU, φV)`` has a
  VANISHING left side (exactness kills ``ℒ^R`` in a ``ρ∘χ = 0``
  direction) and its right side unfolds, through the declared
  symmetric part + (9.7) + (9.9) + ``𝔻 = χd``, to exactly the
  (9.15) combination;
* **Prop 9.1 (9.12)** — with g-ISOTROPIC ``χ`` (declared:
  ``g(χz, u) = ι_{ρ(u)} z`` beyond the definitional φ-leg), the
  invariance instance at ``(φU, φV, χz)`` yields the axiom (9.14).

Together with Thm 9.1 (7.B.2f.1) this closes the circle: a Bourbaki
pre-calculus builds an exact pre-metric-Bourbaki algebroid, and an
exact pre-metric-Bourbaki algebroid induces a Bourbaki pre-calculus.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Sum,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.algebroid.context import (
    Algebroid,
    AlgebroidBracket,
    AnchoredVF,
    algebroid,
)
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.generalized.bourbaki_precalculus import (
    BD,
    BIota,
    BLieZ,
    _SlotAtom,
)
from jacopy.packages.generalized.bourbaki_structure import (
    BDop,
    BLieRE,
    BMetric,
    bourbaki_structure_engine,
)


class PhiSec(_SlotAtom):
    """``φ(U) ∈ 𝔛(E)`` — the right-splitting's image of an M-vector
    (``ρφ = id``)."""

    def __init__(self, U: Expr) -> None:
        super().__init__(f"φ({U._repr_inner()})", U)

    @property
    def U(self) -> Expr:
        return self._slots[0]

    def with_slots(self, U: Expr) -> "PhiSec":
        return PhiSec(U)


class ChiSec(_SlotAtom):
    """``χ(z) ∈ 𝔛(E)`` — the inclusion of a Z-section
    (``ρχ = 0``)."""

    def __init__(self, z: Expr) -> None:
        super().__init__(f"χ({z._repr_inner()})", z)

    @property
    def z(self) -> Expr:
        return self._slots[0]

    def with_slots(self, z: Expr) -> "ChiSec":
        return ChiSec(z)


# ------------------------------------------------------------------ #
# Splitting rules                                                    #
# ------------------------------------------------------------------ #


class SplitAnchorDefinition(Definition):
    """Exactness + splitting on the anchor: ``ρ(φU) → U`` and
    ``ρ(χz) → 0`` (8.5)."""

    name = "exact splitting anchors: ρ∘φ = id, ρ∘χ = 0 (8.5)"
    anchor = AnchoredVF

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, AnchoredVF) and isinstance(
            expr.section, (PhiSec, ChiSec)
        )

    def rewrite(self, expr: Expr) -> Expr:
        s = expr.section
        if isinstance(s, PhiSec):
            return s.U
        return Integer(0)


class LieZDefinitionRule(Definition):
    """(9.7) — the DEFINITION of ``ℒ^Z``:
    ``[φV, χz]_E → χ(ℒ^Z_V z)`` (well-defined by Lemma 9.1)."""

    name = "(9.7): [φV, χz]_E = χ(ℒ^Z_V z)"
    anchor = AlgebroidBracket

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and isinstance(expr.u, PhiSec)
            and isinstance(expr.v, ChiSec)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return ChiSec(BLieZ(expr.u.U, expr.v.z))


class SymPartChiSwapDeclaration(Definition):
    """DECLARED symmetric part (6.11), oriented for the χ-φ order:
    ``[χz, φU]_E → −[φU, χz]_E + 𝔻g(χz, φU)`` — terminating (the
    χ-first bracket disappears; (9.7) then absorbs the φ-first
    one)."""

    name = "(6.11) at χ-φ: [χz,φU] = −[φU,χz] + 𝔻g(χz,φU)"
    anchor = AlgebroidBracket

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, AlgebroidBracket)
            and isinstance(expr.u, ChiSec)
            and isinstance(expr.v, PhiSec)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Sum(
            Neg(AlgebroidBracket("E", expr.v, expr.u)),
            BDop(BMetric(expr.u, expr.v)),
        )


class DopViaChiDefinition(Definition):
    """(8.3): ``𝔻r → χ(d r)`` — the exact decomposition of 𝔻."""

    name = "(8.3): 𝔻 = χ∘d"
    anchor = BDop

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, BDop)

    def rewrite(self, expr: Expr) -> Expr:
        return ChiSec(BD(expr.r))


class Iota99Definition(Definition):
    """(9.9) — the DEFINITION of ``ι``:
    ``g(χz, φV) → ι_V z`` (either slot order; the metric is
    symmetric)."""

    name = "(9.9): g(χz, φV) = ι_V z"
    anchor = BMetric

    def _parts(self, expr: Expr):
        u, v = expr.u, expr.v
        if isinstance(u, ChiSec) and isinstance(v, PhiSec):
            return v.U, u.z
        if isinstance(v, ChiSec) and isinstance(u, PhiSec):
            return u.U, v.z
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, BMetric)
            and self._parts(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        V, z = self._parts(expr)
        return BIota(V, z)


class IsotropyDeclaration(Definition):
    """DECLARED g-isotropy of ``χ`` (beyond the definitional
    φ-leg): ``g(χz, u) → ι_{ρ(u)} z`` for ``u`` a bracket of
    φ-sections (resolved through the anchor morphism to a Lie
    bracket of M-vectors) and ``g(χz, χy) → 0``."""

    name = "declared g-isotropy: g(χz, u) = ι_ρ(u) z, g(χz, χy) = 0"
    anchor = BMetric

    def _chi_other(self, expr: Expr):
        if isinstance(expr.u, ChiSec):
            return expr.u, expr.v
        if isinstance(expr.v, ChiSec):
            return expr.v, expr.u
        return None

    def _rho(self, u: Expr) -> Optional[Expr]:
        if isinstance(u, PhiSec):
            return u.U
        if isinstance(u, ChiSec):
            return Integer(0)
        if isinstance(u, AlgebroidBracket):
            ru = self._rho(u.u)
            rv = self._rho(u.v)
            # a ZERO leg kills the Lie bracket regardless of the
            # other leg (checked before the None guard — the
            # (χz, χy, w) case has a generic w whose anchor stays
            # symbolic while ρχz = 0 already decides the product)
            if ru == Integer(0) or rv == Integer(0):
                return Integer(0)
            if ru is None or rv is None:
                return None
            # anchor morphism (pre-Bourbaki level)
            return lie_bracket(ru, rv)
        return None

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BMetric):
            return False
        pair = self._chi_other(expr)
        if pair is None:
            return False
        chi, other = pair
        if isinstance(other, PhiSec):
            return False  # (9.9) definitional case handles it
        return self._rho(other) is not None

    def rewrite(self, expr: Expr) -> Expr:
        chi, other = self._chi_other(expr)
        rho = self._rho(other)
        if rho == Integer(0):
            return Integer(0)
        return BIota(rho, chi.z)


class BLieREChiDirectionVanishes(Definition):
    """``ℒ^R`` in a ``ρ∘χ = 0`` direction vanishes: exactness plus
    the ℝ-linearity of ``ℒ^R`` in the direction slot —
    ``ℒ^R_{ρ(χz)} r = ℒ^R_0 r = 0``."""

    name = "exactness: ℒ^R_{ρχz} = 0 (ℝ-linearity in the direction)"
    anchor = BLieRE

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, BLieRE) and isinstance(
            expr.u, ChiSec
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


# ------------------------------------------------------------------ #
# Context + engine                                                   #
# ------------------------------------------------------------------ #


def exact_bourbaki_context(name: str = "E") -> Algebroid:
    """The exact pre-metric-Bourbaki data: an anchored E with the
    ``anchor-morphism`` declaration (pre-Bourbaki level — needed by
    Lemma 9.1 and (9.12))."""
    return algebroid(name, declare=("anchor-morphism",))


def exact_bourbaki_engine(
    alg: Algebroid,
    registry: Optional[PropertyRegistry] = None,
    *,
    isotropic: bool = False,
    invariance: bool = False,
) -> ExpansionEngine:
    """The splitting rules layered over the 7.B.2f.2 structure
    engine. ``invariance`` controls the (7.8) auto-rewrite (kept
    OFF inside the Prop 9.1 provers, which cite the instance
    manually); ``isotropic`` opts into the declared g-isotropy."""
    rules: List[Definition] = [
        SplitAnchorDefinition(),
        BLieREChiDirectionVanishes(),
        SymPartChiSwapDeclaration(),
        LieZDefinitionRule(),
        DopViaChiDefinition(),
        Iota99Definition(),
    ]
    if isotropic:
        rules.append(IsotropyDeclaration())
    eng = ExpansionEngine(rules)
    from jacopy.packages.generalized.bourbaki_precalculus import (
        BDFirstOrderDefinition,
        BIotaBilinearityDefinition,
        BLieZAdditivityDefinition,
        BLieZLawsDefinition,
        BSymbolBilinearityDefinition,
    )

    for extra in (
        BIotaBilinearityDefinition(registry),
        BDFirstOrderDefinition(registry),
        BLieZAdditivityDefinition(registry),
        BLieZLawsDefinition(registry),
        BSymbolBilinearityDefinition(registry),
    ):
        eng.register(extra)
    for d_ in bourbaki_structure_engine(
        alg, registry, invariance=invariance
    ).definitions:
        eng.register(d_)
    return eng


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


def _invariance_instance(
    alg: Algebroid, u: Expr, v: Expr, w: Expr
) -> Expr:
    """The declared (7.8) instance
    ``ℒ^R_u g(v,w) − g([u,v],w) − g(v,[u,w])`` as a zero."""
    br = alg.bracket
    return Sum(
        BLieRE(u, BMetric(v, w)),
        Neg(BMetric(br(u, v), w)),
        Neg(BMetric(v, br(u, w))),
    )


# ------------------------------------------------------------------ #
# Lemma 9.1                                                          #
# ------------------------------------------------------------------ #


def prove_lemma_91(
    alg: Algebroid,
    u: Expr,
    z: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Lemma 9.1: ``[u, χz]_E`` lies in the image of ``χ`` —
    mechanized as ``ρ([u, χz]) = 0`` probed on the generic scalar
    ``h`` (anchor morphism + ``ρχ = 0``)."""
    engine = exact_bourbaki_engine(alg, registry)
    node = Act(
        alg.anchor(alg.bracket(u, ChiSec(z))), h
    )
    nf = _normalize(engine, node, registry)
    if nf != Integer(0):
        raise ProofFailure(
            "Lemma 9.1: ρ([u, χz])(h) does not vanish — "
            + nf._repr_inner()[:140]
        )
    chain = ProofChain(
        [
            ProofStep(
                node,
                Integer(0),
                rule=(
                    "anchor morphism + exactness ρχ = 0 kill "
                    "the anchor of [u, χz]"
                ),
                justification="engine normal form",
                provenance_tag="axiom",
            )
        ]
    )
    theorem = Theorem(
        name=f"bourbaki_lemma91_{alg.name}",
        statement=(
            "ρ([u, χz]_E) = 0 — [u, χz]_E is in the image of χ "
            "(Lemma 9.1; well-definedness of ℒ^Z via (9.7))"
        ),
        lhs=node,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"anchor morphism ({alg.name}, declared — "
            "pre-Bourbaki level)",
            "exactness ρ∘χ = 0 (8.5)",
        ),
        notes="pre-metric-bourbaki.pdf Lemma 9.1",
    )
    return chain, theorem


# ------------------------------------------------------------------ #
# Proposition 9.1                                                    #
# ------------------------------------------------------------------ #


def _cited_zero_proof(
    name: str,
    statement: str,
    target: Expr,
    instance: Expr,
    instance_label: str,
    engine,
    registry,
    *,
    from_axioms,
    notes: str,
) -> Tuple[ProofChain, Theorem]:
    """Close ``target = 0`` by citing ONE declared-zero instance:
    ``NF(target ± instance) = 0`` (the sign is searched; both
    orientations of a zero are zeros)."""
    for sign, tag in (((lambda e: e), "+"), (Neg, "−")):
        nf = _normalize(
            engine, Sum(target, sign(instance)), registry
        )
        if nf == Integer(0):
            chain = ProofChain(
                [
                    ProofStep(
                        target,
                        Integer(0),
                        rule=(
                            f"cite {instance_label} ({tag}) "
                            "and normalize"
                        ),
                        justification=(
                            "adding a declared-zero instance; "
                            "the exact-splitting rules unfold "
                            "the rest to literal 0"
                        ),
                        provenance_tag="axiom",
                    )
                ]
            )
            theorem = Theorem(
                name=name,
                statement=statement,
                lhs=target,
                rhs=Integer(0),
                proof=chain,
                generality="generic-function",
                from_axioms=from_axioms,
                notes=notes,
            )
            return chain, theorem
    nf = _normalize(engine, target, registry)
    raise ProofFailure(
        f"{name}: neither orientation closes — target NF "
        + nf._repr_inner()[:160]
    )


def prove_911(
    alg: Algebroid,
    U: Expr,
    V: Expr,
    z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Prop 9.1, eq (9.11) — the induced quintet satisfies the
    (9.15) axiom:

    ``ι_U(ℒ^Z_V z − dι_V z) + ι_V(ℒ^Z_U z − dι_U z) = 0``

    from the metric-invariance instance at ``(χz, φU, φV)``: its
    left side dies by exactness, and its right side unfolds through
    (6.11) + (9.7) + (9.9) + ``𝔻 = χd`` to the target."""
    engine = exact_bourbaki_engine(alg, registry)
    target = Sum(
        BIota(U, Sum(BLieZ(V, z), Neg(BD(BIota(V, z))))),
        BIota(V, Sum(BLieZ(U, z), Neg(BD(BIota(U, z))))),
    )
    instance = _invariance_instance(
        alg, ChiSec(z), PhiSec(U), PhiSec(V)
    )
    return _cited_zero_proof(
        f"bourbaki_911_{alg.name}",
        "ι_U(ℒ^Z_V z − dι_V z) + ι_V(ℒ^Z_U z − dι_U z) = 0 "
        "(Prop 9.1 (9.11): the induced pre-calculus satisfies "
        "the (9.15) axiom)",
        target,
        instance,
        "the (7.8) invariance instance at (χz, φU, φV)",
        engine,
        registry,
        from_axioms=(
            "(7.8) metric invariance (declared instance)",
            "(6.11) symmetric part (declared, χ-φ orientation)",
            "exactness ρχ = 0 + ℝ-linearity of ℒ^R",
            "(9.7)/(9.9)/(8.3) — definitions of ℒ^Z, ι, 𝔻 = χd",
        ),
        notes="pre-metric-bourbaki.pdf Prop 9.1, first identity",
    )


def prove_912(
    alg: Algebroid,
    U: Expr,
    V: Expr,
    z: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Prop 9.1, eq (9.12) — with g-ISOTROPIC ``χ`` the induced
    quintet satisfies the (9.14) axiom:

    ``ℒ^R_U ι_V z = ι_{[U,V]} z + ι_V ℒ^Z_U z``

    from the metric-invariance instance at ``(φU, φV, χz)``."""
    engine = exact_bourbaki_engine(
        alg, registry, isotropic=True
    )
    target = Sum(
        BLieRE(PhiSec(U), BIota(V, z)),
        Neg(BIota(lie_bracket(U, V), z)),
        Neg(BIota(V, BLieZ(U, z))),
    )
    instance = _invariance_instance(
        alg, PhiSec(U), PhiSec(V), ChiSec(z)
    )
    return _cited_zero_proof(
        f"bourbaki_912_{alg.name}",
        "ℒ^R_U ι_V z = ι_{[U,V]} z + ι_V ℒ^Z_U z "
        "(Prop 9.1 (9.12): the induced pre-calculus satisfies "
        "the (9.14) axiom — needs the DECLARED g-isotropy of χ)",
        target,
        instance,
        "the (7.8) invariance instance at (φU, φV, χz)",
        engine,
        registry,
        from_axioms=(
            "(7.8) metric invariance (declared instance)",
            "declared g-isotropy of χ",
            f"anchor morphism ({alg.name}, declared)",
            "(9.7)/(9.9) — definitions of ℒ^Z, ι",
        ),
        notes="pre-metric-bourbaki.pdf Prop 9.1, second identity",
    )
