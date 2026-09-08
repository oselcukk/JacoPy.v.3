"""
The Bourbaki pre-calculus quintet (Phase 7.B.2f.1; PDF item 10h /
7.B Bourbaki targets), from the PRIMARY paper pre-metric-bourbaki.pdf
[arXiv:2210.00548] — Definition 9.1 and Theorem 9.1, mechanized.

**Definition 9.1**: on a pair of vector bundles ``(R, Z)`` the
quintet ``(ι, d, l, ℒ^R, ℒ^Z)`` is a Bourbaki pre-calculus if

* ``ι : 𝔛(M) × 𝔛(Z) → 𝔛(R)`` is C∞-bilinear,
* ``d : 𝔛(R) → 𝔛(Z)`` is first-order with symbol ``l``
  (``d(f·r) = f·dr + l_{df} r``),
* ``ℒ^R : 𝔛(M) × 𝔛(R) → 𝔛(R)`` is ℝ-bilinear,
* ``ℒ^Z : 𝔛(M) × 𝔛(Z) → 𝔛(Z)`` is first-order with
  (1) ``ℒ^Z_V(f·z) = f·ℒ^Z_V z + V(f)·z`` and
  (2) ``ℒ^Z_{fV} z = f·ℒ^Z_V z + l_{df} ι_V z``,

satisfying the two AXIOMS

    (9.14)  ℒ^R_U ι_V z = ι_{[U,V]} z + ι_V ℒ^Z_U z,
    (9.15)  ι_U(ℒ^Z_V z − dι_V z) = −ι_V(ℒ^Z_U z − dι_U z).

**Theorem 9.1** (the standard construction): a Bourbaki pre-calculus
makes ``T(M) ⊕ Z`` an exact pre-metric-Bourbaki algebroid with the
STANDARD bracket and R-valued metric

    [U+z, V+y]_S = [U,V] + ℒ^Z_U y − ℒ^Z_V z + dι_V z    (9.17)
    g_S(U+z, V+y) = ι_U y + ι_V z                         (9.18)

— proven here by the paper's three checks: the symmetric part is
``d g_S`` (mechanical), and the metric invariance (7.8) follows from
(9.14) + (9.15) (axiom citations). **Definition 9.2's remark** — the
usual Cartan calculus IS a Bourbaki pre-calculus with
``(ι, d, ω∧·, ℒ, ℒ)`` — is verified concretely: (9.14) is the
``[ℒ,ι]`` commutator, (9.15) is Cartan magic + ``ι² = 0``, and the
ℒ^Z laws are (2.10). The abstract layer NEVER identifies R or Z
with form bundles — the audit's 10h scope point (the general
R-valued structure, not its scalar shadow).
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
from jacopy.central.objects.bundle import Bundle
from jacopy.central.objects.vector_field import vector_fields
from jacopy.central.tangent.lie_bracket import lie_bracket


# ------------------------------------------------------------------ #
# The quintet's operator nodes (opaque, slot-protocol atoms)          #
# ------------------------------------------------------------------ #


class _SlotAtom(Derivation):
    """Shared base: a named operator atom over labelled Expr slots
    (degree-0; the slot protocol makes the engine walk inside)."""

    __slots__ = ("_slots",)

    def __init__(self, display: str, *slots: Expr) -> None:
        for s in slots:
            if not isinstance(s, Expr):
                raise TypeError(
                    f"{type(self).__name__} requires Expr slots"
                )
        super().__init__(display, degree=0)
        self._slots = tuple(slots)

    @property
    def rewritable_slots(self):
        return self._slots

    def _key(self) -> Any:
        return (type(self).__name__, self._slots)


class BIota(_SlotAtom):
    """``ι_U z ∈ 𝔛(R)`` — the quintet's interior."""

    def __init__(self, U: Expr, z: Expr) -> None:
        super().__init__(
            f"ι_{U._repr_inner()}({z._repr_inner()})", U, z
        )

    @property
    def U(self) -> Expr:
        return self._slots[0]

    @property
    def z(self) -> Expr:
        return self._slots[1]

    def with_slots(self, U: Expr, z: Expr) -> "BIota":
        return BIota(U, z)


class BD(_SlotAtom):
    """``d r ∈ 𝔛(Z)`` — the quintet's first-order differential."""

    def __init__(self, r: Expr) -> None:
        super().__init__(f"d({r._repr_inner()})", r)

    @property
    def r(self) -> Expr:
        return self._slots[0]

    def with_slots(self, r: Expr) -> "BD":
        return BD(r)


class BSymbol(_SlotAtom):
    """``l_{df} r ∈ 𝔛(Z)`` — the symbol leg of ``d`` (recorded with
    the generating scalar ``f``); C∞-bilinear."""

    def __init__(self, f: Expr, r: Expr) -> None:
        super().__init__(
            f"l_d{f._repr_inner()}({r._repr_inner()})", f, r
        )

    @property
    def f(self) -> Expr:
        return self._slots[0]

    @property
    def r(self) -> Expr:
        return self._slots[1]

    def with_slots(self, f: Expr, r: Expr) -> "BSymbol":
        return BSymbol(f, r)


class BLieR(_SlotAtom):
    """``ℒ^R_U r ∈ 𝔛(R)`` — ℝ-bilinear ONLY (no C∞ laws; that is
    the point of the abstract layer)."""

    def __init__(self, U: Expr, r: Expr) -> None:
        super().__init__(
            f"L^R_{U._repr_inner()}({r._repr_inner()})", U, r
        )

    @property
    def U(self) -> Expr:
        return self._slots[0]

    @property
    def r(self) -> Expr:
        return self._slots[1]

    def with_slots(self, U: Expr, r: Expr) -> "BLieR":
        return BLieR(U, r)


class BLieZ(_SlotAtom):
    """``ℒ^Z_V z ∈ 𝔛(Z)`` — first-order, laws (1)-(2) of Def 9.1."""

    def __init__(self, V: Expr, z: Expr) -> None:
        super().__init__(
            f"L^Z_{V._repr_inner()}({z._repr_inner()})", V, z
        )

    @property
    def V(self) -> Expr:
        return self._slots[0]

    @property
    def z(self) -> Expr:
        return self._slots[1]

    def with_slots(self, V: Expr, z: Expr) -> "BLieZ":
        return BLieZ(V, z)


# ------------------------------------------------------------------ #
# Definitional rules of the quintet                                  #
# ------------------------------------------------------------------ #


def _split_scalar(expr: Expr, registry):
    if isinstance(expr, Product) and len(expr.children) >= 2:
        head = expr.children[0]
        if is_scalar_function(head, registry):
            rest = expr.children[1:]
            return head, (
                rest[0] if len(rest) == 1 else Product(*rest)
            )
    return None


class _BilinearSlots(Definition):
    """Shared Sum/Neg/zero/scalar distribution over an atom's slots
    (which slots admit the SCALAR pull-out is per-node: ℒ^R is only
    ℝ-bilinear, so its scalar case stays inert)."""

    node_cls: type = _SlotAtom
    scalar_slots: Tuple[int, ...] = ()

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, self.node_cls):
            return False
        for i, s in enumerate(expr.rewritable_slots):
            if isinstance(s, (Sum, Neg)) or s == Integer(0):
                return True
            if i in self.scalar_slots and _split_scalar(
                s, self._registry
            ):
                return True
        return False

    def rewrite(self, expr: Expr) -> Expr:
        slots = list(expr.rewritable_slots)
        for i, s in enumerate(slots):
            def rebuilt(new_s: Expr) -> Expr:
                new = list(slots)
                new[i] = new_s
                return expr.with_slots(*new)

            if s == Integer(0):
                return Integer(0)
            if isinstance(s, Sum):
                return Sum(*(rebuilt(c) for c in s.children))
            if isinstance(s, Neg):
                return Neg(rebuilt(s.arg))
            if i in self.scalar_slots:
                split = _split_scalar(s, self._registry)
                if split:
                    f, rest = split
                    return Product(f, rebuilt(rest))
        raise AssertionError  # pragma: no cover


class BIotaBilinearityDefinition(_BilinearSlots):
    node_cls = BIota
    scalar_slots = (0, 1)
    name = "ι C∞-bilinearity (Def 9.1)"

    anchor = BIota


class BSymbolBilinearityDefinition(_BilinearSlots):
    node_cls = BSymbol
    scalar_slots = (1,)
    name = "l C∞-bilinearity (Def 9.1)"

    anchor = BSymbol


class BLieRAdditivityDefinition(_BilinearSlots):
    node_cls = BLieR
    scalar_slots = ()  # ℝ-bilinear ONLY
    name = "ℒ^R ℝ-bilinearity (Def 9.1 — no C∞ law)"

    anchor = BLieR


class BLieZAdditivityDefinition(_BilinearSlots):
    node_cls = BLieZ
    scalar_slots = ()  # the C∞ behaviour is laws (1)-(2) below
    name = "ℒ^Z ℝ-additivity (first-order operator)"

    anchor = BLieZ


class BDFirstOrderDefinition(Definition):
    """``d(f·r) = f·dr + l_{df} r`` (+ ℝ-additivity) — ``d`` is
    first-order with symbol ``l`` (Def 9.1)."""

    name = "d first-order with symbol l: d(f·r) = f·dr + l_df r"
    anchor = BD

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BD):
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
            return Sum(*(BD(c) for c in r.children))
        if isinstance(r, Neg):
            return Neg(BD(r.arg))
        f, rest = _split_scalar(r, self._registry)
        return Sum(Product(f, BD(rest)), BSymbol(f, rest))


class BLieZLawsDefinition(Definition):
    """Laws (1)-(2) of Def 9.1:
    ``ℒ^Z_V(f·z) = f·ℒ^Z_V z + V(f)·z`` and
    ``ℒ^Z_{fV} z = f·ℒ^Z_V z + l_{df} ι_V z``."""

    name = (
        "ℒ^Z first-order laws: ℒ^Z_V(fz) = fℒ^Z_Vz + V(f)z, "
        "ℒ^Z_{fV}z = fℒ^Z_Vz + l_df ι_Vz"
    )
    anchor = BLieZ

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BLieZ):
            return False
        return (
            _split_scalar(expr.z, self._registry) is not None
            or _split_scalar(expr.V, self._registry)
            is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        split_z = _split_scalar(expr.z, self._registry)
        if split_z is not None:
            f, z = split_z
            return Sum(
                Product(f, BLieZ(expr.V, z)),
                Product(Act(expr.V, f), z),
            )
        f, V = _split_scalar(expr.V, self._registry)
        return Sum(
            Product(f, BLieZ(V, expr.z)),
            BSymbol(f, BIota(V, expr.z)),
        )


class Axiom914Declaration(Definition):
    """DECLARED axiom (9.14):
    ``ℒ^R_U ι_V z → ι_{[U,V]} z + ι_V ℒ^Z_U z`` — terminating
    (the ℒ^R node disappears)."""

    name = "(9.14): ℒ^R_U ι_V z = ι_[U,V] z + ι_V ℒ^Z_U z"
    anchor = BLieR

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, BLieR) and isinstance(
            expr.r, BIota
        )

    def rewrite(self, expr: Expr) -> Expr:
        inner = expr.r
        return Sum(
            BIota(lie_bracket(expr.U, inner.U), inner.z),
            BIota(inner.U, BLieZ(expr.U, inner.z)),
        )


def axiom_915_instance(U: Expr, V: Expr, z: Expr) -> Expr:
    """The (9.15) axiom instance as a DECLARED ZERO:
    ``ι_U(ℒ^Z_V z − dι_V z) + ι_V(ℒ^Z_U z − dι_U z)``."""
    return Sum(
        BIota(U, Sum(BLieZ(V, z), Neg(BD(BIota(V, z))))),
        BIota(V, Sum(BLieZ(U, z), Neg(BD(BIota(U, z))))),
    )


# ------------------------------------------------------------------ #
# Context + engine                                                   #
# ------------------------------------------------------------------ #


class BourbakiPreCalculus:
    """The abstract quintet context on a bundle pair ``(R, Z)`` —
    sections are opaque; the ONLY structure is Def 9.1."""

    def __init__(self, name: str = "B") -> None:
        self._name = name
        self._R = Bundle(f"{name}-R")
        self._Z = Bundle(f"{name}-Z")

    @property
    def name(self) -> str:
        return self._name

    def r_sections(self, names: str):
        return vector_fields(names, bundle=self._R)

    def z_sections(self, names: str):
        return vector_fields(names, bundle=self._Z)


def bourbaki_engine(
    registry: Optional[PropertyRegistry] = None,
) -> ExpansionEngine:
    """Definitional rules of the quintet + the declared (9.14),
    layered on the tangent engine (for ``[U,V]`` bookkeeping)."""
    from jacopy.central.tangent.engine import tangent_engine

    eng = ExpansionEngine(
        [
            Axiom914Declaration(),
            BIotaBilinearityDefinition(registry),
            BSymbolBilinearityDefinition(registry),
            BLieRAdditivityDefinition(registry),
            BLieZAdditivityDefinition(registry),
            BDFirstOrderDefinition(registry),
            BLieZLawsDefinition(registry),
        ]
    )
    for d_ in tangent_engine(registry=registry).definitions:
        eng.register(d_)
    return eng


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


# ------------------------------------------------------------------ #
# Theorem 9.1 — the standard construction                            #
# ------------------------------------------------------------------ #


def standard_bracket(
    U: Expr, z: Expr, V: Expr, y: Expr
) -> Tuple[Expr, Expr]:
    """(9.17): ``[U+z, V+y]_S = [U,V] ⊕ (ℒ^Z_U y − ℒ^Z_V z +
    dι_V z)`` as a ``(vector, Z)`` component pair."""
    return (
        lie_bracket(U, V),
        Sum(
            BLieZ(U, y),
            Neg(BLieZ(V, z)),
            BD(BIota(V, z)),
        ),
    )


def standard_metric(
    U: Expr, z: Expr, V: Expr, y: Expr
) -> Expr:
    """(9.18): ``g_S(U+z, V+y) = ι_U y + ι_V z ∈ 𝔛(R)``."""
    return Sum(BIota(U, y), BIota(V, z))


def prove_standard_symmetric_part(
    U: Expr,
    z: Expr,
    V: Expr,
    y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Thm 9.1, check 1: ``[U+z,V+y]_S + [V+y,U+z]_S = d g_S`` —
    the ℒ^Z terms cancel pairwise and ``d``'s additivity collects
    the rest; the vector side is the Lie antisymmetry."""
    engine = bourbaki_engine(registry)
    v12, z12 = standard_bracket(U, z, V, y)
    v21, z21 = standard_bracket(V, y, U, z)
    diffs = [
        Sum(v12, v21),
        Sum(
            z12,
            z21,
            Neg(BD(standard_metric(U, z, V, y))),
        ),
    ]
    steps: List[ProofStep] = []
    labels = (
        "vector component normalizes to 0 (Lie antisymmetry)",
        "Z component normalizes to 0 (d-additivity)",
    )
    for label, diff in zip(labels, diffs):
        nf = _normalize(engine, diff, registry)
        if nf != Integer(0):
            raise ProofFailure(
                f"Thm 9.1 symmetric part: {label} FAILS — "
                + nf._repr_inner()[:140]
            )
        steps.append(
            ProofStep(
                diff, Integer(0), rule=label,
                justification="engine normal form",
            )
        )
    chain = ProofChain(steps)
    theorem = Theorem(
        name="bourbaki_standard_symmetric_part",
        statement=(
            "[U+z,V+y]_S + [V+y,U+z]_S = d g_S(U+z,V+y) on "
            "T(M) ⊕ Z (Thm 9.1, check 1)"
        ),
        lhs=diffs[1],
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "standard bracket/metric definitions (9.17)-(9.18)",
            "d ℝ-additivity (first-order operator)",
        ),
        notes="pre-metric-bourbaki.pdf Thm 9.1",
    )
    return chain, theorem


def prove_standard_metric_invariance(
    U: Expr,
    z: Expr,
    V: Expr,
    y: Expr,
    W: Expr,
    x: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """Thm 9.1, check 2: the metric invariance (7.8) of the
    standard construction,

    ``ℒ^R_U g_S(V+y, W+x) = g_S([U+z,V+y]_S, W+x)
                          + g_S(V+y, [U+z,W+x]_S)``

    — (9.14) opens the left side; the difference is exactly a
    combination of (9.15) instances, cited as declared zeros."""
    engine = bourbaki_engine(registry)
    lhs = BLieR(U, standard_metric(V, y, W, x))
    bv1, bz1 = standard_bracket(U, z, V, y)
    bv2, bz2 = standard_bracket(U, z, W, x)
    rhs = Sum(
        standard_metric(bv1, bz1, W, x),
        standard_metric(V, y, bv2, bz2),
    )
    diff = Sum(lhs, Neg(rhs))
    residual = _normalize(engine, diff, registry)
    steps: List[ProofStep] = [
        ProofStep(
            diff,
            residual,
            rule="normalize ((9.14) opens ℒ^R; bilinearity)",
            justification="registered rules + declared (9.14)",
            provenance_tag="axiom",
        )
    ]
    if residual != Integer(0):
        from jacopy.packages.poisson.koszul_jacobi import (
            _node_size,
        )

        seeds = []
        for (A, B, s) in (
            (U, V, z),
            (U, W, z),
            (U, V, y),
            (U, W, x),
            (V, W, z),
        ):
            inst = axiom_915_instance(A, B, s)
            for sgn in ((lambda e: e), Neg):
                nf = _normalize(engine, sgn(inst), registry)
                if nf != Integer(0):
                    seeds.append(nf)
        for _round in range(12):
            if residual == Integer(0):
                break
            progressed = False
            for seed in seeds:
                cand = _normalize(
                    engine,
                    Sum(residual, Neg(seed)),
                    registry,
                )
                if _node_size(cand) < _node_size(residual):
                    steps.append(
                        ProofStep(
                            residual,
                            cand,
                            rule="cite declared axiom (9.15)",
                            justification=(
                                "subtracting a declared zero "
                                "instance"
                            ),
                            provenance_tag="axiom",
                        )
                    )
                    residual = cand
                    progressed = True
                    break
            if not progressed:
                break
    if residual != Integer(0):
        raise ProofFailure(
            "Thm 9.1 metric invariance: residual survives — "
            + residual._repr_inner()[:160]
        )
    chain = ProofChain(steps)
    theorem = Theorem(
        name="bourbaki_standard_metric_invariance",
        statement=(
            "ℒ^R_U g_S(V+y,W+x) = g_S([U+z,V+y]_S, W+x) + "
            "g_S(V+y, [U+z,W+x]_S) on T(M) ⊕ Z (Thm 9.1, "
            "check 2 — from the declared (9.14) + (9.15))"
        ),
        lhs=diff,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "declared (9.14)",
            "declared (9.15) (cited instances)",
            "quintet definitional rules (Def 9.1)",
        ),
        notes="pre-metric-bourbaki.pdf Thm 9.1",
    )
    return chain, theorem


# ------------------------------------------------------------------ #
# Definition 9.2 remark — Cartan calculus IS a Bourbaki pre-calculus #
# ------------------------------------------------------------------ #


def prove_cartan_satisfies_914(
    U: Expr,
    V: Expr,
    omega: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """(9.14) for the USUAL Cartan quintet
    ``(ι, d, ω∧·, ℒ, ℒ)``: ``ℒ_U ι_V ω = ι_{[U,V]} ω + ι_V ℒ_U ω``
    — the ``[ℒ, ι]`` commutator, evaluated on the probe slots."""
    from jacopy.central.tangent.cartan import L as _L
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.central.objects.interior import Interior
    from jacopy.packages.drinfeld.double import _ev
    from jacopy.proof.strategies import ExpandAndSimplify

    def iota(X, a):
        return Act(Interior(X), a)

    lhs = _L(U, iota(V, omega))
    rhs = Sum(
        iota(lie_bracket(U, V), omega),
        iota(V, _L(U, omega)),
    )
    node = _ev(Sum(lhs, Neg(rhs)), slots)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=tangent_engine(registry=registry),
    )


def prove_cartan_satisfies_915(
    U: Expr,
    V: Expr,
    omega: Expr,
    slots,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """(9.15) for the USUAL Cartan quintet:
    ``ι_U(ℒ_V ω − dι_V ω) + ι_V(ℒ_U ω − dι_U ω) = 0`` — Cartan
    magic turns each parenthesis into ``ι dω``, and ``ι² = 0``
    (alternation) kills the symmetric sum."""
    from jacopy.central.tangent.cartan import L as _L
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.central.tangent.exterior import d
    from jacopy.central.objects.interior import Interior
    from jacopy.packages.drinfeld.double import _ev
    from jacopy.proof.strategies import ExpandAndSimplify

    def iota(X, a):
        return Act(Interior(X), a)

    node = _ev(
        Sum(
            iota(U, Sum(_L(V, omega), Neg(d(iota(V, omega))))),
            iota(V, Sum(_L(U, omega), Neg(d(iota(U, omega))))),
        ),
        slots,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=tangent_engine(registry=registry),
    )
