"""
The cotangent-algebroid showcase (Phase 5.C) — PDF item 12h.

``(T*M, π♯, [·,·]_π)`` is a Lie algebroid — and in v3 this is a
PROVED statement, not a declared one: the axioms the Phase 3
declaration system treats as opt-in assumptions close mechanically
for the concrete Koszul structure:

* **right-Leibniz** ``[α, f·β]_π = f·[α,β]_π + (π♯α)(f)·β``
  (evaluation-level, generic vector field) — no assumption at all;
* **Poisson Jacobi** ``{f,{g,h}} + {g,{h,f}} + {h,{f,g}} = 0`` —
  under the DECLARED ``[π,π]_SN = 0``, via the two-leg tactic on the
  SN evaluation (honest failure without the declaration);
* **the Hamiltonian anchor-morphism** ``X_{{f,g}} = [X_f, X_g]`` —
  the anchor property on exact generators, again under the
  declaration (agreement on generators).

The one new definitional ingredient is the EVALUATION VIEW of the
Schouten bracket of an atomic bivector against exact 1-forms
(:class:`SNBivectorExactEvalDefinition`):

    [π, π](df, dg, dh) = 2·[ π(df, d π(dg, dh))
                           + π(dg, d π(dh, df))
                           + π(dh, d π(df, dg)) ],

the arity-level counterpart of the Palais formula for ``d`` — for an
ATOMIC bivector there is nothing more primitive to expand into. It is
guarded to exact arguments; the general-1-form version carries extra
``dα`` terms and arrives with the graded (5.E) layer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.brackets.base import BracketApply
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.schouten import SN
from jacopy.packages.poisson.core import (
    PoissonStructure,
    poisson_engine,
)
from jacopy.packages.poisson.koszul import (
    KoszulBracketDefinition,
    SharpVFLinearityDefinition,
    koszul_bracket,
)


def _is_exact(expr: Expr) -> bool:
    from jacopy.central.calculus.bracket_calculus import (
        ExteriorDerivative,
    )

    return (
        isinstance(expr, Act)
        and isinstance(expr.op, ExteriorDerivative)
        and expr.op.calculus_name == CARTAN_TM.name
    )


class SNBivectorExactEvalDefinition(Definition):
    """``[π,π](df, dg, dh) → 2·Σ_cyc π(df, d π(dg, dh))`` — the
    evaluation view of the Schouten square of an atomic bivector on
    EXACT 1-forms (guarded; the general version has extra ``dα``
    terms, Phase 5.E)."""

    anchor = MultiEval

    def __init__(self, structure: PoissonStructure) -> None:
        self._structure = structure
        self.name = (
            f"SN evaluation ({structure.pi._repr_inner()}): "
            "[π,π](df,dg,dh) = 2Σ_cyc π(df, dπ(dg,dh))"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        head = expr.head
        if not (
            isinstance(head, BracketApply)
            and head.bracket is SN
            and head.a == self._structure.pi
            and head.b == self._structure.pi
        ):
            return False
        return expr.arity == 3 and all(
            _is_exact(a) for a in expr.args
        )

    def rewrite(self, expr: Expr) -> Expr:
        pi = self._structure.pi
        a, b, c = expr.args

        def ev(x: Expr, y: Expr) -> Expr:
            return MultiEval(
                pi, x, y, alternating=True, slot_kind="covector"
            )

        return Product(
            Integer(2),
            Sum(
                ev(a, d(ev(b, c))),
                ev(b, d(ev(c, a))),
                ev(c, d(ev(a, b))),
            ),
        )


class SNBivectorGeneralEvalDefinition(Definition):
    """``[π,π](α, β, γ) → 2(⟨γ, [π♯α, π♯β]_VF⟩ − ⟨γ, π♯([α,β]_π)⟩)``
    — the evaluation view of the Schouten square on GENERAL 1-forms:
    the anchor-morphism-defect form (Phase 5.E.2). The defect
    ``D(α,β) = [π♯α, π♯β] − π♯[α,β]_π`` is tensorial and
    ``⟨γ, D(α,β)⟩ = ½[π,π](α,β,γ)``.

    Complementary to :class:`SNBivectorExactEvalDefinition`: the
    all-exact triple keeps the ``Σ_cyc`` shape (existing normal forms
    and cited-theorem lhs unchanged); this rule fires only when at
    least one argument is NOT exact. Consistency of the two views on
    exact arguments is a mechanical theorem
    (:func:`prove_sn_eval_views_agree_on_exacts`)."""

    anchor = MultiEval

    def __init__(self, structure: PoissonStructure) -> None:
        self._structure = structure
        self.name = (
            f"SN evaluation, general ({structure.pi._repr_inner()}): "
            "[π,π](α,β,γ) = 2⟨γ, [π♯α,π♯β] − π♯[α,β]_π⟩"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        head = expr.head
        if not (
            isinstance(head, BracketApply)
            and head.bracket is SN
            and head.a == self._structure.pi
            and head.b == self._structure.pi
        ):
            return False
        return expr.arity == 3 and not all(
            _is_exact(a) for a in expr.args
        )

    def rewrite(self, expr: Expr) -> Expr:
        # The Leibniz-expanded equivalent of the defect pairing —
        # ⟨γ,[V,W]⟩ = V⟨γ,W⟩ − ⟨L_V γ, W⟩ with V = π♯α, W = π♯β —
        # so every piece lives in the π-evaluation vocabulary the
        # Koszul expansions produce (a stuck ⟨γ,[π♯α,π♯β]⟩ pairing
        # would never meet its counterpart):
        #
        #   ½[π,π](α,β,γ) = (π♯α)(π(β,γ)) − π(β, L_{π♯α}γ)
        #                   − π([α,β]_π, γ).
        #
        # Deliberately NOT cyclically symmetrized: the symmetrized
        # form was tried (5.E.2b) and rejected — the three cyclic
        # legs are not mechanically inter-derivable (a collection
        # obstruction), which breaks the general id2/id3 instance
        # normal forms while not closing the general Jacobi either.
        from jacopy.central.tangent.exterior import CARTAN_TM as _C
        from jacopy.packages.poisson.core import SharpVF
        from jacopy.packages.poisson.koszul import KoszulBracket

        pi = self._structure.pi
        a, b, c = expr.args

        def ev(x: Expr, y: Expr) -> Expr:
            return MultiEval(
                pi, x, y, alternating=True, slot_kind="covector"
            )

        return Product(
            Integer(2),
            Sum(
                Act(SharpVF(pi, a), ev(b, c)),
                Neg(ev(b, Act(_C.lie(SharpVF(pi, a)), c))),
                Neg(ev(KoszulBracket(pi, a, b), c)),
            ),
        )


def _contains_sharp(expr: Expr, P: PoissonStructure) -> bool:
    from jacopy.packages.poisson.core import SharpVF

    if isinstance(expr, SharpVF) and expr.pi == P.pi:
        return True
    for c in expr.children:
        if _contains_sharp(c, P):
            return True
    slots = getattr(expr, "rewritable_slots", None)
    if slots:
        for s in slots:
            if _contains_sharp(s, P):
                return True
    op = getattr(expr, "op", None)
    if isinstance(op, Expr) and _contains_sharp(op, P):
        return True
    return False


class SharpPairingLinearityDefinition(Definition):
    """Pairing linearity in the VECTOR slot, guarded to slots whose
    tree contains a sharp vector field of this structure:
    ``⟨α, Σᵢ sᵢ·Vᵢ⟩ → Σᵢ sᵢ·⟨α, Vᵢ⟩``.

    The guard is what prevents ping-pong with the pairing-collection
    phase of ``simplify``: once the sharp evaluations consume the
    ``π♯``-cored pairings, regrouped leftovers no longer contain a
    sharp and the split stays quiet."""

    name = "sharp pairing linearity: ⟨α, Σ sᵢVᵢ⟩ = Σ sᵢ⟨α, Vᵢ⟩ (π♯ in the slot)"
    anchor = Pairing

    def __init__(
        self,
        structure: PoissonStructure,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._structure = structure
        self._registry = registry

    def _scalar_split(self, term: Expr):
        from jacopy.central.calculus.scalars import is_scalar_function

        sign = False
        core = term
        if isinstance(core, Neg):
            sign = True
            core = core.arg
        if isinstance(core, Product) and len(core.children) >= 2:
            scalars = [
                c
                for c in core.children
                if is_scalar_function(c, self._registry)
            ]
            rest = [
                c
                for c in core.children
                if not is_scalar_function(c, self._registry)
            ]
            if scalars and len(rest) == 1:
                return sign, scalars, rest[0]
        return sign, [], core

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        X = expr.X
        if not _contains_sharp(X, self._structure):
            return False
        if isinstance(X, (Sum, Neg)):
            return True
        _, scalars, _ = self._scalar_split(X)
        return bool(scalars)

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.X
        if isinstance(X, Sum):
            return Sum(
                *(Pairing(expr.alpha, t) for t in X.children)
            )
        if isinstance(X, Neg):
            return Neg(Pairing(expr.alpha, X.arg))
        sign, scalars, core = self._scalar_split(X)
        out: Expr = Product(*scalars, Pairing(expr.alpha, core))
        return Neg(out) if sign else out


def showcase_engine(
    P: PoissonStructure,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
):
    """The 5.C engine: the Poisson/Koszul layer plus the bracket
    module structure and head-scalar pairing linearity (both needed by
    the evaluation-level algebroid-axiom proofs). ``declare_poisson``
    controls the ``[π,π]_SN = 0`` declaration."""
    from jacopy.central.calculus import HeadScalarDefinition
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )

    engine = poisson_engine(
        registry=registry, structures=(P,) if declare_poisson else ()
    )
    if not declare_poisson:
        # definitional Poisson layer without the [π,π] declaration
        from jacopy.packages.poisson.core import (
            HamiltonianActionDefinition,
            PoissonBracketDefinition,
            SharpActionDefinition,
            SharpEvaluationDefinition,
        )

        engine.register(PoissonBracketDefinition(P))
        engine.register(HamiltonianActionDefinition(P, registry))
        engine.register(SharpEvaluationDefinition(P))
        engine.register(SharpActionDefinition(P, registry))
    engine.register(KoszulBracketDefinition(P))
    engine.register(SharpVFLinearityDefinition(P, registry))
    engine.register(SNBivectorExactEvalDefinition(P))
    engine.register(SNBivectorGeneralEvalDefinition(P))
    engine.register(SharpPairingLinearityDefinition(P, registry))
    engine.register(HeadScalarDefinition(registry))
    engine.register(LieBracketLeibnizDefinition(registry))
    return engine


# --------------------------------------------------------------------- #
# The algebroid axioms, proved                                           #
# --------------------------------------------------------------------- #


def prove_koszul_right_leibniz(
    P: PoissonStructure,
    alpha: Expr,
    beta: Expr,
    f: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨[α, f·β]_π, Y⟩ = ⟨f·[α,β]_π + (π♯α)(f)·β, Y⟩`` — the
    algebroid RIGHT-LEIBNIZ axiom holds for the Koszul bracket,
    mechanically and with NO assumption (evaluated against the generic
    vector field ``Y``)."""
    lhs = Pairing(koszul_bracket(P, alpha, Product(f, beta)), Y)
    rhs = Pairing(
        Sum(
            Product(f, koszul_bracket(P, alpha, beta)),
            Product(Act(P.sharp_vf(alpha), f), beta),
        ),
        Y,
    )
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=showcase_engine(P, registry=registry),
    )


def _cyclic_jacobi_sum(P: PoissonStructure, f, g, h) -> Expr:
    return Sum(
        P.bracket(f, P.bracket(g, h)),
        P.bracket(g, P.bracket(h, f)),
        P.bracket(h, P.bracket(f, g)),
    )


def prove_poisson_jacobi(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, "object"]:
    """``{f,{g,h}} + {g,{h,f}} + {h,{f,g}} = 0`` — under the DECLARED
    ``[π,π]_SN = 0`` (the two-leg tactic on the SN evaluation):

    1. ``[π,π](df,dg,dh) = 2·Σ_cyc {f,{g,h}}`` (the evaluation view,
       declaration withheld),
    2. ``[π,π](df,dg,dh) = 0`` (the declaration),
    3. hence the cyclic sum vanishes.

    Returns ``(chain, theorem)`` — the instance Theorem is citable.
    """
    from jacopy.proof.theorems import Theorem
    from jacopy.central.tangent.schouten import sn_bracket

    sn_eval = MultiEval(
        sn_bracket(P.pi, P.pi),
        d(f),
        d(g),
        d(h),
        alternating=True,
        slot_kind="covector",
    )

    # Leg 1: the evaluation view (declaration withheld) — the sum of
    # nested brackets appears.
    eng_no_decl = showcase_engine(
        P, registry=registry, declare_poisson=False
    )
    target = Product(Integer(2), _cyclic_jacobi_sum(P, f, g, h))
    leg1 = ExpandAndSimplify().prove(
        sn_eval,
        target,
        registry=registry,
        engine=eng_no_decl,
    )
    step1 = ProofStep(
        sn_eval,
        target,
        rule="SN evaluation on exact forms",
        justification="[π,π](df,dg,dh) = 2·Σ_cyc {f,{g,h}}",
    )
    for s in leg1:
        step1.add_child(s)

    # Leg 2: the declaration kills the head.
    eng_decl = showcase_engine(P, registry=registry)
    leg2 = ExpandAndSimplify().prove(
        sn_eval,
        Integer(0),
        registry=registry,
        engine=eng_decl,
    )
    step2 = ProofStep(
        sn_eval,
        Integer(0),
        rule="declared Poisson structure",
        justification="[π,π]_SN = 0, so its evaluation vanishes",
    )
    for s in leg2:
        step2.add_child(s)

    step3 = ProofStep(
        _cyclic_jacobi_sum(P, f, g, h),
        Integer(0),
        rule="combine the two legs",
        justification=(
            "2·Σ_cyc {f,{g,h}} = [π,π](df,dg,dh) = 0; divide by 2"
        ),
    )
    chain = ProofChain([step1, step2, step3])
    names = tuple(x._repr_inner() for x in (f, g, h))
    # The citable lhs must match the CITING engine's normal form
    # (the π-evaluation shape), not the pretty bracket-value nodes.
    from jacopy.algorithms.product_rule import product_rule
    from jacopy.algorithms.simplify import simplify

    normalized = _cyclic_jacobi_sum(P, f, g, h)
    for _ in range(8):
        expanded, _steps = eng_no_decl.expand(normalized)
        after = product_rule(expanded, registry)
        reduced = simplify(after, registry)
        if reduced == normalized:
            break
        normalized = reduced
    theorem = Theorem(
        name="poisson_jacobi_"
        + P.pi._repr_inner()
        + "_"
        + "_".join(names),
        statement=(
            "{%s,{%s,%s}} + {%s,{%s,%s}} + {%s,{%s,%s}} = 0"
            % (
                names[0], names[1], names[2],
                names[1], names[2], names[0],
                names[2], names[0], names[1],
            )
        ),
        lhs=normalized,
        rhs=Integer(0),
        proof=chain,
        generality="instance",
        from_axioms=(
            f"Poisson ({P.pi._repr_inner()}): [π, π]_SN = 0",
        ),
        notes="Jacobi ⟺ [π,π]_SN = 0, mechanized on the generators",
    )
    return chain, theorem


def prove_sn_eval_views_agree_on_exacts(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """The two evaluation views of the Schouten square agree on exact
    forms (Phase 5.E.2 consistency backing for the general rule):

        2⟨dh, [π♯df, π♯dg] − π♯[df,dg]_π⟩
            = 2·Σ_cyc π(df, d π(dg, dh)).

    Proved mechanically with the declaration WITHHELD (neither view
    is assumed; the ``L_X(df) → d(X(f))`` derived rule is opted in,
    Phase 2 theorem)."""
    from jacopy.algebra.lie_bracket_vf import LieBracketVF
    from jacopy.central.tangent.exterior import d as d_
    from jacopy.packages.poisson.core import SharpVF
    from jacopy.packages.poisson.koszul import KoszulBracket
    from jacopy.packages.poisson.tilde import (
        LieDCommutationDefinition,
    )

    pi = P.pi

    def ev(x: Expr, y: Expr) -> Expr:
        return MultiEval(
            pi, x, y, alternating=True, slot_kind="covector"
        )

    lhs = Product(
        Integer(2),
        Pairing(
            d_(h),
            Sum(
                LieBracketVF(
                    SharpVF(pi, d_(f)), SharpVF(pi, d_(g))
                ),
                Neg(SharpVF(pi, KoszulBracket(pi, d_(f), d_(g)))),
            ),
        ),
    )
    rhs = Product(
        Integer(2),
        Sum(
            ev(d_(f), d_(ev(d_(g), d_(h)))),
            ev(d_(g), d_(ev(d_(h), d_(f)))),
            ev(d_(h), d_(ev(d_(f), d_(g)))),
        ),
    )
    engine = showcase_engine(P, registry=registry, declare_poisson=False)
    engine.register(LieDCommutationDefinition(registry))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=engine
    )


def prove_hamiltonian_morphism(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``X_{{f,g}}(h) = [X_f, X_g](h)`` — the anchor property on the
    exact generators (equivalent to the Jacobi instance), under the
    declared Poisson structure. Mechanical: both sides reduce to
    nested brackets and the difference is the Jacobi combination,
    killed by the cited instance theorem."""
    from jacopy.algorithms.simplify import simplify
    from jacopy.proof.chain import ProofChain as _PC
    from jacopy.proof.theorems import Theorem, TheoremBook, cite

    _, thm = prove_poisson_jacobi(P, f, g, h, registry=registry)
    book = TheoremBook()
    book.add(thm)
    # the residual may surface with the opposite overall sign
    neg_chain = _PC(list(thm.proof.steps))
    neg_chain.append(
        ProofStep(
            simplify(Neg(thm.lhs), registry),
            Integer(0),
            rule="negate both sides",
            justification="the combination vanishes, so does its negation",
        )
    )
    thm_neg = Theorem(
        name=thm.name + "_neg",
        statement=f"−({thm.statement})",
        lhs=simplify(Neg(thm.lhs), registry),
        rhs=Integer(0),
        proof=neg_chain,
        generality="instance",
        from_axioms=thm.from_axioms,
    )
    book.add(thm_neg)
    engine = showcase_engine(P, registry=registry)
    cite(engine, book, thm.name, thm_neg.name)
    lhs = Act(P.hamiltonian(P.bracket(f, g)), h)
    rhs = Act(LieBracketVF(P.hamiltonian(f), P.hamiltonian(g)), h)
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=engine
    )
