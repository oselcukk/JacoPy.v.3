"""
The tilde calculus (Phase 5.D) — PDF items 12i-12o.

The Phase 3.F instantiation applied to the COTANGENT algebroid: the
generic :class:`BracketCalculus` with the pair ``(π♯, [·,·]_π)``
yields the tilde operators

    d̃ : 𝔛^p(M) → 𝔛^{p+1}(M),   L̃_α,   ι̃_α,

with the ROLE SWAP the grading hooks encode: the tilde calculus'
SECTIONS are 1-forms and its FORMS are multivectors (functions 0,
vector fields 1, wedges additive). No new operator code — the Palais
formula, the intrinsic L and the evaluation machinery are the Phase 2
generic rules, re-graded.

Mechanical theorems here:

* ``⟨d̃f, α⟩ = π(α, df)`` (so ``d̃f`` "is" the Hamiltonian field of
  ``f`` in evaluation),
* the Palais unroll of ``d̃X`` on two 1-forms,
* the tilde Cartan magic on functions
  ``L̃_α f = (d̃ ι̃_α + ι̃_α d̃) f``,
* ``(d̃ d̃ f)(dg, dh) = 0`` under the DECLARED Poisson structure —
  the tilde counterpart of the conditional ``d² = 0`` of Phase 3.F
  (closes via the cited Poisson-Jacobi instances; honest without),
* the tilde anholonomy ``γ̃_c^{ab} := ⟨[e^a, e^b]_π, e_c⟩``
  (item 12i — the 2.E coefficient-extraction pattern on the coframe).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, degree_of
from jacopy.core.expr import Atom, Expr, Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.central.calculus import BracketCalculus
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.frame import (
    CoframeField,
    Frame,
    FrameField,
    IndexLike,
    _index_str,
)
from jacopy.central.objects.tilde_interior import TildeInterior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.schouten import multivector_degree
from jacopy.packages.poisson.core import PoissonStructure
from jacopy.packages.poisson.koszul import KoszulBracket
from jacopy.packages.poisson.showcase import showcase_engine


def tilde_multivector_degree(
    calc_name: str, expr: Expr, registry
) -> Optional[int]:
    """The tilde-form (multivector) degree, OPERATOR-AWARE: on top of
    the structural :func:`multivector_degree`, tilde-operator
    applications grade as ``|d̃x| = |x|+1``, ``|L̃x| = |x|``,
    ``|ι̃x| = |x|−1``."""
    from jacopy.central.calculus.bracket_calculus import (
        ExteriorDerivative,
        LieDerivative,
    )

    if isinstance(expr, Act):
        op = expr.op
        inner = tilde_multivector_degree(calc_name, expr.arg, registry)
        if inner is None:
            return None
        if (
            isinstance(op, ExteriorDerivative)
            and op.calculus_name == calc_name
        ):
            return inner + 1
        if (
            isinstance(op, LieDerivative)
            and op.calculus_name == calc_name
        ):
            return inner
        if isinstance(op, TildeInterior):
            return inner - 1
        return None
    return multivector_degree(expr, registry)


def tilde_calculus(P: PoissonStructure) -> BracketCalculus:
    """The tilde calculus of the Poisson structure: the bracket
    calculus of the cotangent algebroid ``(T*M, π♯, [·,·]_π)`` with
    the swapped grading."""
    if not isinstance(P, PoissonStructure):
        raise TypeError("tilde_calculus expects a PoissonStructure")
    calc_name = f"tilde-{P.pi._repr_inner()}"

    def _form_degree(expr: Expr, registry) -> Degree:
        m = tilde_multivector_degree(calc_name, expr, registry)
        if m is None:
            # Distinguish "unknown/symbolic" (caller-asserted, raise →
            # the arity policy allows it) from "determinably an
            # ORDINARY form of degree ≥ 1" — that is a tilde-SECTION,
            # not a tilde-form: d̃α / L̃-evaluation on it is
            # ill-typed. Return the −1 sentinel, which matches no
            # arity, so the rules stay honestly inert (type hole
            # caught by the Phase 5 edge-case audit).
            try:
                if degree_of(expr, registry).as_int() >= 1:
                    return Degree.const(-1)
            except ValueError:
                pass
            raise ValueError(
                f"multivector degree of {expr!r} undetermined"
            )
        return Degree.const(m)

    def _section_test(expr: Expr) -> bool:
        # tilde-sections are 1-FORMS. Anything with a determinable
        # TILDE-form degree (d̃s, L̃-/ι̃-compounds, vector fields) is
        # a tilde-FORM, not a section — the ordinary ``degree_of``
        # reads a tilde-operator Act as +1 wedge degree and would
        # misclassify ``d̃s`` as an ordinary 1-form, turning
        # ``L̃_β(d̃s)`` into the ill-typed ``[β, d̃s]_π`` (caught by
        # the 5.D.2 mixed-world audit).
        if tilde_multivector_degree(calc_name, expr, None) is not None:
            return False
        try:
            return degree_of(expr, None) == Degree.const(1)
        except ValueError:
            return False

    return BracketCalculus(
        calc_name,
        anchor=P.sharp_vf,
        bracket=lambda a, b: KoszulBracket(P.pi, a, b),
        d_name="d̃",
        lie_name="L̃",
        form_degree=_form_degree,
        section_test=_section_test,
    )


def d_tilde(P: PoissonStructure, V: Expr) -> Act:
    """``d̃V`` — the Lichnerowicz-style tilde differential node."""
    return Act(tilde_calculus(P).d, V)


def lie_tilde(P: PoissonStructure, alpha: Expr):
    """``L̃_α`` — the tilde Lie derivative operator."""
    return tilde_calculus(P).lie(alpha)


class TildeInteriorActDefinition(Definition):
    """``ι̃_α`` on the degenerate tilde degrees: scalars die,
    tilde-1-forms (vector fields) evaluate —
    ``ι̃_α f → 0``, ``ι̃_α X → ⟨X, α⟩``."""

    anchor = Act

    def __init__(
        self,
        P: PoissonStructure,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._P = P
        self._registry = registry
        self.name = (
            "tilde interior: ι̃_α f = 0, ι̃_α X = ⟨X, α⟩ "
            "(tilde-degree 0 / 1)"
        )

    def _tilde_degree(self, expr: Expr) -> Optional[int]:
        return tilde_multivector_degree(
            f"tilde-{self._P.pi._repr_inner()}", expr, self._registry
        )

    def matches(self, expr: Expr) -> bool:
        if not (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeInterior)
        ):
            return False
        return self._tilde_degree(expr.arg) in (0, 1)

    def rewrite(self, expr: Expr) -> Expr:
        if self._tilde_degree(expr.arg) == 0:
            return Integer(0)
        return Pairing(expr.arg, expr.op.form)


class TildeInteriorEvalDefinition(Definition):
    """Slot insertion under tilde evaluation:
    ``(ι̃_α V)(β…) → V(α, β…)`` (the tilde arguments are 1-forms)."""

    anchor = (Pairing, MultiEval)

    def __init__(
        self,
        P: PoissonStructure,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._P = P
        self._registry = registry
        self.name = (
            "tilde interior evaluation: (ι̃_α V)(β…) = V(α, β…)"
        )

    def _is_tilde_interior_act(self, head: Expr) -> bool:
        return isinstance(head, Act) and isinstance(
            head.op, TildeInterior
        )

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Pairing):
            return self._is_tilde_interior_act(expr.alpha)
        if isinstance(expr, MultiEval):
            return self._is_tilde_interior_act(expr.head)
        return False

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            head = expr.alpha
            return MultiEval(
                head.arg,
                head.op.form,
                expr.X,
                alternating=True,
                slot_kind="covector",
            )
        head = expr.head
        return MultiEval(
            head.arg,
            head.op.form,
            *expr.args,
            alternating=True,
            slot_kind="covector",
        )


class LieDCommutationDefinition(Definition):
    """``L_X(df) → d(X(f))`` — the TM Lie derivative commutes with
    ``d`` on functions. A DERIVED rule (theorem-classified: the
    Phase 2 theorem ``prove_L_commutes_with_d_on_functions`` is its
    builder); the tangent engine deliberately omits it, the tilde
    layer opts in (Koszul expansions park ``L_{π♯dg}(dh)`` inside
    evaluation slots where only this conversion unblocks them)."""

    anchor = Act

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry
        self.name = (
            "L-d commutation (derived): L_X(df) = d(X(f))"
        )

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.calculus.bracket_calculus import (
            ExteriorDerivative,
            LieDerivative,
        )
        from jacopy.central.tangent.exterior import CARTAN_TM

        if not (
            isinstance(expr, Act)
            and isinstance(expr.op, LieDerivative)
            and expr.op.calculus_name == CARTAN_TM.name
        ):
            return False
        arg = expr.arg
        return (
            isinstance(arg, Act)
            and isinstance(arg.op, ExteriorDerivative)
            and arg.op.calculus_name == CARTAN_TM.name
            and is_scalar_function(arg.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.op.vector
        f = expr.arg.arg
        return d(Act(X, f))

    def theorem_proof_builder(self):
        registry = self._registry

        def _builder(matched: Expr) -> ProofChain:
            from jacopy.central.tangent.cartan import (
                prove_L_commutes_with_d_on_functions,
            )

            X = matched.op.vector
            f = matched.arg.arg
            return prove_L_commutes_with_d_on_functions(
                X, f, registry=registry
            )

        return _builder


def tilde_engine(
    P: PoissonStructure,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
):
    """The 5.D engine: the 5.C showcase layer plus the tilde
    calculus' intrinsic d/L rules and the tilde interior."""
    from jacopy.central.calculus import (
        IntrinsicDDefinition,
        IntrinsicLDefinition,
    )

    engine = showcase_engine(
        P, registry=registry, declare_poisson=declare_poisson
    )
    calc = tilde_calculus(P)
    engine.register(IntrinsicDDefinition(calc, registry))
    engine.register(IntrinsicLDefinition(calc, registry))
    engine.register(TildeInteriorActDefinition(P, registry))
    engine.register(TildeInteriorEvalDefinition(P, registry))
    engine.register(LieDCommutationDefinition(registry))
    return engine


# --------------------------------------------------------------------- #
# Tilde anholonomy (item 12i)                                            #
# --------------------------------------------------------------------- #


class TildeAnholonomyCoefficient(Atom):
    """``γ̃_c^{ab}`` — a tilde anholonomy coefficient: the Koszul
    bracket of two coframe fields decomposed against the frame
    (a genuine function, degree 0)."""

    __slots__ = ("_pi", "_frame_name", "_first", "_second", "_lower")

    def __init__(
        self, pi, frame_name: str, first: str, second: str, lower: str
    ) -> None:
        self._pi = pi
        self._frame_name = frame_name
        self._first = first
        self._second = second
        self._lower = lower

    @property
    def degree(self) -> Degree:
        return Degree.const(0)

    @property
    def upper(self) -> Tuple[str, str]:
        return (self._first, self._second)

    @property
    def lower(self) -> str:
        return self._lower

    @property
    def index_names(self) -> Tuple[str, ...]:
        return (self._first, self._second, self._lower)

    def _key(self) -> Any:
        return (
            self._pi,
            self._frame_name,
            self._first,
            self._second,
            self._lower,
        )

    def _repr_inner(self) -> str:
        return f"γ̃_{self._lower}^{self._first}{self._second}"


def tilde_anholonomy_coefficient(
    P: PoissonStructure,
    fr: Frame,
    first: IndexLike,
    second: IndexLike,
    lower: IndexLike,
) -> Expr:
    """``γ̃_c^{ab}`` in canonical form (``a > b`` normalizes to
    ``−γ̃_c^{ba}``, ``a == b`` gives 0 — the Koszul antisymmetry as a
    canonical form, theorem-backed by
    :func:`~jacopy.packages.poisson.koszul.prove_koszul_antisymmetry`)."""
    a = _index_str(first)
    b = _index_str(second)
    c = _index_str(lower)
    if a == b:
        return Integer(0)
    if a > b:
        return Neg(
            TildeAnholonomyCoefficient(P.pi, fr.name, b, a, c)
        )
    return TildeAnholonomyCoefficient(P.pi, fr.name, a, b, c)


class TildeAnholonomyDefinition(Definition):
    """``⟨[e^a, e^b]_π, e_c⟩ → γ̃_c^{ab}`` — the coefficient
    extraction (the 2.E pattern on the coframe)."""

    anchor = Pairing

    def __init__(self, P: PoissonStructure, fr: Frame) -> None:
        self._P = P
        self._frame = fr
        self.name = (
            f"tilde anholonomy ({P.pi._repr_inner()}, {fr.name}): "
            "⟨[e^a, e^b]_π, e_c⟩ = γ̃_c^{ab}"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        kb, X = expr.alpha, expr.X
        if not (
            isinstance(kb, KoszulBracket) and kb.pi == self._P.pi
        ):
            return False
        if not (
            isinstance(kb.alpha, CoframeField)
            and isinstance(kb.beta, CoframeField)
            and isinstance(X, FrameField)
        ):
            return False
        return all(
            leg.base_name == self._frame.name
            for leg in (kb.alpha, kb.beta, X)
        )

    def rewrite(self, expr: Expr) -> Expr:
        kb = expr.alpha
        return tilde_anholonomy_coefficient(
            self._P,
            self._frame,
            kb.alpha.index,
            kb.beta.index,
            expr.X.index,
        )


# --------------------------------------------------------------------- #
# Theorems                                                               #
# --------------------------------------------------------------------- #


def prove_d_tilde_on_functions(
    P: PoissonStructure,
    f: Expr,
    alpha: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨d̃f, α⟩ = π(α, df)`` — the tilde 0-form case: the tilde
    differential of a function is the Hamiltonian direction."""
    rhs = MultiEval(
        P.pi, alpha, d(f), alternating=True, slot_kind="covector"
    )
    return ExpandAndSimplify().prove(
        Pairing(d_tilde(P, f), alpha),
        rhs,
        registry=registry,
        engine=tilde_engine(P, registry=registry),
    )


def prove_d_tilde_palais_on_vectors(
    P: PoissonStructure,
    X: Expr,
    alpha: Expr,
    beta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``(d̃X)(α, β) = (π♯α)⟨X,β⟩ − (π♯β)⟨X,α⟩ − ⟨X, [α,β]_π⟩`` —
    the Palais unroll in the tilde grading."""
    lhs = MultiEval(
        d_tilde(P, X), alpha, beta, alternating=True, slot_kind="covector"
    )
    rhs = Sum(
        Act(P.sharp_vf(alpha), Pairing(X, beta)),
        Neg(Act(P.sharp_vf(beta), Pairing(X, alpha))),
        Neg(Pairing(X, KoszulBracket(P.pi, alpha, beta))),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=tilde_engine(P, registry=registry)
    )


def prove_tilde_magic_on_functions(
    P: PoissonStructure,
    alpha: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``L̃_α f = (d̃ ι̃_α + ι̃_α d̃) f`` — the tilde Cartan magic on
    functions, purely definitional."""
    iota = TildeInterior(alpha)
    calc = tilde_calculus(P)
    rhs = Sum(
        Act(calc.d, Act(iota, f)),
        Act(iota, d_tilde(P, f)),
    )
    return ExpandAndSimplify().prove(
        Act(lie_tilde(P, alpha), f),
        rhs,
        registry=registry,
        engine=tilde_engine(P, registry=registry),
    )


def prove_d_tilde_squared_on_exacts(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``(d̃ d̃ f)(dg, dh) = 0`` — the tilde counterpart of the
    conditional ``d² = 0`` (Phase 3.F): closes under the DECLARED
    Poisson structure via the cited Poisson-Jacobi instances; fails
    honestly without."""
    from jacopy.proof.theorems import TheoremBook, cite
    from jacopy.algorithms.simplify import simplify
    from jacopy.proof.theorems import Theorem
    from jacopy.packages.poisson.showcase import prove_poisson_jacobi

    _, thm = prove_poisson_jacobi(P, f, g, h, registry=registry)
    book = TheoremBook()
    book.add(thm)
    neg = Theorem(
        name=thm.name + "_neg",
        statement=f"−({thm.statement})",
        lhs=simplify(Neg(thm.lhs), registry),
        rhs=Integer(0),
        proof=thm.proof,
        generality="instance",
        from_axioms=thm.from_axioms,
    )
    book.add(neg)
    engine = tilde_engine(P, registry=registry)
    cite(engine, book, thm.name, neg.name)
    lhs = MultiEval(
        d_tilde(P, d_tilde(P, f)),
        d(g),
        d(h),
        alternating=True,
        slot_kind="covector",
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


def prove_tilde_magic_on_vectors(
    P: PoissonStructure,
    alpha: Expr,
    X: Expr,
    beta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L̃_α X, β⟩ = ⟨(d̃ ι̃_α + ι̃_α d̃) X, β⟩`` — the tilde
    Cartan magic on a VECTOR FIELD (a tilde-1-form), general 1-forms;
    purely definitional, no Jacobi needed."""
    iota = TildeInterior(alpha)
    calc = tilde_calculus(P)
    rhs = Sum(
        Pairing(Act(calc.d, Act(iota, X)), beta),
        Pairing(Act(iota, Act(calc.d, X)), beta),
    )
    return ExpandAndSimplify().prove(
        Pairing(Act(lie_tilde(P, alpha), X), beta),
        rhs,
        registry=registry,
        engine=tilde_engine(P, registry=registry),
    )


def prove_lie_tilde_iota_commutator_on_vectors(
    P: PoissonStructure,
    alpha: Expr,
    beta: Expr,
    X: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``L̃_α(ι̃_β X) − ι̃_β(L̃_α X) = ι̃_{[α,β]_π} X`` — the tilde
    ``[L, ι]`` commutator on a vector field, general 1-forms;
    definitional (no Jacobi)."""
    lhs = Sum(
        Act(lie_tilde(P, alpha), Act(TildeInterior(beta), X)),
        Neg(Act(TildeInterior(beta), Act(lie_tilde(P, alpha), X))),
    )
    rhs = Act(TildeInterior(KoszulBracket(P.pi, alpha, beta)), X)
    return ExpandAndSimplify().prove(
        lhs,
        rhs,
        registry=registry,
        engine=tilde_engine(P, registry=registry),
    )


def _normalized_by(engine, seed: Expr, registry) -> Expr:
    """The citing engine's normal form of ``seed`` (theorem lhs must
    match what the engine actually produces)."""
    from jacopy.algorithms.product_rule import product_rule
    from jacopy.algorithms.simplify import simplify

    cur = seed
    for _ in range(10):
        expanded, _steps = engine.expand(cur)
        after = product_rule(expanded, registry)
        reduced = simplify(after, registry)
        if reduced == cur:
            break
        cur = reduced
    return cur


def _cite_jacobi_instances(engine, P, triples, registry) -> None:
    """Register ``±`` Poisson-Jacobi instance theorems for each
    ``(a, b, c)`` scalar triple, plus — for triples tagged with
    ``d_lift=True`` — the d-LIFTED instances ``±d(Jacobiator) = 0``
    (the differential of a proven-zero scalar is zero; the lift is
    recorded as an explicit synthetic step over the cited chain)."""
    from jacopy.algorithms.simplify import simplify
    from jacopy.proof.theorems import Theorem, TheoremBook, cite
    from jacopy.packages.poisson.showcase import (
        _cyclic_jacobi_sum,
        prove_poisson_jacobi,
    )

    book = TheoremBook()
    names = []
    for (a, b, c), d_lift in triples:
        _, thm = prove_poisson_jacobi(P, a, b, c, registry=registry)
        if thm.name not in book:
            book.add(thm)
            names.append(thm.name)
        neg_name = thm.name + "_neg"
        if neg_name not in book:
            book.add(
                Theorem(
                    name=neg_name,
                    statement=f"−({thm.statement})",
                    lhs=simplify(Neg(thm.lhs), registry),
                    rhs=Integer(0),
                    proof=thm.proof,
                    generality="instance",
                    from_axioms=thm.from_axioms,
                )
            )
            names.append(neg_name)
        if not d_lift:
            continue
        jac = _cyclic_jacobi_sum(P, a, b, c)
        lift = ProofStep(
            Act(CARTAN_TM.d, jac),
            Integer(0),
            rule="differential of a proven-zero scalar",
            justification="Jacobiator = 0 (cited), hence d(Jac) = 0",
        )
        for s in thm.proof:
            lift.add_child(s)
        lift_chain = ProofChain([lift])
        for sign, tag in (((lambda x: x), "pos"), (Neg, "neg")):
            name = f"{thm.name}_d_lift_{tag}"
            if name in book:
                continue
            book.add(
                Theorem(
                    name=name,
                    statement=f"{'-' if tag == 'neg' else ''}"
                    "d(Jacobiator) = 0",
                    lhs=_normalized_by(
                        engine, sign(Act(CARTAN_TM.d, jac)), registry
                    ),
                    rhs=Integer(0),
                    proof=lift_chain,
                    generality="instance",
                    from_axioms=thm.from_axioms,
                )
            )
            names.append(name)
    cite(engine, book, *names)


def prove_lie_tilde_commutator_on_exacts(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    X: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨(L̃_{df} L̃_{dg} − L̃_{dg} L̃_{df} − L̃_{[df,dg]_π}) X, dh⟩
    = 0`` — the first Q3 algebroid-calculus identity in the tilde
    world, on exact generators. Requires the DECLARED Poisson
    structure: the residual splits into the scalar Jacobiator on
    ``(f, g, ⟨X, dh⟩)`` and the d-LIFTED Jacobiator on ``(f, g, h)``,
    both entering by citation. The general-form version needs the
    general SN evaluation (Phase 5.E)."""
    engine = tilde_engine(P, registry=registry)
    _cite_jacobi_instances(
        engine,
        P,
        (
            ((f, g, Pairing(X, d(h))), False),
            ((f, g, h), True),
        ),
        registry,
    )
    alpha, beta, gamma = d(f), d(g), d(h)
    lhs = Sum(
        Pairing(
            Act(lie_tilde(P, alpha), Act(lie_tilde(P, beta), X)), gamma
        ),
        Neg(
            Pairing(
                Act(lie_tilde(P, beta), Act(lie_tilde(P, alpha), X)),
                gamma,
            )
        ),
        Neg(
            Pairing(
                Act(lie_tilde(P, KoszulBracket(P.pi, alpha, beta)), X),
                gamma,
            )
        ),
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


def prove_lie_tilde_d_iota_commutation_on_exacts(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    X: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L̃_{df} d̃ι̃_{dg} X − d̃ι̃_{[df,dg]_π} X
    − d̃ι̃_{dg} L̃_{df} X, dh⟩ = 0`` — the second Q3 identity in the
    tilde world, exact generators; the residual is the scalar
    Jacobiator on ``(f, h, ⟨X, dg⟩)`` (cited)."""
    engine = tilde_engine(P, registry=registry)
    _cite_jacobi_instances(
        engine, P, (((f, h, Pairing(X, d(g))), False),), registry
    )
    calc = tilde_calculus(P)
    alpha, beta, gamma = d(f), d(g), d(h)
    lhs = Sum(
        Pairing(
            Act(
                lie_tilde(P, alpha),
                Act(calc.d, Act(TildeInterior(beta), X)),
            ),
            gamma,
        ),
        Neg(
            Pairing(
                Act(
                    calc.d,
                    Act(
                        TildeInterior(KoszulBracket(P.pi, alpha, beta)),
                        X,
                    ),
                ),
                gamma,
            )
        ),
        Neg(
            Pairing(
                Act(
                    calc.d,
                    Act(
                        TildeInterior(beta),
                        Act(lie_tilde(P, alpha), X),
                    ),
                ),
                gamma,
            )
        ),
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


def prove_lie_tilde_d_iota_exact_on_exacts(
    P: PoissonStructure,
    f: Expr,
    g: Expr,
    X: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L̃_{dg} d̃ι̃_{df} X − d̃ι̃_{dg} d̃ι̃_{df} X, dh⟩ = 0`` —
    the third Q3 identity (tilde magic on the exact tilde-form
    ``d̃ι̃_{df} X``), exact generators; the residual is the scalar
    Jacobiator on ``(g, h, ⟨X, df⟩)`` (cited)."""
    engine = tilde_engine(P, registry=registry)
    _cite_jacobi_instances(
        engine, P, (((g, h, Pairing(X, d(f))), False),), registry
    )
    calc = tilde_calculus(P)
    alpha, beta, gamma = d(f), d(g), d(h)
    inner = Act(calc.d, Act(TildeInterior(alpha), X))
    lhs = Sum(
        Pairing(Act(lie_tilde(P, beta), inner), gamma),
        Neg(
            Pairing(
                Act(calc.d, Act(TildeInterior(beta), inner)), gamma
            )
        ),
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


def _cite_sn_general_instances(engine, P, triples, registry) -> None:
    """Register ``±`` (and half-scale) evaluated instances of the
    DECLARED ``[π,π]_SN = 0`` on GENERAL 1-form triples, via the
    5.E.2 general evaluation view. Two legs per instance: the bare
    expansion (declaration withheld, the anchor-defect view fires)
    and the declared kill — combined explicitly."""
    from jacopy.core.expr import Rational, Product
    from jacopy.core.multi_eval import MultiEval
    from jacopy.central.tangent.schouten import sn_bracket
    from jacopy.proof.theorems import Theorem, TheoremBook, cite
    from jacopy.packages.poisson.showcase import showcase_engine

    book = TheoremBook()
    names = []
    seen = set()
    bare = showcase_engine(P, registry=registry, declare_poisson=False)
    declared = showcase_engine(P, registry=registry)
    for i, (a, b, c) in enumerate(triples):
        node = MultiEval(
            sn_bracket(P.pi, P.pi),
            a,
            b,
            c,
            alternating=True,
            slot_kind="covector",
        )
        N = _normalized_by(bare, node, registry)
        if N == Integer(0):
            continue
        leg1 = ExpandAndSimplify().prove(
            node, N, registry=registry, engine=bare
        )
        step1 = ProofStep(
            node,
            N,
            rule="general SN evaluation (anchor-defect view)",
            justification="declaration withheld",
        )
        for s in leg1:
            step1.add_child(s)
        leg2 = ExpandAndSimplify().prove(
            node, Integer(0), registry=registry, engine=declared
        )
        step2 = ProofStep(
            node,
            Integer(0),
            rule="declared Poisson structure",
            justification="[π,π]_SN = 0, so its evaluation vanishes",
        )
        for s in leg2:
            step2.add_child(s)
        step3 = ProofStep(
            N,
            Integer(0),
            rule="combine the two legs",
            justification="the expanded view equals [π,π](·,·,·) = 0",
        )
        chain = ProofChain([step1, step2, step3])
        for j, seed in enumerate((N, Product(Rational(1, 2), N))):
            for sign, tag in (((lambda x: x), "pos"), (Neg, "neg")):
                lhs = _normalized_by(engine, sign(seed), registry)
                if lhs == Integer(0) or lhs in seen:
                    continue
                seen.add(lhs)
                name = f"sn_general_inst_{i}_{j}_{tag}"
                book.add(
                    Theorem(
                        name=name,
                        statement=(
                            "[π,π](α,β,γ) evaluated instance = 0"
                        ),
                        lhs=lhs,
                        rhs=Integer(0),
                        proof=chain,
                        generality="instance",
                        from_axioms=(
                            f"Poisson ({P.pi._repr_inner()}): "
                            "[π, π]_SN = 0",
                        ),
                    )
                )
                names.append(name)
    if names:
        cite(engine, book, *names)


def prove_lie_tilde_d_iota_commutation_general(
    P: PoissonStructure,
    alpha: Expr,
    beta: Expr,
    X: Expr,
    gamma: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L̃_α d̃ι̃_β X − d̃ι̃_{[α,β]_π} X − d̃ι̃_β L̃_α X, γ⟩ = 0``
    on GENERAL 1-forms (Phase 5.E.2) — the Jacobi obstruction enters
    through the cited general SN instance on ``(α, γ, d⟨X,β⟩)``."""
    engine = tilde_engine(P, registry=registry)
    _cite_sn_general_instances(
        engine,
        P,
        ((alpha, gamma, d(Pairing(X, beta))),),
        registry,
    )
    calc = tilde_calculus(P)
    lhs = Sum(
        Pairing(
            Act(
                lie_tilde(P, alpha),
                Act(calc.d, Act(TildeInterior(beta), X)),
            ),
            gamma,
        ),
        Neg(
            Pairing(
                Act(
                    calc.d,
                    Act(
                        TildeInterior(
                            KoszulBracket(P.pi, alpha, beta)
                        ),
                        X,
                    ),
                ),
                gamma,
            )
        ),
        Neg(
            Pairing(
                Act(
                    calc.d,
                    Act(
                        TildeInterior(beta),
                        Act(lie_tilde(P, alpha), X),
                    ),
                ),
                gamma,
            )
        ),
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


def prove_lie_tilde_d_iota_exact_general(
    P: PoissonStructure,
    alpha: Expr,
    beta: Expr,
    X: Expr,
    gamma: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L̃_β d̃ι̃_α X − d̃ι̃_β d̃ι̃_α X, γ⟩ = 0`` on GENERAL
    1-forms (Phase 5.E.2) — cited general SN instance on
    ``(β, γ, d⟨X,α⟩)``."""
    engine = tilde_engine(P, registry=registry)
    _cite_sn_general_instances(
        engine,
        P,
        ((beta, gamma, d(Pairing(X, alpha))),),
        registry,
    )
    calc = tilde_calculus(P)
    inner = Act(calc.d, Act(TildeInterior(alpha), X))
    lhs = Sum(
        Pairing(Act(lie_tilde(P, beta), inner), gamma),
        Neg(
            Pairing(
                Act(calc.d, Act(TildeInterior(beta), inner)), gamma
            )
        ),
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=engine
    )


def prove_tilde_gamma_antisymmetry(
    P: PoissonStructure,
    fr: Frame,
    a: IndexLike,
    b: IndexLike,
    c: IndexLike,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨[e^a, e^b]_π, e_c⟩ = −⟨[e^b, e^a]_π, e_c⟩`` — the tilde
    anholonomy antisymmetry (the canonical index order's mechanical
    backing)."""
    engine = tilde_engine(P, registry=registry)
    engine.register(TildeAnholonomyDefinition(P, fr))
    co = fr.dual()
    lhs = Pairing(
        KoszulBracket(P.pi, co.field(a), co.field(b)), fr.field(c)
    )
    rhs = Neg(
        Pairing(
            KoszulBracket(P.pi, co.field(b), co.field(a)), fr.field(c)
        )
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=engine
    )
