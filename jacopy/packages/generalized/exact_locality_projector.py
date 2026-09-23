"""
Locality projectors on the EXACT Courant algebroid ``TM ⊕ T*M``
[MC Prop 4.2] — ledger K3.b (2026-09-23; audit round 0058df0 folded in).

The paper's statement: on an exact almost-Courant algebroid there is
a UNIQUE locality projector, the projection ``pr₂`` onto
``ker ρ ≃ T*M``. Its proof (taking ``E = TM ⊕ T*M``): a locality
projector is ``C^∞``-linear with image in ``ker ρ``, so it has the
form ``(U, ω) ↦ (0, ω̃)``; it fixes coboundary values ``Df``
(``L̂ ∈ [L̃]``), hence fixes ``(0, dg)``; every 1-form is locally a
sum of exact ones, hence it fixes every ``(0, ω)``.

What is mechanized here, and a FINDING (three separate results,
each certified on its own):

* :func:`prove_projector_fixes_locally_exact_forms` — the paper's
  actual argument, on an OPAQUE ``C^∞``-linear form-slot map ``C``
  whose only hypothesis is absorption on exact forms ``C(dg) = dg``:
  ``C(Σ_i f_i dg_i) = Σ_i f_i dg_i``. So a locality projector is the
  identity on ``ker ρ = T*M``.
* :func:`prove_linear_projector_is_a_locality_projector` — the
  argument says NOTHING about ``P(U, 0)``. For ANY bundle map
  ``C: TM → T*M`` (a ``(0,2)``-tensor, ``C(U) := C(U, ·)``) the map

      P_C(U, ω) := (0, ω + C(U, ·))

  is ``C^∞``-linear (both slots, sums and scalars), has image in
  ``ker ρ``, absorbs coboundary values, is idempotent and is the
  identity on ``ker ρ`` — every requirement of [MC Def 3.10] and of
  the admissible-paper Def 3.8. The ``B``-field family
  ``C(U, ·) = ι_U B`` (:func:`b_field_projector`) is the ANTISYMMETRIC
  subfamily; in dimension one every 2-form vanishes while
  ``C(u∂ₓ) = u dx`` still works, so the freedom is NOT only a
  ``B``-transform (audit 0058df0, F3). This result holds for ``C = 0``
  too — it certifies "is a locality projector", nothing more.
* :func:`prove_projector_differs_from_pr2_on_a_frame` — the
  NON-uniqueness, certified by a concrete WITNESS instead of a
  residual the engine failed to reduce (audit 0058df0, F1: a
  non-zero normal form is not a non-zero certificate — the same
  expression may vanish under a rule family the engine lacks). On a
  frame of declared dimension with ``B = e⁰ ∧ e¹`` and ``U = e₀``,
  the form component of ``P_B(U, 0) − pr₂(U, 0)`` paired with ``e₁``
  normalizes to the LITERAL ``1`` (wedge evaluation + concrete
  Kronecker deltas), so the gap is a 1-form taking the value 1 on a
  basis vector: non-zero by definition of a frame.

Conclusion on MC Prop 4.2: the uniqueness holds on ``ker ρ`` only;
on a complement of ``T*M`` the projector is free up to an arbitrary
bundle map ``TM → T*M`` (the paper's proof never constrains
``P(U, 0)``). The DFT choice ``pr₂`` is the ``C = 0`` representative
and depends on the splitting ``τ``. Under an extra isotropy
condition on the complement one would expect ``C`` to be forced
antisymmetric (a ``B``-field); the definitions examined carry no such
condition.

Every input is TYPE-CHECKED (a 2-form ``B``, a ``(0,2)``-tensor
``C``, a 1-form ``ω``, a vector ``U``, scalar functions ``h, g``):
an unknown or wrong type is refused, never certified (audit 0058df0,
F2).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from jacopy.algebra.derivation import Act, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.endomorphism import EndoForm
from jacopy.central.objects.interior import Interior
from jacopy.central.objects.partial_eval import musical_view
from jacopy.central.objects.tensor import Tensor, signature_of
from jacopy.central.tangent.exterior import d


# --------------------------------------------------------------------- #
# Input typing                                                          #
# --------------------------------------------------------------------- #


def _form_degree(expr: Expr, registry) -> Optional[int]:
    try:
        return degree_of(expr, registry).as_int()
    except ValueError:
        return None


def _require_form(expr: Expr, degree: int, what: str, registry) -> None:
    if expr == Integer(0):
        return  # the zero form inhabits every degree
    k = _form_degree(expr, registry)
    if k != degree:
        raise TypeError(
            f"{what} must be a {degree}-form; got "
            + ("an object of undeterminable degree" if k is None else f"degree {k}")
        )


def _require_vector(expr: Expr, what: str, registry) -> None:
    lift = getattr(expr, "wedge_degree", None)
    if not (isinstance(lift, Degree) and lift == Degree.const(1)):
        raise TypeError(f"{what} must be a vector field (exterior degree 1)")


def _require_function(expr: Expr, what: str, registry) -> None:
    from jacopy.central.calculus.scalars import is_scalar_function

    if not is_scalar_function(expr, registry):
        raise TypeError(f"{what} must be a declared scalar function")


def _require_bundle_map(C: Expr, what: str) -> None:
    if signature_of(C) != (0, 2):
        raise TypeError(f"{what} must be a (0,2)-tensor (a bundle map TM → T*M)")


# --------------------------------------------------------------------- #
# Engine                                                                #
# --------------------------------------------------------------------- #


class ProjectorAbsorbsExactFormsDefinition(Definition):
    """The [MC Def 3.10] hypothesis ``L̂ ∈ [L̃]`` on the exact Courant
    algebroid, where ``L(Df, u, v) = g(u, v)·Df`` [MC Prop 4.1]: the
    form-slot map ``C`` of a locality projector FIXES exact 1-forms,
    ``C(dg) → dg`` for a function ``g``. DECLARED (it is the defining
    property of a locality projector, not derivable)."""

    anchor = EndoForm

    def __init__(self, map_name: str, registry: Optional[PropertyRegistry] = None) -> None:
        self._name = map_name
        self._registry = registry
        self.name = f"declared locality projector ({map_name}): {map_name}(dg) = dg on exact forms"

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.calculus.bracket_calculus import ExteriorDerivative
        from jacopy.central.calculus.scalars import is_scalar_function

        return (
            isinstance(expr, EndoForm)
            and expr.endo_name == self._name
            and not expr.inverted
            and isinstance(expr.arg, Act)
            and isinstance(expr.arg.op, ExteriorDerivative)
            and is_scalar_function(expr.arg.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return expr.arg


def _engine(
    registry: Optional[PropertyRegistry],
    *,
    projector_name: Optional[str] = None,
    concrete_frame=None,
):
    from jacopy.central.calculus import InteriorVectorLinearityDefinition
    from jacopy.central.calculus.indexed_rules import WedgeEvalDefinition
    from jacopy.central.objects.endomorphism import EndoLinearityDefinition
    from jacopy.packages.drinfeld.double import _double_engine

    eng = _double_engine(registry)
    eng.register(InteriorVectorLinearityDefinition(registry))
    eng.register(EndoLinearityDefinition(registry))
    if projector_name is not None:
        eng.register(ProjectorAbsorbsExactFormsDefinition(projector_name, registry))
    if concrete_frame is not None:
        from jacopy.packages.metric_affine.component_workflow import (
            ComponentInput,
            ConcreteDeltaDefinition,
        )

        fr, dim = concrete_frame
        eng.register(WedgeEvalDefinition(registry))
        eng.register(ConcreteDeltaDefinition(ComponentInput(fr, dim=dim)))
    return eng


def _norm(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


# --------------------------------------------------------------------- #
# The paper's argument: identity on ker ρ                               #
# --------------------------------------------------------------------- #


def locally_exact_form(fs: Sequence[Expr], gs: Sequence[Expr]) -> Expr:
    """``Σ_i f_i dg_i`` — the local shape of an arbitrary 1-form."""
    if len(fs) != len(gs) or not fs:
        raise ValueError("locally_exact_form needs equally many f_i and g_i (≥ 1)")
    return Sum(*(Product(f, d(g)) for f, g in zip(fs, gs)))


def prove_projector_fixes_locally_exact_forms(
    fs: Sequence[Expr],
    gs: Sequence[Expr],
    *,
    projector_name: str = "𝒫",
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[MC Prop 4.2, the argument] — a ``C^∞``-linear form-slot map
    ``C`` that fixes exact forms fixes every locally exact 1-form:

        C(Σ_i f_i dg_i) = Σ_i f_i dg_i.

    Declared: the absorption hypothesis ``C(dg) = dg``. Everything
    else is linearity. Conclusion: a locality projector is the
    identity on ``ker ρ = T*M``."""
    for x in list(fs) + list(gs):
        _require_function(x, "every f_i and g_i", registry)
    omega = locally_exact_form(fs, gs)
    diff = Sum(EndoForm(projector_name, omega), Neg(omega))
    engine = _engine(registry, projector_name=projector_name)
    nf = _norm(engine, diff, registry)
    if nf != Integer(0):
        raise ProofFailure(
            "projector_fixes_locally_exact_forms FAILS — residual " + nf._repr_inner()[:160]
        )
    chain = ProofChain(
        [
            ProofStep(
                diff,
                Integer(0),
                rule=f"{projector_name}(Σ f_i dg_i) − Σ f_i dg_i normalizes to 0 "
                "(C∞-linearity + declared absorption on exact forms)",
                justification="engine normal form",
            )
        ]
    )
    theorem = Theorem(
        name="locality_projector_is_identity_on_ker_rho",
        statement="a C∞-linear locality projector fixing coboundary values fixes "
        "every locally exact 1-form: 𝒫|_{ker ρ = T*M} = id (MC Prop 4.2, the part "
        "the proof establishes)",
        lhs=EndoForm(projector_name, omega),
        rhs=omega,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            f"declared: {projector_name}(dg) = dg (L̂ ∈ [L̃] on the exact Courant algebroid)",
            "C∞-linearity of the projector (endomorphism linearity)",
        ),
        notes="MC Prop 4.2; the TM part of the projector is NOT fixed by this "
        "argument — see prove_linear_projector_is_a_locality_projector and "
        "prove_projector_differs_from_pr2_on_a_frame",
    )
    return chain, theorem


# --------------------------------------------------------------------- #
# The family P_C and its B-field subfamily                              #
# --------------------------------------------------------------------- #


def linear_projector(C: Expr, U: Expr, omega: Expr) -> Tuple[Expr, Expr]:
    """``P_C(U, ω) := (0, ω + C(U, ·))`` for a ``(0,2)``-tensor ``C``
    read as the bundle map ``TM → T*M``, ``U ↦ C(U, ·)``."""
    _require_bundle_map(C, "C")
    return (Integer(0), Sum(omega, musical_view(C, U)))


def b_field_projector(B: Expr, U: Expr, omega: Expr, *, registry=None) -> Tuple[Expr, Expr]:
    """``P_B(U, ω) := (0, ω + ι_U B)`` — the antisymmetric subfamily
    (``C(U, ·) = ι_U B`` for a 2-form ``B``)."""
    _require_form(B, 2, "B", registry)
    return (Integer(0), Sum(omega, Act(Interior(U), B)))


def _pair_diff(a, b) -> Tuple[Expr, Expr]:
    return (Sum(a[0], Neg(b[0])), Sum(a[1], Neg(b[1])))


def _projector_checks(P, U, V, omega, eta, h, g):
    """The six requirements on a candidate ``P(U, ω) -> (vec, form)``."""
    e, e2 = (U, omega), (V, eta)
    return [
        (
            "C∞-linearity: P(h·(U,ω)) − h·P(U,ω)",
            _pair_diff(P(Product(h, U), Product(h, omega)), tuple(Product(h, c) for c in P(*e))),
        ),
        (
            "additivity on two DIFFERENT sections: P((U,ω)+(V,η)) − P(U,ω) − P(V,η)",
            _pair_diff(P(Sum(U, V), Sum(omega, eta)), tuple(Sum(x, y) for x, y in zip(P(*e), P(*e2)))),
        ),
        ("image in ker ρ: vector component of P(U,ω)", (P(*e)[0], Integer(0))),
        ("absorption: P(0, dg) − (0, dg)", _pair_diff(P(Integer(0), d(g)), (Integer(0), d(g)))),
        ("idempotence: P(P(U,ω)) − P(U,ω)", _pair_diff(P(*P(*e)), P(*e))),
        ("identity on ker ρ: P(0, ω) − (0, ω)", _pair_diff(P(Integer(0), omega), (Integer(0), omega))),
    ]


def _run_checks(name: str, checks, engine, registry) -> List[ProofStep]:
    steps: List[ProofStep] = []
    for label, (dv, df) in checks:
        for comp, diff in (("vector", dv), ("form", df)):
            nf = _norm(engine, diff, registry)
            if nf != Integer(0):
                raise ProofFailure(f"{name}: {label} [{comp}] FAILS — residual " + nf._repr_inner()[:160])
        steps.append(
            ProofStep(
                Sum(dv, df),
                Integer(0),
                rule=label + " normalizes to (0, 0)",
                justification="engine normal form of both components",
            )
        )
    return steps


def _check_section_inputs(U, V, omega, eta, h, g, registry) -> None:
    _require_vector(U, "U", registry)
    _require_vector(V, "V", registry)
    _require_form(omega, 1, "ω", registry)
    _require_form(eta, 1, "η", registry)
    _require_function(h, "h", registry)
    _require_function(g, "g", registry)


def prove_linear_projector_is_a_locality_projector(
    C: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    h: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``P_C(U, ω) = (0, ω + C(U, ·))``, for ANY ``(0,2)``-tensor ``C``,
    satisfies every requirement of a locality projector on the exact
    Courant algebroid — six mechanical checks (linearity in scalars,
    additivity on two different sections, image in ``ker ρ``,
    absorption of coboundary values, idempotence, identity on
    ``ker ρ``). This certifies "is a locality projector" and nothing
    about ``P_C ≠ pr₂`` (true for ``C = 0`` as well); the
    non-uniqueness is a separate, witnessed theorem."""
    _require_bundle_map(C, "C")
    _check_section_inputs(U, V, omega, eta, h, g, registry)
    engine = _engine(registry)
    steps = _run_checks(
        "linear_projector",
        _projector_checks(lambda u, w: linear_projector(C, u, w), U, V, omega, eta, h, g),
        engine,
        registry,
    )
    chain = ProofChain(steps)
    theorem = Theorem(
        name="linear_locality_projector_family",
        statement="P_C(U,ω) = (0, ω + C(U,·)) is a locality projector on the exact Courant "
        "algebroid for every bundle map C: TM → T*M ((0,2)-tensor): C∞-linear, image in "
        "ker ρ, absorbs coboundary values, idempotent, identity on ker ρ",
        lhs=steps[0].before,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "exact Courant algebroid TM ⊕ T*M with ρ(U,ω) = U, D f = (0, df) up to normalization",
            "partial evaluation of a (0,2)-tensor (C∞-linear in the fixed slot)",
        ),
        notes="MC Prop 4.2: the proof fixes P on ker ρ only; the B-field family "
        "ι_U B is the antisymmetric subfamily C(U,·) = ι_U B",
    )
    return chain, theorem


def prove_b_field_projector_is_a_locality_projector(
    B: Expr,
    U: Expr,
    omega: Expr,
    h: Expr,
    g: Expr,
    *,
    V: Optional[Expr] = None,
    eta: Optional[Expr] = None,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The antisymmetric subfamily: ``P_B(U, ω) = (0, ω + ι_U B)`` for a
    2-form ``B`` satisfies every requirement of a locality projector
    (six checks; additivity is tested on two DIFFERENT sections,
    fresh ones when ``V``/``η`` are not given). Certifies membership
    only — see :func:`prove_projector_differs_from_pr2_on_a_frame` for
    the non-uniqueness."""
    from jacopy.central.objects import forms, vector_fields

    _require_form(B, 2, "B", registry)
    if V is None:
        (V,) = vector_fields("V′")
    if eta is None:
        (eta,) = forms("η′", degree=1)
    _check_section_inputs(U, V, omega, eta, h, g, registry)
    engine = _engine(registry)
    steps = _run_checks(
        "b_field_projector",
        _projector_checks(lambda u, w: b_field_projector(B, u, w, registry=registry), U, V, omega, eta, h, g),
        engine,
        registry,
    )
    chain = ProofChain(steps)
    theorem = Theorem(
        name="b_field_locality_projector",
        statement="P_B(U,ω) = (0, ω + ι_U B) is a locality projector on the exact Courant "
        "algebroid for every 2-form B (C∞-linear, im ⊂ ker ρ, absorbs coboundary values, "
        "idempotent, identity on ker ρ) — the antisymmetric subfamily of the P_C family",
        lhs=steps[0].before,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "exact Courant algebroid TM ⊕ T*M with ρ(U,ω) = U, D f = (0, df) up to normalization",
            "interior product and Cartan calculus definitions (engine rules)",
        ),
        notes="membership only; P_B = pr₂ when B = 0 — non-uniqueness needs a witness",
    )
    return chain, theorem


# --------------------------------------------------------------------- #
# Non-uniqueness, witnessed                                             #
# --------------------------------------------------------------------- #


def prove_projector_differs_from_pr2_on_a_frame(
    fr,
    *,
    dim: int = 2,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """THE FINDING on [MC Prop 4.2], certified by a WITNESS: on a frame
    ``(e_a)`` of declared dimension ``dim ≥ 2`` take ``B = e⁰ ∧ e¹``
    and ``U = e₀``. Then the form component of
    ``P_B(U, 0) − pr₂(U, 0) = (0, ι_{e₀}(e⁰ ∧ e¹))`` paired with ``e₁``
    normalizes to the literal ``1`` (and with ``e₀`` to ``0``): the gap
    is a 1-form with value 1 on a basis vector, hence non-zero by
    definition of a frame. Together with
    :func:`prove_b_field_projector_is_a_locality_projector` this shows
    a second locality projector besides ``pr₂``: uniqueness holds on
    ``ker ρ`` only.

    Why a witness: a normal form the engine merely fails to reduce is
    not a non-zero certificate (audit 0058df0, F1 — for
    ``B = df ∧ dg``, ``U = Π(B)`` the gap ``U(f)dg − U(g)df`` is zero by
    alternation, which the Cartan engine does not know)."""
    if dim < 2:
        raise ValueError("the witness needs a frame of dimension ≥ 2 (a 2-form on a line is 0)")
    from jacopy.core.pairing import Pairing
    from jacopy.core.wedge import Wedge

    co = fr.dual()
    e0, e1 = fr.field("0"), fr.field("1")
    B = Wedge(co.field("0"), co.field("1"))
    gap_vec, gap_form = _pair_diff(b_field_projector(B, e0, Integer(0)), (Integer(0), Integer(0)))
    engine = _engine(registry, concrete_frame=(fr, dim))
    one = _norm(engine, Pairing(gap_form, e1), registry)
    zero = _norm(engine, Pairing(gap_form, e0), registry)
    vec = _norm(engine, gap_vec, registry)
    if one != Integer(1) or zero != Integer(0) or vec != Integer(0):
        raise ProofFailure(
            "projector_differs_from_pr2: witness did not evaluate — "
            f"⟨gap, e₁⟩ → {one._repr_inner()}, ⟨gap, e₀⟩ → {zero._repr_inner()}"
        )
    chain = ProofChain(
        [
            ProofStep(gap_vec, Integer(0), rule="vector component of P_B(e₀,0) − pr₂(e₀,0) normalizes to 0",
                      justification="engine normal form"),
            ProofStep(Pairing(gap_form, e1), Integer(1),
                      rule="⟨form component of P_B(e₀,0) − pr₂(e₀,0), e₁⟩ = ⟨ι_{e₀}(e⁰∧e¹), e₁⟩ normalizes to the LITERAL 1",
                      justification="wedge evaluation + concrete Kronecker deltas on the declared frame"),
            ProofStep(Pairing(gap_form, e0), Integer(0),
                      rule="⟨form component, e₀⟩ normalizes to 0 (the gap is e¹)",
                      justification="engine normal form"),
        ]
    )
    theorem = Theorem(
        name="locality_projector_not_unique_off_ker_rho",
        statement="on a frame of dimension ≥ 2, P_B with B = e⁰∧e¹ is a locality projector "
        "(prove_b_field_projector_is_a_locality_projector) whose gap to pr₂ at e₀ is the "
        "1-form e¹ (value 1 on e₁): MC Prop 4.2's uniqueness holds on ker ρ only",
        lhs=Pairing(gap_form, e1),
        rhs=Integer(1),
        proof=chain,
        generality="instance",
        from_axioms=(
            "frame of declared dimension (concrete basis positions: δ^a_b = [a = b])",
            "wedge evaluation and interior product definitions",
        ),
        notes="witnessed non-identity (a literal 1), not a failed reduction",
    )
    return chain, theorem
