"""
Locality projectors on the EXACT Courant algebroid ``TM ⊕ T*M``
[MC Prop 4.2] — ledger K3.b (2026-09-23).

The paper's statement: on an exact almost-Courant algebroid there is
a UNIQUE locality projector, the projection ``pr₂`` onto
``ker ρ ≃ T*M``. Its proof (taking ``E = TM ⊕ T*M``): a locality
projector is ``C^∞``-linear with image in ``ker ρ``, so it has the
form ``(U, ω) ↦ (0, ω̃)``; it fixes coboundary values ``Df``
(``L̂ ∈ [L̃]``), hence fixes ``(0, dg)``; every 1-form is locally a
sum of exact ones, hence it fixes every ``(0, ω)``.

What is mechanized here, and a FINDING:

* :func:`prove_projector_fixes_locally_exact_forms` — the paper's
  actual argument, on an OPAQUE ``C^∞``-linear form-slot map ``C``
  whose only hypothesis is absorption on exact forms ``C(dg) = dg``:
  ``C(Σ_i f_i dg_i) = Σ_i f_i dg_i`` (linearity + the hypothesis).
  So a locality projector is the identity on ``ker ρ = T*M``.
* :func:`prove_b_field_projector_is_a_locality_projector` — the
  argument says NOTHING about ``P(U, 0)``. The ``B``-field map

      P_B(U, ω) := (0, ω + ι_U B),   B ∈ Ω²(M),

  is ``C^∞``-linear (both slots), has image in ``ker ρ``, absorbs
  coboundary values and is idempotent — every requirement of
  [MC Def 3.10] and of the admissible-paper Def 3.8
  (``P|_{ker ρ} = id``) — yet ``P_B(U, 0) = (0, ι_U B) ≠ 0 = pr₂(U, 0)``
  for a generic ``B``. Hence the uniqueness claim of Prop 4.2 holds
  ON ``ker ρ`` only: a locality projector on an exact Courant
  algebroid is determined up to a ``B``-transform of its action on
  a complement of ``T*M`` (the paper's proof establishes
  ``P|_{T*M} = id`` and does not constrain the ``TM`` part). The
  DFT choice ``pr₂`` is the ``B = 0`` representative, i.e. it
  depends on the splitting ``τ``.

Everything is engine-normalized on ``(vector, form)`` components;
the non-uniqueness is shown by a normal form that is literally
``ι_U B``, not by a failed proof.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.endomorphism import EndoForm
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import d


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


def _engine(registry: Optional[PropertyRegistry], *, projector_name: Optional[str] = None):
    from jacopy.central.calculus import InteriorVectorLinearityDefinition
    from jacopy.central.objects.endomorphism import EndoLinearityDefinition
    from jacopy.packages.drinfeld.double import _double_engine

    eng = _double_engine(registry)
    eng.register(InteriorVectorLinearityDefinition(registry))
    eng.register(EndoLinearityDefinition(registry))
    if projector_name is not None:
        eng.register(ProjectorAbsorbsExactFormsDefinition(projector_name, registry))
    return eng


def _norm(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


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

    Declared: the absorption hypothesis ``C(dg) = dg`` (the locality
    projector's defining property on the exact Courant algebroid,
    where ``L(Df,u,v) = g(u,v)·Df``). Everything else is linearity.
    Conclusion: a locality projector is the identity on
    ``ker ρ = T*M``."""
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
        "argument — see prove_b_field_projector_is_a_locality_projector",
    )
    return chain, theorem


def b_field_projector(B: Expr, U: Expr, omega: Expr) -> Tuple[Expr, Expr]:
    """``P_B(U, ω) := (0, ω + ι_U B)`` — the ``B``-field deformation of
    ``pr₂``."""
    return (Integer(0), Sum(omega, Act(Interior(U), B)))


def _pair_diff(a, b) -> Tuple[Expr, Expr]:
    return (Sum(a[0], Neg(b[0])), Sum(a[1], Neg(b[1])))


def prove_b_field_projector_is_a_locality_projector(
    B: Expr,
    U: Expr,
    omega: Expr,
    h: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """THE FINDING on [MC Prop 4.2]: ``P_B(U, ω) = (0, ω + ι_U B)``
    satisfies every requirement of a locality projector —

    1. ``C^∞``-linearity: ``P_B(h·e) = h·P_B(e)`` and additivity;
    2. image in ``ker ρ``: the vector component is literally ``0``;
    3. absorption of coboundary values: ``P_B(0, dg) = (0, dg)``;
    4. idempotence / ``P|_{ker ρ} = id``: ``P_B∘P_B = P_B`` and
       ``P_B(0, ω) = (0, ω)``;

    yet ``P_B ≠ pr₂``: ``P_B(U, 0) − pr₂(U, 0) = (0, ι_U B)``, whose
    normal form is the non-zero 1-form ``ι_U B``. The uniqueness of
    Prop 4.2 therefore holds on ``ker ρ`` only; on a complement the
    projector is free up to a ``B``-transform (the paper's proof
    never constrains ``P(U, 0)``). The last step records the non-zero
    normal form explicitly (a certified NON-identity, not a failed
    proof)."""
    engine = _engine(registry)
    e = (U, omega)
    checks: List[Tuple[str, Tuple[Expr, Expr]]] = [
        (
            "C∞-linearity: P_B(h·(U,ω)) − h·P_B(U,ω)",
            _pair_diff(
                b_field_projector(B, Product(h, U), Product(h, omega)),
                tuple(Product(h, c) for c in b_field_projector(B, *e)),
            ),
        ),
        (
            "additivity: P_B((U,ω) + (U,ω)) − 2·P_B(U,ω)",
            _pair_diff(
                b_field_projector(B, Sum(U, U), Sum(omega, omega)),
                tuple(Sum(c, c) for c in b_field_projector(B, *e)),
            ),
        ),
        (
            "image in ker ρ: vector component of P_B(U,ω)",
            (b_field_projector(B, *e)[0], Integer(0)),
        ),
        (
            "absorption: P_B(0, dg) − (0, dg)",
            _pair_diff(b_field_projector(B, Integer(0), d(g)), (Integer(0), d(g))),
        ),
        (
            "idempotence: P_B(P_B(U,ω)) − P_B(U,ω)",
            _pair_diff(b_field_projector(B, *b_field_projector(B, *e)), b_field_projector(B, *e)),
        ),
        (
            "identity on ker ρ: P_B(0, ω) − (0, ω)",
            _pair_diff(b_field_projector(B, Integer(0), omega), (Integer(0), omega)),
        ),
    ]
    steps: List[ProofStep] = []
    for label, (dv, df) in checks:
        for comp, diff in (("vector", dv), ("form", df)):
            nf = _norm(engine, diff, registry)
            if nf != Integer(0):
                raise ProofFailure(
                    f"b_field_projector: {label} [{comp}] FAILS — residual " + nf._repr_inner()[:160]
                )
        steps.append(
            ProofStep(
                Sum(dv, df),
                Integer(0),
                rule=label + " normalizes to (0, 0)",
                justification="engine normal form of both components",
            )
        )
    # the certified non-identity: P_B(U, 0) − pr₂(U, 0) = (0, ι_U B)
    gap = _pair_diff(b_field_projector(B, U, Integer(0)), (Integer(0), Integer(0)))
    gap_nf = _norm(engine, gap[1], registry)
    if gap_nf == Integer(0):
        raise ProofFailure(
            "b_field_projector: P_B(U,0) − pr₂(U,0) normalized to 0 — the deformation "
            "collapsed; the finding does not stand for this B"
        )
    steps.append(
        ProofStep(
            gap[1],
            gap_nf,
            rule="P_B(U,0) − pr₂(U,0): form component normalizes to the NON-ZERO 1-form ι_U B",
            justification="certified non-identity (normal form ≠ 0) — P_B ≠ pr₂",
        )
    )
    chain = ProofChain(steps)
    theorem = Theorem(
        name="b_field_locality_projector",
        statement="P_B(U,ω) = (0, ω + ι_U B) is a locality projector on the exact Courant "
        "algebroid (C∞-linear, im ⊂ ker ρ, absorbs coboundary values, idempotent, "
        "identity on ker ρ) with P_B(U,0) = (0, ι_U B) ≠ pr₂(U,0): the uniqueness "
        "of MC Prop 4.2 holds on ker ρ only — off ker ρ the projector is free up to a "
        "B-transform",
        lhs=gap[1],
        rhs=gap_nf,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "exact Courant algebroid TM ⊕ T*M with ρ(U,ω) = U, D f = (0, df) up to "
            "normalization",
            "interior product and Cartan calculus definitions (engine rules)",
        ),
        notes="finding on MC Prop 4.2 (2026-09-23): the paper's proof establishes "
        "P|_{T*M} = id and does not constrain P on TM; pr₂ is the B = 0 representative "
        "and depends on the splitting τ",
    )
    return chain, theorem
