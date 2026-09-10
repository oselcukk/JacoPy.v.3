"""
The EXCEPTIONAL DRINFEL'D DOUBLE identification (the question left
open at the end of ``examples/bracket_twist_research_interface``):
the exceptional Courant bracket on ``TM ⊕ Λ² ⊕ Λ⁵`` rotated by the
E6 twist ``Ψ_Π`` (4.10)-(4.14) IS the Drinfel'd double of
``A = TM`` and ``Z = Λ² ⊕ Λ⁵`` built from the block map

    Π : Z → TM,   Π(ω₂, ω₅) = Π₃ω₂ + Π̂ω₅,   Π̂ := Π₆ + Π₃⊛Π₃,

up to the DERIVED TWIST ``R′`` — exactly as the Ψ_Π-twisted Dorfman
bracket is the Nambu double up to ``R′(ω,η) = [Πω,Πη] − Π[ω,η]_Kos``
(:func:`jacopy.packages.drinfeld.twist.prove_pi_twist_vec_is_nambu_plus_r`).
Everything here is DECLARATION-FREE unless stated: ``Π₃`` and ``Π₆``
are arbitrary (the twist does not know whether they are Nambu-
Poisson).

* the Z-BRACKET born from the rotation (:func:`exceptional_z_bracket`)
  is the higher Koszul bracket of the block map on both slots plus the
  M-theory cross-term in the 5-slot:

      [ω, η]_Z = ( ℒ_{Πω}η₂ − ι_{Πη}dω₂ ,
                   ℒ_{Πω}η₅ − ι_{Πη}dω₅ − η₂ ∧ dω₂ );

* the EXCEPTIONAL DOUBLE (:func:`exceptional_double`) is the Nambu
  double's formula with ``Π`` the block map (tilde Lie derivative
  ``ℒ̃_ω W = [Πω, W] + Π(ι_W dω)``, ``d̃ = −Π d``, ``ι̃_η U = ι_U η``);

* THE IDENTIFICATION (:func:`prove_rotated_is_exceptional_double_plus_r`):

      Ψ_Π⁻¹[Ψ_Π x, Ψ_Π y]_exc = [x, y]_double + ( R′(ω, η), 0, 0 ),
      R′(ω, η) := [Πω, Πη]_Lie − Π [ω, η]_Z

  — the derived twist is the ANCHOR DEFECT of the Z-bracket under the
  block map, and it is the vector part of the rotated bracket on two
  pure forms;

* the MIXED CONDITIONS (:func:`exceptional_r_twist_components`,
  :func:`prove_exceptional_r_twist_bidegree_split`): ``R′`` splits
  EXACTLY by form bidegree,

      R′₂₂(ω₂,η₂) = [Π₃ω₂,Π₃η₂] − Π₃[ω₂,η₂]_{Π₃} + Π̂(η₂ ∧ dω₂),
      R′₂₅(ω₂,η₅) = [Π₃ω₂,Π̂η₅] − Π̂(ℒ_{Π₃ω₂}η₅) + Π₃(ι_{Π̂η₅}dω₂),
      R′₅₂(ω₅,η₂) = [Π̂ω₅,Π₃η₂] − Π₃(ℒ_{Π̂ω₅}η₂) + Π̂(ι_{Π₃η₂}dω₅),
      R′₅₅(ω₅,η₅) = [Π̂ω₅,Π̂η₅] − Π̂(ℒ_{Π̂ω₅}η₅ − ι_{Π̂η₅}dω₅),

  so "rotated bracket = exceptional Drinfel'd double" is the FOUR
  conditions ``R′ᵢⱼ = 0`` (the bidegrees are independent: set the
  other slots to zero). Their faces:

  - ``R′₂₂`` is the Nambu-Poisson derived twist of ``Π₃`` PLUS the
    cross-term lifted by ``Π̂``; under the DECLARED fundamental
    identity of ``Π₃`` it reduces to ``Π̂(η₂ ∧ dω₂)``
    (:func:`prove_r22_reduces_to_lifted_cross_term`) — so a Nambu-
    Poisson ``Π₃`` makes the first condition "``Π₆ + Π₃⊛Π₃`` kills the
    5-forms ``η₂ ∧ dω₂``": the exceptional structure couples the
    hexavector to the trivector through the M-theory term;
  - ``R′₂₅`` is the ``ℒ̃``-EQUIVARIANCE defect of ``Π̂``:
    ``ℒ̃_{ω₂}(Π̂η₅) − Π̂(ℒ_{Π₃ω₂}η₅)`` with the ``Π₃`` tilde Lie
    derivative (:func:`prove_r25_is_lie_tilde_equivariance_defect`);
  - ``R′₅₅`` is the Nambu-type derived twist of the composite
    hexavector map ``Π̂`` (its bracket-morphism fundamental identity
    when ``Π̂`` is read as a single 6-vector — the (4.13) sum);

* the SYMMETRIC PART (:func:`prove_exceptional_r_twist_symmetric_part`):

      R′(ω,η) + R′(η,ω) = −Π₃ d P₂′(ω,η) − Π̂ d P₅′(ω,η)

  with ``P′`` the TRANSPORTED pairing ``⟨Ψ_Πω, Ψ_Πη⟩`` on pure forms
  (``P₂′ = ι_{Πω}η₂ + ι_{Πη}ω₂``, ``P₅′ = ι_{Πω}η₅ + ι_{Πη}ω₅ −
  ω₂∧η₂``) — declaration-free; hence the Drinfel'd conditions force
  ``Π D′⟨ω,η⟩ = 0``, the exceptional face of the ``g_Z``-family
  consequence of the fundamental identity
  (:class:`jacopy.packages.drinfeld.tilde_calculus.FISharpPairingSwapDefinition`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.cartan import L as _L
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.drinfeld.examples import boxtimes
from jacopy.packages.generalized.exceptional_rotation import (
    pi_block,
    rotated_exceptional_bracket,
)


def _iota(X: Expr, x: Expr) -> Expr:
    return Act(Interior(X), x)


# ------------------------------------------------------------------- #
# Constructions                                                        #
# ------------------------------------------------------------------- #


def pi_hat(N3, N6, om5: Expr) -> Expr:
    """``Π̂ω₅ := Π₆ω₅ + (Π₃⊛Π₃)ω₅`` — the hexavector block of the
    (4.10) twist (the (4.13) sum), as a map ``Λ⁵ → TM``."""
    return Sum(N6.sharp_vf(om5), boxtimes(N3, om5))


def exceptional_z_bracket(
    N3, N6, om2: Expr, om5: Expr, et2: Expr, et5: Expr
) -> Tuple[Expr, Expr]:
    """The Z-bracket born from the rotation (the rotated bracket on
    two PURE forms, form slots):

        [ω,η]_Z = ( ℒ_{Πω}η₂ − ι_{Πη}dω₂ ,
                    ℒ_{Πω}η₅ − ι_{Πη}dω₅ − η₂∧dω₂ )

    — the higher Koszul bracket of the block map ``Π`` on each slot
    plus the Ψ_Π-invariant M-theory cross-term."""
    P1 = pi_block(N3, N6, om2, om5)
    P2 = pi_block(N3, N6, et2, et5)
    return (
        Sum(_L(P1, et2), Neg(_iota(P2, d(om2)))),
        Sum(
            _L(P1, et5),
            Neg(_iota(P2, d(om5))),
            Neg(Wedge(et2, d(om2))),
        ),
    )


def exceptional_lie_tilde(
    N3, N6, om2: Expr, om5: Expr, W: Expr
) -> Expr:
    """``ℒ̃_ω W = [Πω, W]_Lie + Π(ι_W dω)`` for the block map —
    ``Π(ι_W dω) = Π₃(ι_W dω₂) + Π̂(ι_W dω₅)`` (the p ≥ 2 tilde Lie
    derivative of :func:`jacopy.packages.drinfeld.double.lie_tilde_nambu`,
    slot by slot)."""
    return Sum(
        lie_bracket(pi_block(N3, N6, om2, om5), W),
        N3.sharp_vf(_iota(W, d(om2))),
        pi_hat(N3, N6, _iota(W, d(om5))),
    )


def exceptional_double(
    N3,
    N6,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
) -> Tuple[Expr, Expr, Expr]:
    """The exceptional Drinfel'd double ``[x, y]`` on
    ``TM ⊕ (Λ² ⊕ Λ⁵)`` — the Nambu double's formula with the block
    map ``Π`` as the Z-side anchor:

        vector: [U,V] + ℒ̃_ω V − ℒ̃_η U + d̃ ι̃_η U     (d̃ = −Πd, ι̃_η U = ι_U η)
        forms:  [ω,η]_Z + ℒ_U η − ι_V dω               (slot by slot)."""
    z2, z5 = exceptional_z_bracket(N3, N6, om2, om5, et2, et5)
    vec = Sum(
        lie_bracket(U, V),
        exceptional_lie_tilde(N3, N6, om2, om5, V),
        Neg(exceptional_lie_tilde(N3, N6, et2, et5, U)),
        Neg(N3.sharp_vf(d(_iota(U, et2)))),
        Neg(pi_hat(N3, N6, d(_iota(U, et5)))),
    )
    return (
        vec,
        Sum(z2, _L(U, et2), Neg(_iota(V, d(om2)))),
        Sum(z5, _L(U, et5), Neg(_iota(V, d(om5)))),
    )


def exceptional_r_twist(
    N3, N6, om2: Expr, om5: Expr, et2: Expr, et5: Expr
) -> Expr:
    """The DERIVED TWIST ``R′(ω,η) = [Πω, Πη]_Lie − Π[ω,η]_Z`` — the
    anchor defect of the Z-bracket under the block map (vector-
    valued); the rotated bracket's vector part on two pure forms."""
    z2, z5 = exceptional_z_bracket(N3, N6, om2, om5, et2, et5)
    return Sum(
        lie_bracket(
            pi_block(N3, N6, om2, om5), pi_block(N3, N6, et2, et5)
        ),
        Neg(N3.sharp_vf(z2)),
        Neg(pi_hat(N3, N6, z5)),
    )


def exceptional_r_twist_components(
    N3, N6, om2: Expr, om5: Expr, et2: Expr, et5: Expr
) -> Dict[str, Expr]:
    """The four bidegree pieces of ``R′`` (keys ``"22"``, ``"25"``,
    ``"52"``, ``"55"``) — the MIXED Π₃/Π₆/⊛ conditions of the
    exceptional Drinfel'd algebroid, each written in closed form
    (see the module docstring)."""
    from jacopy.packages.poisson.nambu import nambu_koszul_bracket

    S3 = N3.sharp_vf
    hat = lambda x: pi_hat(N3, N6, x)  # noqa: E731
    return {
        "22": Sum(
            lie_bracket(S3(om2), S3(et2)),
            Neg(S3(nambu_koszul_bracket(N3, om2, et2))),
            hat(Wedge(et2, d(om2))),
        ),
        "25": Sum(
            lie_bracket(S3(om2), hat(et5)),
            Neg(hat(_L(S3(om2), et5))),
            S3(_iota(hat(et5), d(om2))),
        ),
        "52": Sum(
            lie_bracket(hat(om5), S3(et2)),
            Neg(S3(_L(hat(om5), et2))),
            hat(_iota(S3(et2), d(om5))),
        ),
        "55": Sum(
            lie_bracket(hat(om5), hat(et5)),
            Neg(
                hat(
                    Sum(
                        _L(hat(om5), et5),
                        Neg(_iota(hat(et5), d(om5))),
                    )
                )
            ),
        ),
    }


def transported_form_pairing(
    N3, N6, om2: Expr, om5: Expr, et2: Expr, et5: Expr
) -> Tuple[Expr, Expr]:
    """``⟨Ψ_Π ω, Ψ_Π η⟩`` for two PURE forms — the transported
    pairing's 1-form and 4-form parts:
    ``P₂′ = ι_{Πω}η₂ + ι_{Πη}ω₂``,
    ``P₅′ = ι_{Πω}η₅ + ι_{Πη}ω₅ − ω₂∧η₂``."""
    P1 = pi_block(N3, N6, om2, om5)
    P2 = pi_block(N3, N6, et2, et5)
    return (
        Sum(_iota(P1, et2), _iota(P2, om2)),
        Sum(_iota(P1, et5), _iota(P2, om5), Neg(Wedge(om2, et2))),
    )


# ------------------------------------------------------------------- #
# Engine and theorem scaffolding                                       #
# ------------------------------------------------------------------- #


def _engine(N3, N6, registry, *, declare_fi: bool = False):
    """The exceptional-rotation engine (tilde calculus of ``Π₃`` +
    slot additivity + multivector-interior and sharp linearities for
    both structures + concrete-degree wedge order); ``declare_fi``
    adds the DECLARED fundamental identity of ``Π₃`` (bracket-
    morphism face and its ``g_Z`` consequence)."""
    from jacopy.central.calculus import (
        OperatorSlotAdditivityDefinition,
    )
    from jacopy.central.objects.multivector_interior import (
        MultivectorInteriorLinearityDefinition,
    )
    from jacopy.packages.drinfeld.tilde_calculus import _tilde_engine
    from jacopy.packages.poisson.nambu import (
        NambuSharpLinearityDefinition,
    )
    from jacopy.research.engine_assembly import (
        WedgeGradedOrderDefinition,
    )

    eng = _tilde_engine(N3, registry, declare_fi=declare_fi)
    eng.register(OperatorSlotAdditivityDefinition())
    eng.register(MultivectorInteriorLinearityDefinition(registry))
    eng.register(NambuSharpLinearityDefinition(N3, registry))
    eng.register(NambuSharpLinearityDefinition(N6, registry))
    eng.register(WedgeGradedOrderDefinition(registry))
    return eng


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


def _zero_theorem(
    name: str,
    statement: str,
    diffs: Sequence[Expr],
    engine,
    registry,
    *,
    from_axioms,
    notes: str,
    labels: Sequence[str],
) -> Tuple[ProofChain, Theorem]:
    steps: List[ProofStep] = []
    for label, diff in zip(labels, diffs):
        nf = _normalize(engine, diff, registry)
        if nf != Integer(0):
            raise ProofFailure(
                f"{name}: {label} FAILS — residual "
                + nf._repr_inner()[:200]
            )
        steps.append(
            ProofStep(
                diff,
                Integer(0),
                rule=label,
                justification="engine normal form",
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


_DEFS = (
    "exceptional bracket + (4.10)/(4.11) twist definitions",
    "block map Π = (Π₃, Π₆ + Π₃⊛Π₃) with ⊛ = ½Π₃∘ι_{Π₃} (4.14)",
)


# ------------------------------------------------------------------- #
# Theorems                                                             #
# ------------------------------------------------------------------- #


def prove_rotated_is_exceptional_double_plus_r(
    N3,
    N6,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """THE IDENTIFICATION, declaration-free:

        Ψ_Π⁻¹[Ψ_Π x, Ψ_Π y]_exc = [x,y]_double + (R′(ω,η), 0, 0)

    — vector slot probed on ``h``, form slots at form level. So the
    rotated exceptional bracket IS the exceptional Drinfel'd double
    exactly when the derived twist ``R′`` vanishes (the exceptional
    analogue of "Ψ_Π-twisted Dorfman = Nambu double ⟺ Π is Nambu-
    Poisson")."""
    engine = _engine(N3, N6, registry)
    rv, r2, r5 = rotated_exceptional_bracket(
        N3, N6, U, om2, om5, V, et2, et5
    )
    dv, d2, d5 = exceptional_double(
        N3, N6, U, om2, om5, V, et2, et5
    )
    R = exceptional_r_twist(N3, N6, om2, om5, et2, et5)
    return _zero_theorem(
        "rotated_exceptional_is_double_plus_r",
        "Ψ_Π⁻¹[Ψ_Π x, Ψ_Π y]_exc = [x,y]_double + (R′(ω,η), 0, 0) "
        "with R′(ω,η) = [Πω,Πη] − Π[ω,η]_Z — the rotated exceptional "
        "bracket is the exceptional Drinfel'd double up to the derived "
        "twist (declaration-free)",
        [
            Act(Sum(rv, Neg(dv), Neg(R)), h),
            Sum(r2, Neg(d2)),
            Sum(r5, Neg(d5)),
        ],
        engine,
        registry,
        from_axioms=_DEFS
        + ("Cartan magic formula (engine rule)",),
        notes="exceptional analogue of twist.prove_pi_twist_vec_is_"
        "nambu_plus_r; E6 (4.10)-(4.14) territory",
        labels=(
            "vector slot: rotated − double − R′ on the probe "
            "normalizes to 0",
            "2-form slot: rotated − double normalizes to 0",
            "5-form slot: rotated − double normalizes to 0",
        ),
    )


def prove_exceptional_r_twist_bidegree_split(
    N3,
    N6,
    om2: Expr,
    om5: Expr,
    et2: Expr,
    et5: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``R′(ω,η) = R′₂₂ + R′₂₅ + R′₅₂ + R′₅₅`` with the closed-form
    pieces of :func:`exceptional_r_twist_components` — declaration-
    free (probe ``h``). The exceptional Drinfel'd conditions are the
    four ``R′ᵢⱼ = 0``."""
    engine = _engine(N3, N6, registry)
    parts = exceptional_r_twist_components(
        N3, N6, om2, om5, et2, et5
    )
    diff = Sum(
        exceptional_r_twist(N3, N6, om2, om5, et2, et5),
        *(Neg(p) for p in parts.values()),
    )
    return _zero_theorem(
        "exceptional_r_twist_bidegree_split",
        "R′(ω,η) = R′₂₂(ω₂,η₂) + R′₂₅(ω₂,η₅) + R′₅₂(ω₅,η₂) + "
        "R′₅₅(ω₅,η₅) — the derived twist splits exactly by form "
        "bidegree into the mixed Π₃/Π₆/⊛ conditions",
        [Act(diff, h)],
        engine,
        registry,
        from_axioms=_DEFS + ("bilinearity of Π, [·,·]_Lie, ℒ, ι",),
        notes="the four pieces vanish independently (set the other "
        "slots to zero)",
        labels=("R′ − Σ R′ᵢⱼ on the probe normalizes to 0",),
    )


def prove_r22_reduces_to_lifted_cross_term(
    N3,
    N6,
    om2: Expr,
    et2: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_fi: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """Under the DECLARED fundamental identity of ``Π₃``,

        R′₂₂(ω₂,η₂) = Π̂(η₂ ∧ dω₂),   Π̂ = Π₆ + Π₃⊛Π₃

    — the Nambu derived twist of ``Π₃`` dies and the first Drinfel'd
    condition becomes "the hexavector block kills the cross-term
    5-forms". Without the declaration the difference is the honest
    residual (``ProofFailure``)."""
    if not declare_fi:
        raise ProofFailure(
            "r22_reduces_to_lifted_cross_term: needs the declared "
            "fundamental identity of Π₃ (declare_fi=True); without it "
            "R′₂₂ − Π̂(η₂∧dω₂) is the Nambu derived twist of Π₃, "
            "which does not vanish for an arbitrary trivector"
        )
    engine = _engine(N3, N6, registry, declare_fi=True)
    parts = exceptional_r_twist_components(
        N3, N6, om2, Integer(0), et2, Integer(0)
    )
    diff = Sum(
        parts["22"], Neg(pi_hat(N3, N6, Wedge(et2, d(om2))))
    )
    return _zero_theorem(
        "r22_reduces_to_lifted_cross_term",
        "R′₂₂(ω₂,η₂) = (Π₆ + Π₃⊛Π₃)(η₂∧dω₂) under the declared "
        "fundamental identity of Π₃ — the Nambu derived twist of Π₃ "
        "vanishes and the cross-term lifted by the hexavector block "
        "remains",
        [Act(diff, h)],
        engine,
        registry,
        from_axioms=_DEFS
        + (
            "declared fundamental identity of Π₃ (bracket-morphism "
            "face) and its g_Z consequence",
        ),
        notes="a Nambu-Poisson Π₃ turns the first Drinfel'd condition "
        "into Π̂(η₂∧dω₂) = 0 for all 2-forms",
        labels=(
            "R′₂₂ − Π̂(η₂∧dω₂) on the probe normalizes to 0 "
            "(declared FI of Π₃)",
        ),
    )


def prove_r25_is_lie_tilde_equivariance_defect(
    N3,
    N6,
    om2: Expr,
    et5: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``R′₂₅(ω₂,η₅) = ℒ̃_{ω₂}(Π̂η₅) − Π̂(ℒ_{Π₃ω₂}η₅)`` with
    ``ℒ̃_{ω₂}W = [Π₃ω₂,W] + Π₃(ι_W dω₂)`` the Π₃ tilde Lie
    derivative — the second Drinfel'd condition says the hexavector
    block intertwines the tilde action on vectors with the Lie
    derivative along ``Π₃ω₂`` on 5-forms. Declaration-free."""
    from jacopy.packages.drinfeld.double import lie_tilde_nambu

    engine = _engine(N3, N6, registry)
    parts = exceptional_r_twist_components(
        N3, N6, om2, Integer(0), Integer(0), et5
    )
    hat = pi_hat(N3, N6, et5)
    pred = Sum(
        lie_tilde_nambu(N3, om2, hat),
        Neg(pi_hat(N3, N6, _L(N3.sharp_vf(om2), et5))),
    )
    return _zero_theorem(
        "r25_is_lie_tilde_equivariance_defect",
        "R′₂₅(ω₂,η₅) = ℒ̃_{ω₂}(Π̂η₅) − Π̂(ℒ_{Π₃ω₂}η₅) — the mixed "
        "(2,5) condition is the ℒ̃-equivariance of the hexavector "
        "block Π̂ = Π₆ + Π₃⊛Π₃",
        [Act(Sum(parts["25"], Neg(pred)), h)],
        engine,
        registry,
        from_axioms=_DEFS,
        notes="declaration-free identification of the (2,5) piece",
        labels=(
            "R′₂₅ − equivariance defect on the probe normalizes to 0",
        ),
    )


def prove_exceptional_r_twist_symmetric_part(
    N3,
    N6,
    om2: Expr,
    om5: Expr,
    et2: Expr,
    et5: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """``R′(ω,η) + R′(η,ω) = −Π₃ dP₂′(ω,η) − Π̂ dP₅′(ω,η)`` with
    ``P′`` the transported pairing on pure forms — declaration-free.
    Consequence: the Drinfel'd conditions force ``Π D′⟨ω,η⟩ = 0`` (the
    block map kills the exact forms of the transported pairing), the
    exceptional face of the ``g_Z`` consequence of the fundamental
    identity."""
    engine = _engine(N3, N6, registry)
    P2, P5 = transported_form_pairing(
        N3, N6, om2, om5, et2, et5
    )
    node = Sum(
        exceptional_r_twist(N3, N6, om2, om5, et2, et5),
        exceptional_r_twist(N3, N6, et2, et5, om2, om5),
        N3.sharp_vf(d(P2)),
        pi_hat(N3, N6, d(P5)),
    )
    return _zero_theorem(
        "exceptional_r_twist_symmetric_part",
        "R′(ω,η) + R′(η,ω) = −Π₃ dP₂′(ω,η) − Π̂ dP₅′(ω,η) with P′ the "
        "transported pairing ⟨Ψ_Πω, Ψ_Πη⟩ on pure forms — the "
        "symmetric part of the derived twist is the block map on the "
        "exact transported pairing (declaration-free)",
        [Act(node, h)],
        engine,
        registry,
        from_axioms=_DEFS
        + (
            "Cartan magic formula (engine rule)",
            "graded commutativity of the wedge at concrete degrees",
        ),
        notes="the Drinfel'd conditions therefore force Π D′⟨ω,η⟩ = 0",
        labels=(
            "sym R′ + Π₃dP₂′ + Π̂dP₅′ on the probe normalizes to 0",
        ),
    )
