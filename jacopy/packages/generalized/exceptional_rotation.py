"""
The exceptional Ψ_Π ROTATION (PDF item 13k; the long-reported
missing slice): the exceptional Courant bracket on
``TM ⊕ Λ² ⊕ Λ⁵`` conjugated by the E6 twist (4.10)-(4.12),

    [e₁, e₂]' := Ψ_Π⁻¹ [Ψ_Π e₁, Ψ_Π e₂]_exc,

with the RESULTING bracket's terms SEPARATED (the 13j "separate
every term" demand, on the triple decomposition):

* the two FORM slots of the rotated bracket are the original ones
  computed in the SHIFTED direction — the deviation is exactly the
  Π-block ``ΔU = Π₃ω₂ + Π₆ω₅ + (Π₃⊛Π₃)ω₅`` acting through ``ℒ``
  and ``ι`` (:func:`prove_rotated_form_decomposition`): the tilde-
  flavoured terms of the exceptional Drinfel'd algebroid are BORN
  here, term by term;
* the M-theory cross-term ``−η₂ ∧ dω₂`` is Ψ_Π-INVARIANT (it reads
  only the untouched form slots);
* the VECTOR slot separates as the shifted Lie bracket minus the
  Π-block of the rotated forms
  (:func:`prove_rotated_vector_decomposition`) — the (4.11)
  inverse, exposed.

Together with :mod:`general_twist` (the abstract 13j transport
suite) this delivers the full "rotate and separate" programme; the
triple split (13k) is exercised on the genuine exceptional data.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.cartan import L as _L
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.drinfeld.examples import (
    boxtimes,
    exceptional_courant_bracket,
    exceptional_pi_twist,
    exceptional_pi_twist_inverse,
)


def _iota(X: Expr, x: Expr) -> Expr:
    return Act(Interior(X), x)


def pi_block(N3, N6, om2: Expr, om5: Expr) -> Expr:
    """``ΔU(ω₂, ω₅) = Π₃ω₂ + Π₆ω₅ + (Π₃⊛Π₃)ω₅`` — the vector
    shift of the (4.10) twist."""
    return Sum(
        N3.sharp_vf(om2),
        N6.sharp_vf(om5),
        boxtimes(N3, om5),
    )


def rotated_exceptional_bracket(
    N3,
    N6,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
) -> Tuple[Expr, Expr, Expr]:
    """``Ψ_Π⁻¹ [Ψ_Π e₁, Ψ_Π e₂]_exc`` as a component triple —
    the 13j procedure instantiated on the exceptional data."""
    p1 = exceptional_pi_twist(N3, N6, U, om2, om5)
    p2 = exceptional_pi_twist(N3, N6, V, et2, et5)
    b = exceptional_courant_bracket(*p1, *p2)
    return exceptional_pi_twist_inverse(N3, N6, *b)


def _engine(N3, registry):
    from jacopy.central.calculus import (
        OperatorSlotAdditivityDefinition,
    )
    from jacopy.central.objects.multivector_interior import (
        MultivectorInteriorLinearityDefinition,
    )
    from jacopy.packages.drinfeld.examples import _exc_engine
    from jacopy.packages.poisson.nambu import (
        NambuSharpLinearityDefinition,
    )

    eng = _exc_engine(N3, registry)
    eng.register(OperatorSlotAdditivityDefinition())
    eng.register(
        MultivectorInteriorLinearityDefinition(registry)
    )
    eng.register(
        NambuSharpLinearityDefinition(N3, registry)
    )
    return eng


def _normalize(engine, expr: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, expr, registry)


def _zero_theorem(
    name,
    statement,
    diffs,
    engine,
    registry,
    *,
    from_axioms,
    notes,
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


def prove_rotated_form_decomposition(
    N3,
    N6,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The FORM slots of the rotated bracket separate as

    ``form₂' = form₂ + ℒ_{ΔU} η₂ − ι_{ΔV} dω₂``
    ``form₅' = form₅ + ℒ_{ΔU} η₅ − ι_{ΔV} dω₅``

    with ``ΔU = Π₃ω₂ + Π₆ω₅ + ⊛ω₅`` (and ``ΔV`` its e₂-twin) —
    every term of the 13j "separate d', ℒ̃' …" demand, concrete:
    the tilde-flavoured pieces of the exceptional Drinfel'd bracket
    are exactly the Π-block acting through ℒ and ι. The M-theory
    cross-term ``−η₂∧dω₂`` is Ψ_Π-INVARIANT (it never sees the
    vector slot)."""
    engine = _engine(N3, registry)
    _, r2, r5 = rotated_exceptional_bracket(
        N3, N6, U, om2, om5, V, et2, et5
    )
    _, o2, o5 = exceptional_courant_bracket(
        U, om2, om5, V, et2, et5
    )
    dU = pi_block(N3, N6, om2, om5)
    dV = pi_block(N3, N6, et2, et5)
    pred2 = Sum(
        o2, _L(dU, et2), Neg(_iota(dV, d(om2)))
    )
    pred5 = Sum(
        o5, _L(dU, et5), Neg(_iota(dV, d(om5)))
    )
    return _zero_theorem(
        "exceptional_rotation_form_decomposition",
        "form₂' = form₂ + ℒ_{ΔU}η₂ − ι_{ΔV}dω₂ and "
        "form₅' = form₅ + ℒ_{ΔU}η₅ − ι_{ΔV}dω₅ — the rotated "
        "form slots separate into the original plus the Π-block "
        "acting through ℒ and ι; the −η₂∧dω₂ cross-term is "
        "Ψ_Π-invariant (13j/13k on the exceptional triple)",
        [Sum(r2, Neg(pred2)), Sum(r5, Neg(pred5))],
        engine,
        registry,
        from_axioms=(
            "exceptional bracket + (4.10)/(4.11) twist "
            "definitions",
            "ℒ/ι direction-additivity (operator slot "
            "additivity)",
        ),
        notes="E6 paper (4.10)-(4.15) territory, term-separated",
        labels=(
            "2-form slot decomposition normalizes to 0",
            "5-form slot decomposition normalizes to 0",
        ),
    )


def prove_rotated_vector_decomposition(
    N3,
    N6,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The VECTOR slot of the rotated bracket separates as

    ``vec' = [U + ΔU, V + ΔV]_Lie − Π₃(form₂') − Π₆(form₅')
             − ⊛(form₅')``

    — the (4.11) inverse block applied to the rotated forms,
    exposed term by term."""
    engine = _engine(N3, registry)
    rv, r2, r5 = rotated_exceptional_bracket(
        N3, N6, U, om2, om5, V, et2, et5
    )
    dU = pi_block(N3, N6, om2, om5)
    dV = pi_block(N3, N6, et2, et5)
    pred = Sum(
        lie_bracket(Sum(U, dU), Sum(V, dV)),
        Neg(N3.sharp_vf(r2)),
        Neg(N6.sharp_vf(r5)),
        Neg(boxtimes(N3, r5)),
    )
    return _zero_theorem(
        "exceptional_rotation_vector_decomposition",
        "vec' = [U+ΔU, V+ΔV]_Lie − Π₃(form₂') − Π₆(form₅') − "
        "⊛(form₅') — the rotated vector slot separates into the "
        "shifted Lie bracket minus the Π-block of the rotated "
        "forms (13j/13k)",
        [Sum(rv, Neg(pred))],
        engine,
        registry,
        from_axioms=(
            "exceptional bracket + (4.10)/(4.11) twist "
            "definitions",
        ),
        notes="the (4.11) inverse, term-separated",
        labels=("vector slot decomposition normalizes to 0",),
    )
