"""
The exterior derivative ``d`` on TM — derived from the Lie bracket
(PDF item 9d; Phase 2.C).

This module *instantiates* the generic bracket calculus with the
tangent-case pair ``(ρ = id, [·,·] = [·,·]_Lie)`` — per PDF item 8a the
usual ``d`` is exactly that instantiation, not separate code. The
canonical definition is the Palais intrinsic formula (see
:mod:`jacopy.central.calculus.bracket_calculus`); its special cases:

    df(X)     = X(f),
    dω(X, Y)  = X(ω(Y)) − Y(ω(X)) − ω([X, Y])        (1-form ω).

``d² = 0`` is a **theorem**. On functions it closes right here from
the definition alone (the obstruction reduces to the Lie-bracket
definition); :func:`prove_d_squared_zero_on_functions` proves it and
:func:`register_d_squared_zero` stores it in a
:class:`~jacopy.proof.theorems.TheoremBook` for citation. The ``p ≥ 1``
cases additionally need the vector-level Jacobi identity inside
pairing slots and are scheduled with the Cartan-relation pass (2.D).
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ExpandAndSimplify
from jacopy.proof.theorems import Theorem, TheoremBook
from jacopy.central.calculus import BracketCalculus, ExteriorDerivative
from jacopy.central.tangent.lie_bracket import lie_bracket


#: The TM Cartan calculus: identity anchor, Lie bracket.
CARTAN_TM = BracketCalculus(
    "Cartan-TM",
    anchor=lambda X: X,
    bracket=lie_bracket,
)


def cartan_calculus() -> BracketCalculus:
    """The TM instantiation ``(id, [·,·]_Lie)`` of the bracket calculus."""
    return CARTAN_TM


def exterior_d() -> ExteriorDerivative:
    """The exterior derivative ``d`` of the TM Cartan calculus
    (degree ``+1``)."""
    return CARTAN_TM.d


def d(omega: Expr) -> Act:
    """``dω`` — an inert node; evaluation expands via the Palais rule."""
    if not isinstance(omega, Expr):
        raise TypeError("d expects an Expr")
    return Act(CARTAN_TM.d, omega)


def _engine(registry: Optional[PropertyRegistry]):
    from jacopy.central.tangent.engine import tangent_engine

    return tangent_engine(registry=registry)


# --------------------------------------------------------------------- #
# Theorems / definitional checks                                         #
# --------------------------------------------------------------------- #


def prove_df_on_vector(
    X: Expr, f: Expr, *, registry: Optional[PropertyRegistry] = None
) -> ProofChain:
    """Prove ``⟨df, X⟩ = X(f)`` (the 0-form case of the definition)."""
    return ExpandAndSimplify().prove(
        Pairing(d(f), X),
        Act(X, f),
        registry=registry,
        engine=_engine(registry),
    )


def prove_one_form_intrinsic(
    omega: Expr,
    X: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``dω(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y])`` for a
    1-form ``ω`` (the classical special case of the Palais formula)."""
    lhs = MultiEval(d(omega), X, Y, alternating=True, slot_kind="vector")
    rhs = Sum(
        Act(X, Pairing(omega, Y)),
        Neg(Act(Y, Pairing(omega, X))),
        Neg(Pairing(omega, lie_bracket(X, Y))),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_d_squared_zero_on_functions(
    f: Expr,
    X: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``(d(df))(X, Y) = 0`` — ``d² = 0`` on functions.

    The Palais unroll of the outer ``d`` produces
    ``X(⟨df, Y⟩) − Y(⟨df, X⟩) − ⟨df, [X,Y]⟩``; the 0-form rule turns
    the pairings into directional derivatives and the Lie-bracket
    definition cancels everything — the identity is literally the
    bracket definition read backwards.
    """
    ddf = MultiEval(d(d(f)), X, Y, alternating=True, slot_kind="vector")
    return ExpandAndSimplify().prove(
        ddf, Integer(0), registry=registry, engine=_engine(registry)
    )


def register_d_squared_zero(
    book: TheoremBook,
    f: Expr,
    X: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    name: str = "d_squared_zero_on_functions",
) -> Theorem:
    """Prove ``d² = 0`` on functions and register it in ``book``.

    Returns the registered :class:`Theorem`. Its lhs is the evaluated
    node ``(ddf)(X, Y)`` and its rhs ``0``, so citing it rewrites that
    evaluation away in one step. Generality: ``generic-function`` (any
    ``f``); the ``p ≥ 1`` cases are a 2.D deliverable.
    """
    chain = prove_d_squared_zero_on_functions(f, X, Y, registry=registry)
    thm = Theorem(
        name=name,
        statement="(d∘d f)(X, Y) = 0  —  d² = 0 on functions",
        lhs=MultiEval(d(d(f)), X, Y, alternating=True, slot_kind="vector"),
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "intrinsic d (Cartan-TM)",
            "Lie bracket definition",
        ),
        notes="p ≥ 1 cases scheduled with the Cartan-relation pass (2.D).",
    )
    book.add(thm)
    return thm
