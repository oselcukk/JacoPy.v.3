"""
Lie derivative ``L`` on TM and the Cartan relations — all theorems
(PDF items 9e, 9g; Phase 2.D).

``L_X`` is instantiated from the generic calculus with the TM pair
``(id, Lie)`` (see :mod:`jacopy.central.calculus.bracket_calculus` for
the canonical definition). Everything in this module is *proved* from
the canonical definitions of ``d``, ``L``, ``ι`` and the Lie bracket:

* Cartan magic ``L_X = d ι_X + ι_X d`` — a THEOREM here (v2 carried it
  as an axiom; the definition policy forbids that),
* ``[L_X, d] = 0``, ``[L_X, ι_Y] = ι_{[X,Y]}``,
  ``[L_X, L_Y] = L_{[X,Y]}``, ``ι_X ι_Y + ι_Y ι_X = 0``,
* ``d² = 0`` on 1-forms (closing the frontier documented in 2.C).

Some evaluations leave a residual of the shape ``⟨ω, V⟩`` where ``V``
is a combination of nested brackets that vanishes by the (proved)
Jacobi/antisymmetry identities. :func:`prove_with_bracket_identities`
handles this with a small **repair loop**: on such a failure it proves
``V = 0`` on a generic function (agreement on generators), registers
the instance as a theorem, cites it into the engine and retries — the
mechanized form of "the residual is exactly ⟨ω, Jacobi⟩".
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ExpandAndSimplify, ProofFailure
from jacopy.proof.theorems import Theorem, TheoremBook, cite
from jacopy.central.calculus import LieDerivative
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.lie_bracket import lie_bracket


def lie_derivative(X: Expr) -> LieDerivative:
    """``L_X`` — the TM Lie derivative operator (degree 0)."""
    return CARTAN_TM.lie(X)


def L(X: Expr, T: Expr) -> Act:
    """``L_X T`` — an inert node; expansion is definitional."""
    if not isinstance(X, Expr) or not isinstance(T, Expr):
        raise TypeError("L expects Expr arguments")
    return Act(CARTAN_TM.lie(X), T)


def _engine(registry: Optional[PropertyRegistry]):
    from jacopy.central.tangent.engine import tangent_engine

    return tangent_engine(registry=registry)


# --------------------------------------------------------------------- #
# The repair loop                                                        #
# --------------------------------------------------------------------- #


def _peel(expr: Expr) -> Tuple[int, Expr]:
    sign = 0
    while isinstance(expr, Neg):
        sign ^= 1
        expr = expr.arg
    return sign, expr


def _is_section_combination(expr: Expr) -> bool:
    """Soundness guard for the repair loop (audit 2, 2026-07-23).

    The agreement-on-generators argument — "``V(f) = 0`` for a generic
    ``f`` implies ``V = 0``" — is only faithful for *sections* (vector
    fields and their bracket/linear combinations). It would be FALSE
    for e.g. an interior product, which kills every function without
    being zero. This predicate enforces the invariant instead of
    trusting the slot-builders: only combinations of vector fields,
    bracket atoms, sums and negations qualify.
    """
    from jacopy.algebra.lie_bracket_vf import LieBracketVF
    from jacopy.central.objects.vector_field import VectorField
    from jacopy.core.expr import Sum as _Sum

    if isinstance(expr, (VectorField, LieBracketVF)):
        return True
    if isinstance(expr, Neg):
        return _is_section_combination(expr.arg)
    if isinstance(expr, _Sum):
        return all(_is_section_combination(c) for c in expr.children)
    return False


def prove_with_bracket_identities(
    lhs: Expr,
    rhs: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_repairs: int = 3,
    engine=None,
    max_steps: int = 1024,
) -> Tuple[ProofChain, List[Theorem]]:
    """Prove ``lhs == rhs``, repairing pairing-slot bracket residuals.

    Attempts :class:`ExpandAndSimplify` with the tangent engine. When
    the residual is (a sign times) a single ``⟨head, V⟩`` — the shape
    the pairing-collection phase produces from cancelled-out proofs
    whose only obstruction is a vector-level bracket identity — it

    1. proves ``V(f') = 0`` on the generic function ``f`` (agreement on
       generators: a vector field vanishing on all functions is zero),
    2. registers the instance ``V = 0`` as a
       :class:`~jacopy.proof.theorems.Theorem`,
    3. cites it into the engine and retries.

    Returns ``(chain, used_identities)``; the identities are also the
    caller's to store in a book. Raises the last
    :class:`ProofFailure` when the residual is not of that shape or
    the repair budget is exhausted.
    """
    engine = engine if engine is not None else _engine(registry)
    book = TheoremBook()
    used: List[Theorem] = []

    def _slot_candidates(residual: Expr) -> List[Expr]:
        """Bracket-combination slots of the residual's evaluations."""
        _, core = _peel(residual)
        parts = core.children if isinstance(core, Sum) else (core,)
        out: List[Expr] = []
        for part in parts:
            _, node = _peel(part)
            if isinstance(node, Pairing):
                out.append(node.X)
            elif isinstance(node, MultiEval):
                out.extend(node.args)
        return out

    for attempt in range(max_repairs + 1):
        try:
            chain = ExpandAndSimplify().prove(
                lhs,
                rhs,
                registry=registry,
                engine=engine,
                max_steps=max_steps,
            )
            return chain, used
        except ProofFailure as exc:
            if attempt == max_repairs or exc.residual is None:
                raise
            repaired = False
            for V in _slot_candidates(exc.residual):
                _, V_core = _peel(V)
                if any(V_core == t.lhs for t in used):
                    continue
                if V_core.is_atom:
                    continue
                if not _is_section_combination(V_core):
                    # Not a section: V(f) = 0 would NOT imply V = 0
                    # (e.g. an interior product kills all functions).
                    continue
                try:
                    sub = ExpandAndSimplify().prove(
                        Act(V_core, f),
                        Integer(0),
                        registry=registry,
                        engine=_engine(registry),
                    )
                except ProofFailure:
                    continue
                thm = Theorem(
                    name=f"bracket_identity_{len(used) + 1}",
                    statement=(
                        f"{V_core._repr_inner()} = 0 "
                        "(vector-level bracket identity)"
                    ),
                    lhs=V_core,
                    rhs=Integer(0),
                    proof=sub,
                    generality="generic-function",
                    from_axioms=("Lie bracket definition",),
                    notes="agreement on generators: V(f) = 0 for generic f",
                )
                book.add(thm)
                used.append(thm)
                cite(engine, book, thm.name)
                repaired = True
            if not repaired:
                raise
    raise ProofFailure("prove_with_bracket_identities: repair budget spent")


# --------------------------------------------------------------------- #
# d² = 0 beyond functions (closing the 2.C frontier)                     #
# --------------------------------------------------------------------- #


def prove_d_squared_zero_on_one_forms(
    omega: Expr,
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """Prove ``(d(dω))(X, Y, Z) = 0`` for a 1-form ``ω``.

    The obstruction reduces to ``⟨ω, (rearranged Jacobi)⟩``; the repair
    loop proves that bracket identity on the generic function ``f`` and
    cites it.
    """
    ddw = MultiEval(d(d(omega)), X, Y, Z, alternating=True, slot_kind="vector")
    return prove_with_bracket_identities(
        ddw, Integer(0), f, registry=registry
    )


# --------------------------------------------------------------------- #
# Cartan relations (PDF item 9g) — all THEOREMS                          #
# --------------------------------------------------------------------- #


def prove_cartan_magic_on_functions(
    X: Expr, f: Expr, *, registry: Optional[PropertyRegistry] = None
) -> ProofChain:
    """``L_X f = ι_X(df) + d(ι_X f)`` — Cartan magic on 0-forms."""
    iota = Interior(X)
    rhs = Sum(
        Act(iota, d(f)),
        Act(CARTAN_TM.d, Act(iota, f)),
    )
    return ExpandAndSimplify().prove(
        L(X, f), rhs, registry=registry, engine=_engine(registry)
    )


def prove_cartan_magic_on_one_forms(
    omega: Expr,
    X: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L_X ω, Y⟩ = ⟨ι_X(dω), Y⟩ + ⟨d(ι_X ω), Y⟩`` — Cartan magic on
    1-forms, evaluated on a generic vector field.

    THE theorem of this architecture: v2 shipped it as the axiom
    ``LieDerivativeCartanDefinition``; here it is derived from the
    canonical definitions of ``d``, ``L`` and ``ι``.
    """
    iota = Interior(X)
    lhs = Pairing(L(X, omega), Y)
    rhs = Sum(
        Pairing(Act(iota, d(omega)), Y),
        Pairing(Act(CARTAN_TM.d, Act(iota, omega)), Y),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_L_commutes_with_d_on_functions(
    X: Expr,
    f: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``⟨L_X(df), Y⟩ = ⟨d(L_X f), Y⟩`` — ``[L_X, d] = 0`` on 0-forms."""
    lhs = Pairing(L(X, d(f)), Y)
    rhs = Pairing(d(L(X, f)), Y)
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_L_iota_commutator(
    omega: Expr,
    X: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``L_X(ι_Y ω) − ι_Y(L_X ω) = ι_{[X,Y]} ω`` on a 1-form ``ω``
    (all three terms are scalars)."""
    lhs = Sum(
        L(X, Act(Interior(Y), omega)),
        Neg(Act(Interior(Y), L(X, omega))),
    )
    rhs = Act(Interior(lie_bracket(X, Y)), omega)
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_L_L_commutator_on_functions(
    X: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``L_X(L_Y f) − L_Y(L_X f) = L_{[X,Y]} f`` — the representation
    property on functions."""
    lhs = Sum(L(X, L(Y, f)), Neg(L(Y, L(X, f))))
    rhs = L(lie_bracket(X, Y), f)
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_L_L_commutator_on_one_forms(
    omega: Expr,
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """``⟨L_X L_Y ω − L_Y L_X ω, Z⟩ = ⟨L_{[X,Y]} ω, Z⟩`` on a 1-form —
    may need a vector-level bracket identity (repair loop)."""
    lhs = Sum(
        Pairing(L(X, L(Y, omega)), Z),
        Neg(Pairing(L(Y, L(X, omega)), Z)),
    )
    rhs = Pairing(L(lie_bracket(X, Y), omega), Z)
    return prove_with_bracket_identities(lhs, rhs, f, registry=registry)


def prove_iota_anticommute(
    omega: Expr,
    X: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """``ι_X(ι_Y ω) + ι_Y(ι_X ω) = 0`` on a 2-form ``ω`` — closes via
    the alternating canonical form (no bracket needed)."""
    lhs = Sum(
        Act(Interior(X), Act(Interior(Y), omega)),
        Act(Interior(Y), Act(Interior(X), omega)),
    )
    return ExpandAndSimplify().prove(
        lhs, Integer(0), registry=registry, engine=_engine(registry)
    )


# --------------------------------------------------------------------- #
# Midterm Q3 — the usual-calculus identity suite (Phase 2.G)             #
# --------------------------------------------------------------------- #


def prove_L_commutes_with_d_on_one_forms(
    omega: Expr,
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """``(L_X(dω))(Y, Z) = (d(L_X ω))(Y, Z)`` — ``[L_X, d] = 0`` on
    1-forms (the last missing Cartan relation of the 0/1-form suite).
    May need vector-level bracket identities (repair loop)."""
    lhs = MultiEval(
        L(X, d(omega)), Y, Z, alternating=True, slot_kind="vector"
    )
    rhs = MultiEval(
        d(L(X, omega)), Y, Z, alternating=True, slot_kind="vector"
    )
    return prove_with_bracket_identities(lhs, rhs, f, registry=registry)


def prove_L_d_iota_commutation(
    eta: Expr,
    U: Expr,
    W: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """Midterm Q3: ``L_U dι_W η − dι_{[U,W]} η − dι_W L_U η = 0`` on a
    1-form ``η``, evaluated on a generic ``Y`` (all three terms are
    1-forms)."""
    lhs = Sum(
        Pairing(L(U, d(Act(Interior(W), eta))), Y),
        Neg(Pairing(d(Act(Interior(lie_bracket(U, W)), eta)), Y)),
        Neg(Pairing(d(Act(Interior(W), L(U, eta))), Y)),
    )
    return prove_with_bracket_identities(
        lhs, Integer(0), f, registry=registry
    )


def prove_L_d_iota_exact(
    omega: Expr,
    V: Expr,
    W: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, List[Theorem]]:
    """Midterm Q3: ``L_W dι_V ω − dι_W dι_V ω = 0`` on a 1-form ``ω``,
    evaluated on a generic ``Y``.

    The content is Cartan magic on the exact form ``α = dι_V ω``:
    ``L_W α − dι_W α = ι_W dα = ι_W d²(ι_V ω) = 0``.
    """
    alpha = d(Act(Interior(V), omega))
    lhs = Sum(
        Pairing(L(W, alpha), Y),
        Neg(Pairing(d(Act(Interior(W), alpha)), Y)),
    )
    return prove_with_bracket_identities(
        lhs, Integer(0), f, registry=registry
    )


# --------------------------------------------------------------------- #
# Book registration                                                      #
# --------------------------------------------------------------------- #


def register_cartan_theorems(
    book: TheoremBook,
    omega: Expr,
    X: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[Theorem, ...]:
    """Prove the core Cartan relations (magic on 0- and 1-forms,
    ``[L_X, ι_Y]``, ``ιι`` anticommutation) and register them.

    ``omega`` must be a 1-form except for the ``ιι`` entry, which uses
    its own 2-form. Returns the registered theorems.
    """
    from jacopy.central.objects import forms

    (two_form,) = forms("__cartan_2f", degree=2)
    entries = (
        Theorem(
            name="cartan_magic_on_functions",
            statement="L_X f = ι_X(df) + d(ι_X f)",
            lhs=L(X, f),
            rhs=Sum(Act(Interior(X), d(f)), Act(CARTAN_TM.d, Act(Interior(X), f))),
            proof=prove_cartan_magic_on_functions(X, f, registry=registry),
            generality="generic-function",
            from_axioms=("intrinsic d (Cartan-TM)", "interior product"),
        ),
        Theorem(
            name="cartan_magic_on_one_forms",
            statement="⟨L_X ω, Y⟩ = ⟨ι_X dω + d ι_X ω, Y⟩",
            lhs=Pairing(L(X, omega), Y),
            rhs=Sum(
                Pairing(Act(Interior(X), d(omega)), Y),
                Pairing(Act(CARTAN_TM.d, Act(Interior(X), omega)), Y),
            ),
            proof=prove_cartan_magic_on_one_forms(
                omega, X, Y, registry=registry
            ),
            generality="generic-function",
            from_axioms=(
                "intrinsic d (Cartan-TM)",
                "intrinsic L (Cartan-TM)",
                "interior product",
            ),
        ),
        Theorem(
            name="L_iota_commutator",
            statement="[L_X, ι_Y] ω = ι_{[X,Y]} ω  (1-forms)",
            lhs=Sum(
                L(X, Act(Interior(Y), omega)),
                Neg(Act(Interior(Y), L(X, omega))),
            ),
            rhs=Act(Interior(lie_bracket(X, Y)), omega),
            proof=prove_L_iota_commutator(omega, X, Y, registry=registry),
            generality="generic-function",
            from_axioms=("intrinsic L (Cartan-TM)", "interior product"),
        ),
        Theorem(
            name="iota_anticommute",
            statement="ι_X ι_Y + ι_Y ι_X = 0  (2-forms)",
            lhs=Sum(
                Act(Interior(X), Act(Interior(Y), two_form)),
                Act(Interior(Y), Act(Interior(X), two_form)),
            ),
            rhs=Integer(0),
            proof=prove_iota_anticommute(two_form, X, Y, registry=registry),
            generality="generic-function",
            from_axioms=("interior product", "alternating canonical form"),
        ),
    )
    for thm in entries:
        book.add(thm)
    return entries
