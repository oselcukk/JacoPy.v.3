"""
Example / regression suites from the literature (Phase 6.H) —
[2409.11973 §8]: three concrete doubled structures exercised on the
6.C-6.G machinery.

* **B_n-generalized geometry (8.15)-(8.16)**: ``E = TM ⊕ C∞M ⊕
  T*M`` with triple sections ``(U, f, ω)``,

      [e₁, e₂] = [U,V]_Lie ⊕ U(g) − V(f)
                 ⊕ ℒ_U η − ℒ_V ω + dι_V ω + g·df,

  and the SO(n+1, n) pairing ``g_E = ι_Uη + ι_Vω + f·g``. Theorems:
  the symmetric part is EXACTLY ``(0, 0, d g_E)`` and the bracket is
  right-Leibniz with anchor ``ρ(e) = U`` — both declaration-free.

* **Exceptional Courant bracket (8.1)**: ``E = TM ⊕ Λ² ⊕ Λ⁵`` with
  the M-theory cross-term,

      [e₁, e₂] = [U,V]_Lie ⊕ (ℒ_U η₂ − ι_V dω₂)
                 ⊕ (ℒ_U η₅ − ι_V dω₅ − η₂ ∧ dω₂),

  whose symmetric part closes onto ``d`` of the exceptional pairing
  ``⟨e₁,e₂⟩ = ι_Uη₂ + ι_Vω₂ ⊕ ι_Uη₅ + ι_Vω₅ − ω₂∧η₂`` — the
  cross-term is the ``H``-twist of the ``A = TM ⊕ Λ⁵`` decomposition
  [eq (8.2)] and the ``R``-twist of ``A = TM ⊕ Λ⁵, Z = Λ²`` [eq
  (8.3)].

* **Atiyah algebroids (8.10)-(8.14)**: the (5.13) H-closure
  condition IS the Bianchi identity ``d^∇F = 0`` — the bridge to the
  Phase 4 metric-affine package, tested by citing its proven Bianchi
  II (`prove_bianchi_second`).
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.proof.chain import ProofChain
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.lie_bracket import lie_bracket


def _L(X: Expr, x: Expr) -> Expr:
    return Act(CARTAN_TM.lie(X), x)


def _iota(X: Expr, x: Expr) -> Expr:
    return Act(Interior(X), x)


# ------------------------------------------------------------------- #
# B_n-generalized geometry                                             #
# ------------------------------------------------------------------- #


def bn_bracket(
    U: Expr, f: Expr, omega: Expr, V: Expr, g: Expr, eta: Expr
) -> Tuple[Expr, Expr, Expr]:
    """The B_n bracket [eq (8.15)] on triple sections
    ``(U, f, ω) ∈ TM ⊕ C∞M ⊕ T*M``."""
    return (
        lie_bracket(U, V),
        Sum(Act(U, g), Neg(Act(V, f))),
        Sum(
            _L(U, eta),
            Neg(_L(V, omega)),
            d(_iota(V, omega)),
            Product(g, d(f)),
        ),
    )


def bn_pairing(
    U: Expr, f: Expr, omega: Expr, V: Expr, g: Expr, eta: Expr
) -> Expr:
    """``g_E = ι_Uη + ι_Vω + f·g`` [eq (8.16)] — the SO(n+1, n)
    pairing (scalar-valued)."""
    return Sum(
        _iota(U, eta), _iota(V, omega), Product(f, g)
    )


def _bn_engine(registry):
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        InteriorVectorLinearityDefinition,
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )

    eng = tangent_engine(registry=registry)
    eng.register(LieBracketLeibnizDefinition(registry))
    eng.register(MultiEvalArgLinearityDefinition(registry))
    eng.register(InteriorVectorLinearityDefinition(registry))
    eng.register(ActExpansionDefinition(registry))
    return eng


def prove_bn_symmetric_part(
    U: Expr,
    f: Expr,
    omega: Expr,
    V: Expr,
    g: Expr,
    eta: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, ProofChain, ProofChain]:
    """B_n symmetric part = ``(0, 0, d g_E)`` — the scalar component
    cancels, the vector component is Lie antisymmetry, and the form
    component is exactly ``d`` of the SO(n+1,n) pairing.
    Declaration-free; the form leg is evaluated on the probe ``Y``,
    the vector leg on the probe ``h = f``."""
    from jacopy.core.pairing import Pairing
    from jacopy.proof.strategies import ExpandAndSimplify

    v12, s12, f12 = bn_bracket(U, f, omega, V, g, eta)
    v21, s21, f21 = bn_bracket(V, g, eta, U, f, omega)
    chains = []
    for node in (
        Act(Sum(v12, v21), f),
        Sum(s12, s21),
        Pairing(
            Sum(
                f12,
                f21,
                Neg(d(bn_pairing(U, f, omega, V, g, eta))),
            ),
            Y,
        ),
    ):
        chains.append(
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=registry,
                engine=_bn_engine(registry),
                max_steps=max_steps,
            )
        )
    return chains[0], chains[1], chains[2]


def prove_bn_right_leibniz(
    U: Expr,
    f: Expr,
    omega: Expr,
    V: Expr,
    g: Expr,
    eta: Expr,
    s: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> Tuple[ProofChain, ProofChain, ProofChain]:
    """B_n right-Leibniz ``[e₁, s·e₂] = s[e₁,e₂] + (U s)·e₂`` with
    the anchor ``ρ(e₁) = U`` — componentwise (vector on the probe
    ``f``, scalar directly, form on the probe ``Y``).
    Declaration-free."""
    from jacopy.core.pairing import Pairing
    from jacopy.proof.strategies import ExpandAndSimplify

    vL, sL, fL = bn_bracket(
        U,
        f,
        omega,
        Product(s, V),
        Product(s, g),
        Product(s, eta),
    )
    vB, sB, fB = bn_bracket(U, f, omega, V, g, eta)
    anchor = Act(U, s)
    chains = []
    for node in (
        Act(
            Sum(
                vL,
                Neg(Sum(Product(s, vB), Product(anchor, V))),
            ),
            f,
        ),
        Sum(
            sL,
            Neg(Sum(Product(s, sB), Product(anchor, g))),
        ),
        Pairing(
            Sum(
                fL,
                Neg(
                    Sum(Product(s, fB), Product(anchor, eta))
                ),
            ),
            Y,
        ),
    ):
        chains.append(
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=registry,
                engine=_bn_engine(registry),
                max_steps=max_steps,
            )
        )
    return chains[0], chains[1], chains[2]


# ------------------------------------------------------------------- #
# Exceptional Courant bracket                                          #
# ------------------------------------------------------------------- #


def exceptional_courant_bracket(
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
) -> Tuple[Expr, Expr, Expr]:
    """The exceptional Courant bracket [eq (8.1)] on
    ``TM ⊕ Λ² ⊕ Λ⁵`` — the M-theory cross-term ``−η₂ ∧ dω₂`` lands
    in the top (5-form) slot."""
    return (
        lie_bracket(U, V),
        Sum(_L(U, et2), Neg(_iota(V, d(om2)))),
        Sum(
            _L(U, et5),
            Neg(_iota(V, d(om5))),
            Neg(Wedge(et2, d(om2))),
        ),
    )


def exceptional_pairing_two(
    U: Expr, om2: Expr, V: Expr, et2: Expr
) -> Expr:
    """The 1-form part of the exceptional pairing:
    ``ι_U η₂ + ι_V ω₂``."""
    return Sum(_iota(U, et2), _iota(V, om2))


def exceptional_pairing_five(
    U: Expr, om2: Expr, om5: Expr, V: Expr, et2: Expr, et5: Expr
) -> Expr:
    """The 4-form part: ``ι_U η₅ + ι_V ω₅ − ω₂ ∧ η₂``."""
    return Sum(
        _iota(U, et5),
        _iota(V, om5),
        Neg(Wedge(om2, et2)),
    )


def _exc_engine(N, registry):
    from jacopy.packages.drinfeld.tilde_calculus import (
        _tilde_engine,
    )

    return _tilde_engine(N, registry, declare_fi=False)


def prove_exceptional_symmetric_part(
    N,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
    slots2,
    slots5,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 60000,
) -> Tuple[ProofChain, ProofChain]:
    """Exceptional symmetric part: the 2-form component closes onto
    ``d`` of the 1-form pairing and the 5-form component onto ``d``
    of the 4-form pairing — the cross-terms combine into
    ``−d(ω₂ ∧ η₂)`` (graded Leibniz of ``d`` over the wedge).
    Declaration-free; evaluated on 2 resp. 5 vector slots."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.double import _ev

    _, f2_12, f5_12 = exceptional_courant_bracket(
        U, om2, om5, V, et2, et5
    )
    _, f2_21, f5_21 = exceptional_courant_bracket(
        V, et2, et5, U, om2, om5
    )
    chains = []
    for node in (
        _ev(
            Sum(
                f2_12,
                f2_21,
                Neg(
                    d(exceptional_pairing_two(U, om2, V, et2))
                ),
            ),
            slots2,
        ),
        _ev(
            Sum(
                f5_12,
                f5_21,
                Neg(
                    d(
                        exceptional_pairing_five(
                            U, om2, om5, V, et2, et5
                        )
                    )
                ),
            ),
            slots5,
        ),
    ):
        chains.append(
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=registry,
                engine=_exc_engine(N, registry),
                max_steps=max_steps,
            )
        )
    return chains[0], chains[1]


def prove_exceptional_right_leibniz(
    N,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
    f: Expr,
    slots2,
    slots5,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 60000,
) -> Tuple[ProofChain, ProofChain]:
    """Exceptional right-Leibniz on the two form components with
    anchor ``ρ(e₁) = U``. Declaration-free — the cross-term's
    Leibniz anomaly ``−η₂ ∧ df ∧ ω₂``-type contributions cancel
    between the ``ι_V d`` and wedge legs."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.double import _ev

    _, f2L, f5L = exceptional_courant_bracket(
        U,
        om2,
        om5,
        Product(f, V),
        Product(f, et2),
        Product(f, et5),
    )
    _, f2B, f5B = exceptional_courant_bracket(
        U, om2, om5, V, et2, et5
    )
    anchor = Act(U, f)
    chains = []
    for lhs, base, extra, slots in (
        (f2L, f2B, et2, slots2),
        (f5L, f5B, et5, slots5),
    ):
        node = _ev(
            Sum(
                lhs,
                Neg(
                    Sum(
                        Product(f, base),
                        Product(anchor, extra),
                    )
                ),
            ),
            slots,
        )
        chains.append(
            ExpandAndSimplify().prove(
                node,
                Integer(0),
                registry=registry,
                engine=_exc_engine(N, registry),
                max_steps=max_steps,
            )
        )
    return chains[0], chains[1]


def _exceptional_cross(x2: Expr, y2: Expr) -> Expr:
    """The 5-form cross-term of ``[x, y]_exc``: ``C(x, y) = −y₂ ∧ dx₂``."""
    return Neg(Wedge(y2, d(x2)))


def prove_top_slot_jacobi(
    N,
    U: Expr,
    om2: Expr,
    om5: Expr,
    V: Expr,
    et2: Expr,
    et5: Expr,
    W: Expr,
    ze2: Expr,
    ze5: Expr,
    slots_top,
    *,
    registry: Optional[PropertyRegistry] = None,
    cite_dorfman: bool = False,
    max_steps: int = 60000,
) -> Tuple[ProofChain, Tuple[str, ...]]:
    """The TOP-slot component of the Leibniz–Jacobi identity for the
    whole family ``TM ⊕ Λᵖ ⊕ Λ^{2p+1}`` with the Dorfman formula on
    both form slots and the cross-term ``C(x, y) = −y_p ∧ dx_p``
    landing in the top slot (the E₆ exceptional bracket is ``p = 2``:
    ``Λ² ⊕ Λ⁵``; ``p = 1`` is ``Λ¹ ⊕ Λ³``, and so on — the bracket
    formula :func:`exceptional_courant_bracket` is degree-agnostic,
    only the number of evaluation slots ``2p+1`` changes):

        [x,[y,z]]_top − [[x,y],z]_top − [y,[x,z]]_top = 0.

    Closed by LINEARITY instead of brute force (2026-09-09): the brute
    ``(2p+1)``-slot expansion of three nested brackets does not finish
    at ``p = 2`` (>900 s), but the top slot is ``L(x, y_top) +
    C(x_p, y_p)`` with ``L`` the Dorfman formula, so the Jacobiator
    splits EXACTLY as

        J_top = J^{Dorfman}(U,ω_top; V,η_top; W,ζ_top)
              + J^{cross}(ω_p, η_p, ζ_p; U, V, W),

    ``J^{cross} = ℒ_U C(y,z) + C(x_p,[y,z]_p) − C([x,y]_p,z_p)
    + ι_W dC(x,y) − ℒ_V C(x,z) − C(y_p,[x,z]_p)``. Three legs:

    1. the split identity, at FORM level (no slot evaluation);
    2. ``J^{cross} = 0``, at FORM level (graded Leibniz of ``ℒ``, ``ι``,
       ``d`` over the wedge + the degree-``p`` Dorfman bracket);
    3. ``J^{Dorfman} = 0`` — the degree-general Dorfman Leibniz–Jacobi
       theorem (6.C), PROVEN here on the ``2p+1`` slots (~30 s at
       ``p = 2``) unless ``cite_dorfman=True``, in which case it is
       recorded as a cited library theorem (the slow test suite proves
       the ``p = 2`` instance).

    Legs 1 and 2 are cheap for every ``p``; the arguments are named
    after the ``p = 2`` case for readability.

    PARITY (2026-09-09, found by asking "why five?"): the single
    cross-term bracket is a Leibniz bracket for EVEN ``p`` only. For
    odd ``p`` the cross Jacobiator is ``∓2·dη_p ∧ ι_W dω_p`` for every
    sign/order convention of the cross-term (and for its symmetrised
    variants), and an independent brute-force expansion on three slots
    at ``p = 1`` confirms the non-zero residual — so this prover fails
    HONESTLY there with that residual in the message. Returns the chain
    and the tuple of assumption labels."""
    from jacopy.proof.step import ProofStep
    from jacopy.packages.drinfeld.double import (
        dorfman_double,
        prove_dorfman_jacobi_form,
    )
    from jacopy.packages.poisson.tilde import _normalized_by
    from jacopy.proof.strategies import ProofFailure
    from jacopy.research.engine_assembly import assemble_engine

    exc = exceptional_courant_bracket
    j1 = exc(U, om2, om5, *exc(V, et2, et5, W, ze2, ze5))[2]
    j2 = exc(*exc(U, om2, om5, V, et2, et5), W, ze2, ze5)[2]
    j3 = exc(V, et2, et5, *exc(U, om2, om5, W, ze2, ze5))[2]
    J_exc = Sum(j1, Neg(j2), Neg(j3))

    D = dorfman_double
    d1 = D(U, om5, *D(V, et5, W, ze5))[1]
    d2 = D(*D(U, om5, V, et5), W, ze5)[1]
    d3 = D(V, et5, *D(U, om5, W, ze5))[1]
    J_dorf = Sum(d1, Neg(d2), Neg(d3))

    def dorf2(xv, x2, yv, y2):
        return Sum(_L(xv, y2), Neg(_iota(yv, d(x2))))

    C = _exceptional_cross
    yz2, xy2, xz2 = dorf2(V, et2, W, ze2), dorf2(U, om2, V, et2), dorf2(U, om2, W, ze2)
    J_cross = Sum(
        _L(U, C(et2, ze2)),
        C(om2, yz2),
        Neg(C(xy2, ze2)),
        _iota(W, d(C(om2, et2))),
        Neg(_L(V, C(om2, ze2))),
        Neg(C(et2, xz2)),
    )

    steps = []
    split = Sum(J_exc, Neg(J_dorf), Neg(J_cross))
    eng = assemble_engine(split, registry=registry, structures=(N,))
    for label, node in (
        ("top-slot Jacobiator = Dorfman Jacobiator + cross Jacobiator (linear split)", split),
        ("cross Jacobiator normalizes to 0 (form level)", J_cross),
    ):
        nf = _normalized_by(eng, node, registry)
        if nf != Integer(0):
            raise ProofFailure(
                f"top-slot Jacobi: {label} FAILS — residual "
                + nf._repr_inner()[:160]
            )
        steps.append(
            ProofStep(node, Integer(0), rule=label, justification="engine normal form")
        )
    if cite_dorfman:
        steps.append(
            ProofStep(
                J_dorf,
                Integer(0),
                rule="Dorfman top-slot Leibniz-Jacobi (cited library theorem, 6.C)",
                justification="cited: prove_dorfman_jacobi_form, degree-general",
                provenance_tag="theorem",
            )
        )
        assumptions = ("Dorfman Leibniz-Jacobi on the top slot (CITED, 6.C degree-general)",)
    else:
        chain_d, used = prove_dorfman_jacobi_form(
            U, om5, V, et5, W, ze5, _probe(registry), slots_top,
            registry=registry, max_steps=max_steps,
        )
        steps.extend(chain_d.steps)
        assumptions = tuple(t.name for t in used)
    return ProofChain(steps), assumptions


def prove_exceptional_jacobi_five(*args, **kwargs):
    """The ``p = 2`` (E₆, ``Λ² ⊕ Λ⁵``) instance of
    :func:`prove_top_slot_jacobi` — the 5-form slot of the exceptional
    Courant bracket's Leibniz–Jacobi identity."""
    return prove_top_slot_jacobi(*args, **kwargs)


def _probe(registry):
    from jacopy.central.objects import functions

    (f,) = functions("f_probe", registry=registry)
    return f


# ------------------------------------------------------------------- #
# 6.H.2 — multivector interior, the ⊛ map, decomposition readings      #
# ------------------------------------------------------------------- #


def boxtimes(N3, omega5: Expr) -> Expr:
    """``(Π₃ ⊛ Π₃)(ω₅) := ½ Π₃( ι_{Π₃} ω₅ )`` [E6 paper eq (4.14)]
    — the TM-valued pentavector of the exceptional Ψ_Π twist,
    buildable now that the PARTIAL contraction ``ι_{Π₃}: Λ⁵ → Λ²``
    exists (:class:`MultivectorInterior`). ``N3`` is the
    ``nambu_structure(p=2)`` host of the trivector ``Π₃``."""
    from jacopy.core.expr import Product, Rational
    from jacopy.central.objects.multivector_interior import (
        MultivectorInterior,
    )

    return Product(
        Rational(1, 2),
        N3.sharp_vf(
            Act(MultivectorInterior(N3.pi), omega5)
        ),
    )


def exceptional_h_reading(
    U: Expr, om2: Expr, V: Expr, et2: Expr
) -> Expr:
    """The (8.2) H-twist reading of the cross-term for the
    decomposition ``A = TM ⊕ Λ², Z = Λ⁵``:
    ``H(U+ω₂, V+η₂) = −η₂ ∧ dω₂``."""
    return Neg(Wedge(et2, d(om2)))


def prove_exceptional_h_second_entry_linear(
    U: Expr,
    om2: Expr,
    V: Expr,
    et2: Expr,
    f: Expr,
    slots5,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 60000,
) -> ProofChain:
    """(8.2)-H is C∞-linear in its SECOND entry [the (5.6) twisted
    linearity]: ``H(e₁, f·e₂) = f·H(e₁, e₂)`` — the cross-term is
    tensorial in ``η₂``. Declaration-free, on 5 slots."""
    from jacopy.core.pairing import Pairing
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.double import _ev
    from jacopy.packages.poisson.nambu import nambu_structure

    N = nambu_structure(p=2)
    from jacopy.packages.drinfeld.tilde_calculus import (
        _tilde_engine,
    )

    node = _ev(
        Sum(
            exceptional_h_reading(U, om2, V, Product(f, et2)),
            Neg(
                Product(
                    f,
                    exceptional_h_reading(U, om2, V, et2),
                )
            ),
        ),
        slots5,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


def prove_exceptional_h_first_entry_symbol(
    U: Expr,
    om2: Expr,
    V: Expr,
    et2: Expr,
    f: Expr,
    slots5,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 60000,
) -> ProofChain:
    """(8.2)-H's FIRST-entry anomaly [the (5.7) symbol map]:

        H(f·e₁, e₂) − f·H(e₁, e₂) = −η₂ ∧ df ∧ ω₂

    — the exact symbol, from the ``dω₂`` leg's Leibniz split.
    Declaration-free, on 5 slots."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.packages.drinfeld.double import _ev
    from jacopy.packages.poisson.nambu import nambu_structure
    from jacopy.packages.drinfeld.tilde_calculus import (
        _tilde_engine,
    )

    N = nambu_structure(p=2)
    node = _ev(
        Sum(
            exceptional_h_reading(
                U, Product(f, om2), V, et2
            ),
            Neg(
                Product(
                    f,
                    exceptional_h_reading(U, om2, V, et2),
                )
            ),
            Wedge(et2, Wedge(d(f), om2)),
        ),
        slots5,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_tilde_engine(N, registry, declare_fi=False),
        max_steps=max_steps,
    )


# ------------------------------------------------------------------- #
# B_n decomposition readings (8.17)/(8.21)/(8.26) — 6.H leftovers      #
# ------------------------------------------------------------------- #
#
# The SAME B_n bracket under three A ⊕ Z splits [§8.4]: the calculus
# elements/twists it induces per split, with the (5.21) symmetric-
# part decompositions as MECHANICAL theorems:
#
#   split 1  (A = TM, Z = C∞ ⊕ T*M): twistless — metric-Bourbaki
#            bialgebroid; ℒ_U(g+η) = U(g) + ℒ_Uη, ι_V(f+ω) = ι_Vω;
#            Z-bracket [f+ω, g+η]_Z = g·df with g_Z = f·g, 𝔻_Z = d.
#   split 2  (A = TM ⊕ C∞, Z = T*M): H-twist H(U+f, V+g) = g·df,
#            g_H = g·f, d_H = d — quasi metric-Bourbaki.
#   split 3  (A = TM ⊕ T*M — the Dorfman side, Z = C∞): R-twist
#            R(f,g) = g·df, g_R = f·g, d_R = d — tilde-quasi.
#
# The decomposition-dependence regression: one bracket, three twist
# readings; the (5.21) laws H(U,V) + H(V,U) = d_H g_H(U,V) and
# R(f,g) + R(g,f) = d_R g_R(f,g) close mechanically (they are the
# same scalar Leibniz fact g·df + f·dg = d(f·g)).


def bn_scalar_twist(f: Expr, g: Expr) -> Expr:
    """``g·df`` — the shared twist kernel of splits 2 and 3
    [(8.22)/(8.28)]."""
    return Product(g, d(f))


def prove_bn_twist_symmetric_part(
    f: Expr,
    g: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 10000,
) -> ProofChain:
    """(5.21) for splits 2 and 3 at once:
    ``H(e₁,e₂) + H(e₂,e₁) = d(g·f)`` — i.e. ``g·df + f·dg = d(fg)``,
    the scalar Leibniz rule as the twist's symmetric-part law.
    Declaration-free, probe ``Y``."""
    from jacopy.core.pairing import Pairing
    from jacopy.proof.strategies import ExpandAndSimplify

    node = Pairing(
        Sum(
            bn_scalar_twist(f, g),
            bn_scalar_twist(g, f),
            Neg(d(Product(f, g))),
        ),
        Y,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_bn_engine(registry),
        max_steps=max_steps,
    )


def bn_split1_z_bracket(
    f: Expr, omega: Expr, g: Expr, eta: Expr
) -> Expr:
    """Split 1's Z-bracket [(8.18)]: ``[f+ω, g+η]_Z = g·df`` (the
    form part; the scalar part vanishes)."""
    return bn_scalar_twist(f, g)


def prove_bn_split1_z_symmetric_part(
    f: Expr,
    omega: Expr,
    g: Expr,
    eta: Expr,
    Y: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 10000,
) -> ProofChain:
    """(8.19): the split-1 Z-bracket's symmetric part decomposes
    with ``g_Z(f+ω, g+η) = f·g`` and ``𝔻_Z = d`` — mechanically the
    same scalar Leibniz law."""
    from jacopy.core.pairing import Pairing
    from jacopy.proof.strategies import ExpandAndSimplify

    node = Pairing(
        Sum(
            bn_split1_z_bracket(f, omega, g, eta),
            bn_split1_z_bracket(g, eta, f, omega),
            Neg(d(Product(f, g))),
        ),
        Y,
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=_bn_engine(registry),
        max_steps=max_steps,
    )


def bn_split2_calculus(U: Expr, f: Expr, eta: Expr):
    """Split 2's calculus elements [(8.21)]: sections of
    ``A = TM ⊕ C∞`` act through their VECTOR part only —
    ``ℒ_{U+f}η = ℒ_Uη``, ``ι_{U+f}η = ι_Uη`` (structural
    identifications, returned as the pair)."""
    return _L(U, eta), _iota(U, eta)


def bn_split3_anchor_action(
    U: Expr, omega: Expr, f: Expr
) -> Expr:
    """Split 3's calculus [(8.26)]: the Dorfman side acts on the
    ``C∞`` part through the anchor only — ``ℒ̂_{U+ω}f = U(f)``,
    ``ι ≡ 0`` (structural)."""
    return Act(U, f)


# ------------------------------------------------------------------- #
# Full exceptional Ψ_Π twist with ⊛ (E6 eqs 4.10-4.14)                 #
# ------------------------------------------------------------------- #


def exceptional_pi_twist(N3, N6, U, om2, om5):
    """The exceptional twist [E6 eq (4.10)-(4.12)] on triple
    sections: ``Ψ_Π(U ⊕ ω₂ ⊕ ω₅) = (U + Π₃ω₂ + (Π₆ + Π₃⊛Π₃)ω₅)
    ⊕ ω₂ ⊕ ω₅`` — buildable since 6.H.2 (partial contraction +
    boxtimes). ``N3``/``N6`` host the trivector/hexavector."""
    return (
        Sum(
            U,
            N3.sharp_vf(om2),
            N6.sharp_vf(om5),
            boxtimes(N3, om5),
        ),
        om2,
        om5,
    )


def exceptional_pi_twist_inverse(N3, N6, U, om2, om5):
    """``Ψ_Π⁻¹`` [eq (4.11)]: minus the same Π-block."""
    return (
        Sum(
            U,
            Neg(N3.sharp_vf(om2)),
            Neg(N6.sharp_vf(om5)),
            Neg(boxtimes(N3, om5)),
        ),
        om2,
        om5,
    )


def prove_exceptional_twist_linear(
    N3,
    N6,
    U: Expr,
    om2: Expr,
    om5: Expr,
    f: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 30000,
) -> ProofChain:
    """``Ψ_Π(f·e) = f·Ψ_Π(e)`` (vector component; the form
    components are trivially scaled) — C∞-linearity of the
    exceptional twist block, including the ⊛ leg (multivector-
    interior tensoriality + sharp linearity). Probe ``h``."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.central.objects.multivector_interior import (
        MultivectorInteriorLinearityDefinition,
    )
    from jacopy.packages.poisson.nambu import (
        NambuSharpLinearityDefinition,
    )
    from jacopy.central.tangent.engine import tangent_engine

    vec_s, _, _ = exceptional_pi_twist(
        N3, N6, Product(f, U), Product(f, om2), Product(f, om5)
    )
    vec_b, _, _ = exceptional_pi_twist(N3, N6, U, om2, om5)
    eng = tangent_engine(registry=registry)
    eng.register(NambuSharpLinearityDefinition(N3, registry))
    eng.register(NambuSharpLinearityDefinition(N6, registry))
    eng.register(
        MultivectorInteriorLinearityDefinition(registry)
    )
    eng.register(ActExpansionDefinition(registry))
    eng.register(MultiEvalArgLinearityDefinition(registry))
    node = Act(
        Sum(vec_s, Neg(Product(f, vec_b))), h
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=eng,
        max_steps=max_steps,
    )


def prove_exceptional_twist_inverse(
    N3,
    N6,
    U: Expr,
    om2: Expr,
    om5: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 20000,
) -> ProofChain:
    """``Ψ_Π⁻¹ ∘ Ψ_Π = id`` on the vector component (the Π-blocks
    cancel; forms are untouched). Probe ``h``."""
    from jacopy.proof.strategies import ExpandAndSimplify
    from jacopy.central.tangent.engine import tangent_engine

    v1, f2, f5 = exceptional_pi_twist(N3, N6, U, om2, om5)
    v2, _, _ = exceptional_pi_twist_inverse(N3, N6, v1, f2, f5)
    node = Act(Sum(v2, Neg(U)), h)
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=tangent_engine(registry=registry),
        max_steps=max_steps,
    )
