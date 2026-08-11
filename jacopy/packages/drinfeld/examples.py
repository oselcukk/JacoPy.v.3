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
