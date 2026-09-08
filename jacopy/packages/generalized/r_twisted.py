"""
The R-TWISTED tilde-Dorfman bracket (Phase 7.B.2c; PDF item 14f.iii,
Watamura et al.): the R-flux deformation of the Poisson generalized
double ``(TM)₀ ⊕ (T*M)_θ`` — the exact DUAL of the H-twist
(:mod:`twisted_courant`), with the roles of forms and vectors
swapped: a trivector ``R`` contracts the two FORM parts and lands in
the VECTOR component,

    [x, y]_{D,θ,R} = [x, y]_{D,θ} + ( ι̃_η ι̃_ω R , 0 ).

The R-term is C∞-TENSORIAL and ANTISYMMETRIC in ``ω, η`` (tilde
interiors are odd, degree −1), so exactly as for ``H``:

* [C'3] right-Leibniz — no new defect (tensorial R-term),
* [C'4] symmetric part — the R-term drops out; ``D_θ`` is UNCHANGED,
* the twist commutes with skew-symmetrization (Courant relation).

The structural DIFFERENCE from the H-twist: the R-term is
VECTOR-valued, so the total anchor sees it — the anchor-morphism
defect of the R-twisted bracket is EXACTLY the R-term
(:func:`prove_r_anchor_defect`); a non-zero R breaks the Courant
axioms in the anchor/Jacobi family (Watamura's quasi-Courant
picture), while the DERIVED twist ``R′ = [θ♯·,θ♯·] − θ♯[·,·]_θ``
vanishes under the declared Poisson condition
(:func:`prove_derived_r_vanishes_under_poisson`) — recovering the
untwisted structure.

The tilde-interior structure rules
(:class:`TildeInteriorFormLinearityDefinition`,
:class:`TildeInteriorAnticommuteDefinition`) are the exact mirrors
of the untwisted interior's C∞-linearity and anticommutation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Product,
    Sum,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.proof.theorems import Theorem
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.objects.tilde_interior import TildeInterior
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.packages.drinfeld.double import canonical_pairing
from jacopy.packages.drinfeld.tilde_calculus import _tilde_engine
from jacopy.packages.generalized.poisson_generalized import (
    _normalize,
    _require_poisson,
    d_operator_theta,
    theta_anchor,
    theta_dorfman,
)
from jacopy.packages.poisson.nambu import NambuPoissonStructure


class TildeInteriorFormLinearityDefinition(Definition):
    """C∞-tensoriality of ``ι̃`` in its FORM slot:
    ``ι̃_{fω + η} = f·ι̃_ω + ι̃_η`` (the mirror of the untwisted
    interior's vector-slot linearity)."""

    anchor = Act

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry
        self.name = (
            "tilde interior form-linearity: "
            "ι̃_{fω+η} = f·ι̃_ω + ι̃_η"
        )

    def _split(self, form: Expr):
        if isinstance(form, (Sum, Neg)) or form == Integer(0):
            return True
        if (
            isinstance(form, Product)
            and len(form.children) >= 2
            and is_scalar_function(
                form.children[0], self._registry
            )
        ):
            return True
        return False

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeInterior)
            and self._split(expr.op.form)
        )

    def rewrite(self, expr: Expr) -> Expr:
        form = expr.op.form
        arg = expr.arg
        if form == Integer(0):
            return Integer(0)
        if isinstance(form, Sum):
            return Sum(
                *(
                    Act(TildeInterior(c), arg)
                    for c in form.children
                )
            )
        if isinstance(form, Neg):
            return Neg(Act(TildeInterior(form.arg), arg))
        scalar = form.children[0]
        rest = form.children[1:]
        core = rest[0] if len(rest) == 1 else Product(*rest)
        return Product(scalar, Act(TildeInterior(core), arg))


class TildeInteriorAnticommuteDefinition(Definition):
    """``ι̃_ω ι̃_η x → −ι̃_η ι̃_ω x`` when ``(ω, η)`` violates the
    canonical (repr) order — alternation of multivector slots; the
    mirror of the untwisted interior anticommutation, strictly
    order-reducing hence terminating."""

    anchor = Act

    name = (
        "tilde interior anticommutation: "
        "ι̃_ω ι̃_η = −ι̃_η ι̃_ω (canonical order)"
    )

    def _parts(self, expr: Expr):
        if not (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeInterior)
        ):
            return None
        inner = expr.arg
        sign = False
        if isinstance(inner, Neg):
            sign = True
            inner = inner.arg
        if not (
            isinstance(inner, Act)
            and isinstance(inner.op, TildeInterior)
        ):
            return None
        a = expr.op.form
        b = inner.op.form
        if a._repr_inner() <= b._repr_inner():
            return None
        return a, b, inner.arg, sign

    def matches(self, expr: Expr) -> bool:
        return self._parts(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        a, b, x, sign = self._parts(expr)
        out = Act(TildeInterior(b), Act(TildeInterior(a), x))
        return out if sign else Neg(out)


class SharpSlotAlternatingNormalizeDefinition(Definition):
    """Alternating canonicalization INSIDE a Nambu sharp's form
    slot: ``θ♯(…θ(η,ω)…) → θ♯(…−θ(ω,η)…)``.

    ``simplify``'s :func:`normalize_alternating` walks children
    only; an operator atom's slot content is reached by the ENGINE's
    slot protocol, so without this rule ``θ♯(dθ(η,ω))`` and
    ``θ♯(dθ(ω,η))`` stay distinct atoms and never cancel (the
    operator-atom slot-opacity theme; precedent: the free-ψ
    ``SharpSlotWedgeNormalizeDefinition``)."""

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        from jacopy.packages.poisson.nambu import NambuSharpVF

        self._registry = registry
        self.anchor = NambuSharpVF
        self.name = (
            "sharp-slot alternating normalization: θ♯(slot) with "
            "the slot's MultiEvals in canonical order"
        )

    def _normalized(self, form: Expr) -> Expr:
        from jacopy.algorithms.normalize_alternating import (
            normalize_alternating,
        )

        return normalize_alternating(form, self._registry)

    def matches(self, expr: Expr) -> bool:
        from jacopy.packages.poisson.nambu import NambuSharpVF

        return (
            isinstance(expr, NambuSharpVF)
            and self._normalized(expr.omega) != expr.omega
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.packages.poisson.nambu import NambuSharpVF

        norm = self._normalized(expr.omega)
        sign = False
        if isinstance(norm, Neg):
            sign = True
            norm = norm.arg
        out = NambuSharpVF(expr.pi, norm)
        return Neg(out) if sign else out


def r_term(R: Expr, omega: Expr, eta: Expr) -> Expr:
    """``R(ω, η) := ι̃_η ι̃_ω R`` — the vector-valued R-flux term of
    a trivector ``R`` (the dual of ``H(U,V) = ι_V ι_U H``)."""
    return Act(
        TildeInterior(eta), Act(TildeInterior(omega), R)
    )


def r_twisted_theta_dorfman(
    N: NambuPoissonStructure,
    R: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
) -> Tuple[Expr, Expr]:
    """The R-twisted tilde-Dorfman bracket on ``(TM)₀ ⊕ (T*M)_θ``:
    the vector component gains ``ι̃_η ι̃_ω R``."""
    vec, form = theta_dorfman(N, U, omega, V, eta)
    return Sum(vec, r_term(R, omega, eta)), form


def _engine(N, registry, *, declare_fi: bool):
    eng = _tilde_engine(N, registry, declare_fi=declare_fi)
    eng.register(
        TildeInteriorFormLinearityDefinition(registry)
    )
    eng.register(TildeInteriorAnticommuteDefinition())
    eng.register(
        SharpSlotAlternatingNormalizeDefinition(registry)
    )
    return eng


def _zero_theorem(
    name: str,
    statement: str,
    diffs,
    engine,
    registry,
    *,
    from_axioms,
    notes: str,
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
                justification=(
                    "engine normal form of the difference"
                ),
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


def prove_r_twisted_right_leibniz(
    N: NambuPoissonStructure,
    R: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'3] for the R-twisted bracket: the R-term is C∞-tensorial
    (``ι̃_{fη} ι̃_ω R = f·ι̃_η ι̃_ω R``), so the twist adds no
    Leibniz defect — checked on the twist DIFFERENCE (the untwisted
    [C'3] is the 6.D theorem)."""
    _require_poisson(N)
    engine = _engine(N, registry, declare_fi=False)
    diff = Sum(
        r_term(R, omega, Product(f, eta)),
        Neg(Product(f, r_term(R, omega, eta))),
    )
    return _zero_theorem(
        "r_twisted_right_leibniz",
        "[x, f·y]_{D,θ,R} − f·[x,y]_{D,θ,R} − (ρ(x)f)·y has NO "
        "R-contribution: ι̃_{fη}ι̃_ω R = f·ι̃_ηι̃_ω R ([C'3]; "
        "the untwisted part is the 6.D theorem)",
        [diff],
        engine,
        registry,
        from_axioms=(
            "tilde-interior C∞-tensoriality (definitional)",
        ),
        notes="PDF 14f.iii / Watamura R-flux",
        labels=("R-term Leibniz defect normalizes to 0",),
    )


def prove_r_twisted_symmetric_part(
    N: NambuPoissonStructure,
    R: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """[C'4] for the R-twisted bracket: the R-term is ANTISYMMETRIC
    (``ι̃_ηι̃_ω R + ι̃_ωι̃_η R = 0``), so the symmetric part — and
    with it ``D_θ`` — is untouched by the R-flux."""
    _require_poisson(N)
    engine = _engine(N, registry, declare_fi=False)
    diff = Sum(
        r_term(R, omega, eta), r_term(R, eta, omega)
    )
    return _zero_theorem(
        "r_twisted_symmetric_part",
        "[x,y]_{D,θ,R} + [y,x]_{D,θ,R} = 2·D_θ⟨x,y⟩₊ — the "
        "antisymmetric R-term drops out ([C'4]; the R-flux does "
        "not change D_θ)",
        [diff],
        engine,
        registry,
        from_axioms=(
            "tilde-interior anticommutation (alternation)",
        ),
        notes="PDF 14f.iii / Watamura R-flux",
        labels=("R-term symmetric part normalizes to 0",),
    )


def prove_r_anchor_defect(
    N: NambuPoissonStructure,
    R: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
    max_steps: int = 60000,
) -> ProofChain:
    """THE structural theorem of the R-flux: the anchor-morphism
    defect of the R-twisted bracket is EXACTLY the R-term,

        ρ([x,y]_{D,θ,R}) − [ρ(x), ρ(y)]_Lie = ι̃_η ι̃_ω R

    (probed on ``h``; the θ-part closes by the 6.I.4 capstone under
    the declared Poisson condition). A non-zero R therefore breaks
    the Courant anchor axiom — Watamura's quasi-Courant picture —
    while ``R = 0`` recovers the generalized-geometry double."""
    from jacopy.proof.strategies import ExpandAndSimplify

    _require_poisson(N)
    vec, form = r_twisted_theta_dorfman(
        N, R, U, omega, V, eta
    )
    lhs = Sum(vec, N.sharp_vf(form))
    rhs = Sum(
        lie_bracket(
            theta_anchor(N, U, omega),
            theta_anchor(N, V, eta),
        ),
        r_term(R, omega, eta),
    )
    node = Act(Sum(lhs, Neg(rhs)), h)
    engine = _engine(
        N, registry, declare_fi=declare_poisson
    )
    return ExpandAndSimplify().prove(
        node,
        Integer(0),
        registry=registry,
        engine=engine,
        max_steps=max_steps,
    )


def prove_derived_r_vanishes_under_poisson(
    N: NambuPoissonStructure,
    omega: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The DERIVED twist ``R′(ω,η) = [θ♯ω, θ♯η] − θ♯[ω,η]_θ``
    (eq (7.25)) vanishes under the declared Poisson condition — the
    R-flux of a genuine Poisson structure is zero, so the R-twisted
    family deforms AWAY from generalized geometry exactly when
    ``[θ,θ] ≠ 0`` [Watamura §3]."""
    from jacopy.packages.drinfeld.twist import r_twist

    _require_poisson(N)
    engine = _engine(N, registry, declare_fi=True)
    diff = r_twist(N, omega, eta)
    return _zero_theorem(
        "derived_r_vanishes_under_poisson",
        "R′(ω,η) = [θ♯ω, θ♯η] − θ♯[ω,η]_θ = 0 under the declared "
        "Poisson condition (the derived R-flux of a Poisson "
        "structure vanishes)",
        [diff],
        engine,
        registry,
        from_axioms=(
            "declared Poisson condition (FI bracket-morphism "
            "face)",
        ),
        notes="PDF 14f.iii; eq (7.25) at p = 1",
        labels=("derived R-twist normalizes to 0",),
    )


def prove_r_twisted_courant_relation(
    N: NambuPoissonStructure,
    R: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Tuple[ProofChain, Theorem]:
    """The R-twist commutes with skew-symmetrization: the R-twisted
    Courant bracket (skew form) is the R-twisted Dorfman minus
    ``D_θ⟨x,y⟩₊`` — component-wise, because the R-term is already
    skew."""
    from jacopy.packages.generalized.poisson_generalized import (
        theta_courant,
    )

    _require_poisson(N)
    engine = _engine(N, registry, declare_fi=False)
    c_vec, c_form = theta_courant(N, U, omega, V, eta)
    c_vec = Sum(c_vec, r_term(R, omega, eta))
    d_vec, d_form = r_twisted_theta_dorfman(
        N, R, U, omega, V, eta
    )
    D_vec, D_form = d_operator_theta(
        N, canonical_pairing(U, omega, V, eta)
    )
    return _zero_theorem(
        "r_twisted_courant_relation",
        "[x,y]_{C,θ,R} = [x,y]_{D,θ,R} − D_θ⟨x,y⟩₊ — the R-twist "
        "commutes with the skew-symmetrization",
        [
            Sum(c_vec, Neg(d_vec), D_vec),
            Sum(c_form, Neg(d_form), D_form),
        ],
        engine,
        registry,
        from_axioms=(
            "R-twisted bracket + pairing + D_θ definitions",
        ),
        notes="PDF 14f.iii",
        labels=(
            "vector component normalizes to 0",
            "form component normalizes to 0",
        ),
    )
