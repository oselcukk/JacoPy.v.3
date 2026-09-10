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

The structural picture (corrected — 2026-09-09 audit, finding 2):
the R-term is VECTOR-valued, and the (2.6) anchor
``ρ(U+ω) = θ♯ω`` reads ONLY the form slot — so the R-term lands in
``ker ρ`` and the anchor morphism is UNTOUCHED by the R-flux
(:func:`prove_r_twisted_anchor_morphism`; Watamura §3 constructs
the R-twist to preserve the Courant structure under ``d_θR = 0``).
The DERIVED twist ``R′ = [θ♯·,θ♯·] − θ♯[·,·]_θ`` vanishes under
the declared Poisson condition
(:func:`prove_derived_r_vanishes_under_poisson`) — the R-flux of a
genuine Poisson structure is an independent datum, not the derived
one.

THE R-side Ševera theorem (2026-09-10, closing the 7.B.2c gap): the
Leibniz–Jacobi defect of the R-twisted bracket is EXACTLY the
Lichnerowicz–Poisson differential ``d_θR = [θ,R]_SN`` contracted
with the three forms — ``J_R − J_θ = (ι̃_ζ ι̃_η ι̃_ω d_θR, 0)`` for any
bivector and trivector, declaration-free
(:func:`prove_r_twisted_jacobi_defect_is_dtheta_r`, with
:func:`lichnerowicz_d` the intrinsic-Palais ``d_θ`` in the Nambu
dialect); with [C'1] of the untwisted structure this is *R-twisted
Courant ⟺ d_θR = 0*, the R-flux dual of ``dH = 0``
(:func:`prove_r_twisted_jacobi`).

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


def prove_r_twisted_anchor_morphism(
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
    """THE structural theorem of the R-flux (corrected — 2026-09-09
    audit, finding 2): the R-term lands in ``ker ρ``, so the anchor
    morphism SURVIVES the R-twist,

        ρ([x,y]_{D,θ,R}) = [ρ(x), ρ(y)]_Lie

    (probed on ``h``). The (2.6) anchor ``ρ(U+ω) = θ♯ω`` reads only
    the form slot, the vector-valued R-term never enters it, and
    what remains is the untwisted [C'2] — ``θ♯`` as a Koszul-to-Lie
    morphism under the declared Poisson condition (honest-fail
    without). This matches Watamura §3: the R-twist is built to
    PRESERVE the Courant structure (with ``d_θR = 0`` for the
    Jacobi family — an earlier revision wrongly reported the defect
    as the R-term, an artifact of the triangular total anchor)."""
    from jacopy.proof.strategies import ExpandAndSimplify

    _require_poisson(N)
    vec, form = r_twisted_theta_dorfman(
        N, R, U, omega, V, eta
    )
    # the anchor of the R-twisted result: the FORM slot only —
    # the R-term (in `vec`) is annihilated structurally.
    lhs = theta_anchor(N, vec, form)
    rhs = lie_bracket(
        theta_anchor(N, U, omega),
        theta_anchor(N, V, eta),
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


# ------------------------------------------------------------------ #
# The R-side Ševera theorem: Jacobi defect = d_θR (2026-09-10)         #
# ------------------------------------------------------------------ #


def _tilde_multivector_degree(e: Expr, registry) -> Optional[int]:
    """Multivector degree in the Nambu dialect: tilde contractions
    lower it by one; scalar factors are transparent."""
    from jacopy.central.tangent.schouten import multivector_degree

    if isinstance(e, Act) and isinstance(e.op, TildeInterior):
        m = _tilde_multivector_degree(e.arg, registry)
        return None if m is None else m - 1
    if isinstance(e, Neg):
        return _tilde_multivector_degree(e.arg, registry)
    if isinstance(e, Sum):
        ms = {_tilde_multivector_degree(c, registry) for c in e.children}
        return ms.pop() if len(ms) == 1 else None
    if isinstance(e, Product):
        total = 0
        for c in e.children:
            if is_scalar_function(c, registry):
                continue
            m = _tilde_multivector_degree(c, registry)
            if m is None:
                return None
            total += m
        return total
    return multivector_degree(e, registry)


def lichnerowicz_calculus(N: NambuPoissonStructure, registry=None):
    """The tilde calculus of ``θ`` in the NAMBU dialect as a
    :class:`~jacopy.central.calculus.BracketCalculus`: anchor ``θ♯``
    (``N.sharp_vf``), bracket the Koszul bracket, forms = multivectors.
    Its ``d`` is the Lichnerowicz–Poisson differential ``d_θ = [θ,·]_SN``
    — evaluated on 1-forms by the intrinsic Palais rule."""
    from jacopy.core.symbolic_degree import Degree
    from jacopy.algebra.derivation import degree_of
    from jacopy.central.calculus import BracketCalculus
    from jacopy.packages.poisson.nambu import nambu_koszul_bracket

    def form_degree(e, reg):
        m = _tilde_multivector_degree(e, reg)
        if m is None:
            raise ValueError("multivector degree undetermined")
        return Degree.const(m)

    def section_test(e):
        if _tilde_multivector_degree(e, registry) is not None:
            return False
        try:
            return degree_of(e, None) == Degree.const(1)
        except ValueError:
            return False

    return BracketCalculus(
        f"lichnerowicz-{N.pi._repr_inner()}",
        anchor=N.sharp_vf,
        bracket=lambda a, b: nambu_koszul_bracket(N, a, b),
        d_name="d_θ",
        lie_name="L̃",
        form_degree=form_degree,
        section_test=section_test,
    )


def lichnerowicz_d(N: NambuPoissonStructure, R: Expr, registry=None) -> Expr:
    """``d_θ R`` — the Lichnerowicz–Poisson differential of a
    multivector (the intrinsic ``d`` of :func:`lichnerowicz_calculus`)."""
    return Act(lichnerowicz_calculus(N, registry).d, R)


def _peel_tilde_contractions(e: Expr):
    args = []
    while isinstance(e, Act) and isinstance(e.op, TildeInterior):
        args.append(e.op.form)
        e = e.arg
    return (e, list(reversed(args))) if args else None


class TildeContractionAsEvalDefinition(Definition):
    """Tilde contractions of a multivector ARE its evaluations
    (definitional, the tilde face of interior-product evaluation):

        ⟨γ, ι̃_b ι̃_a V⟩ → V(a, b, γ),    (ι̃_b ι̃_a V)(h) → V(a, b, dh),
        β(Y…, ι̃_b ι̃_a V) → V(a, b, ι_{Y…} β)

    for a contraction that is a VECTOR (multivector degree 1)."""

    name = (
        "tilde contraction = evaluation: ⟨γ, ι̃…V⟩ = V(…, γ), "
        "(ι̃…V)(h) = V(…, dh), β(Y…, ι̃…V) = V(…, ι_{Y…}β)"
    )
    anchor = None

    def __init__(self, registry=None) -> None:
        self._r = registry

    def _is_contraction_vector(self, e: Expr) -> bool:
        return (
            _peel_tilde_contractions(e) is not None
            and _tilde_multivector_degree(e, self._r) == 1
        )

    def matches(self, e: Expr) -> bool:
        from jacopy.core.multi_eval import MultiEval
        from jacopy.core.pairing import Pairing
        from jacopy.central.objects import PVector

        if isinstance(e, Pairing):
            return self._is_contraction_vector(e.X)
        if isinstance(e, Act) and is_scalar_function(e.arg, self._r):
            return self._is_contraction_vector(e.op)
        if isinstance(e, MultiEval) and e.args:
            return self._is_contraction_vector(e.args[-1]) and not isinstance(
                e.head, PVector
            )
        return False

    def rewrite(self, e: Expr) -> Expr:
        from jacopy.core.multi_eval import MultiEval
        from jacopy.core.pairing import Pairing
        from jacopy.central.objects.interior import Interior
        from jacopy.central.tangent.exterior import d

        if isinstance(e, Pairing):
            V, args = _peel_tilde_contractions(e.X)
            return MultiEval(V, *args, e.alpha, alternating=True, slot_kind="covector")
        if isinstance(e, Act):
            V, args = _peel_tilde_contractions(e.op)
            return MultiEval(V, *args, d(e.arg), alternating=True, slot_kind="covector")
        V, args = _peel_tilde_contractions(e.args[-1])
        beta = e.head
        for Y in e.args[:-1]:
            beta = Act(Interior(Y), beta)
        return MultiEval(V, *args, beta, alternating=True, slot_kind="covector")


class ThetaEvalOfContractionInteriorDefinition(Definition):
    """``θ(a, ι_v β) → −V(…, ι_{θ♯a} β)`` for a tilde-contraction
    vector ``v = ι̃…V`` — derived from ``θ(a,b) = ⟨b, θ♯a⟩``, interior
    evaluation ``(ι_v β)(Y) = β(v, Y)`` and alternation (``θ(ι_v β,
    a)`` carries the opposite sign)."""

    def __init__(self, N: NambuPoissonStructure, registry=None) -> None:
        from jacopy.core.multi_eval import MultiEval

        self._N = N
        self._r = registry
        self.anchor = MultiEval
        self.name = (
            "θ(a, ι_v β) = −V(…, ι_{θ♯a} β) for a tilde-contraction v"
        )

    def _site(self, e: Expr):
        from jacopy.core.multi_eval import MultiEval
        from jacopy.central.objects.interior import Interior

        if not (isinstance(e, MultiEval) and e.head == self._N.pi and e.arity == 2):
            return None
        for i in (0, 1):
            b = e.args[i]
            if (
                isinstance(b, Act)
                and isinstance(b.op, Interior)
                and _peel_tilde_contractions(b.op.vector) is not None
                and _tilde_multivector_degree(b.op.vector, self._r) == 1
            ):
                return i
        return None

    def matches(self, e: Expr) -> bool:
        return self._site(e) is not None

    def rewrite(self, e: Expr) -> Expr:
        from jacopy.core.multi_eval import MultiEval
        from jacopy.central.objects.interior import Interior

        i = self._site(e)
        a, b = e.args[1 - i], e.args[i]
        V, args = _peel_tilde_contractions(b.op.vector)
        out = MultiEval(
            V, *args, Act(Interior(self._N.sharp_vf(a)), b.arg),
            alternating=True, slot_kind="covector",
        )
        return Neg(out) if i == 1 else out


def _lichnerowicz_engine(N, registry):
    from jacopy.central.calculus import IntrinsicDDefinition

    eng = _engine(N, registry, declare_fi=False)
    eng.register(IntrinsicDDefinition(lichnerowicz_calculus(N, registry), registry))
    eng.register(TildeContractionAsEvalDefinition(registry))
    eng.register(ThetaEvalOfContractionInteriorDefinition(N, registry))
    return eng


def prove_r_twisted_jacobi_defect_is_dtheta_r(
    N: NambuPoissonStructure,
    R: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    zeta: Expr,
    h: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    max_steps: int = 40000,
) -> Tuple[ProofChain, Theorem]:
    """THE R-side Ševera theorem [Watamura §3], declaration-free: the
    Leibniz–Jacobi defect of the R-twisted bracket relative to the
    untwisted one is EXACTLY the Lichnerowicz–Poisson differential of
    ``R`` contracted with the three forms,

        vec(J_R − J_θ)(h) = (d_θR)(ω, η, ζ, dh),   form(J_R − J_θ) = 0,

    for ANY bivector ``θ`` and trivector ``R`` (the vector component
    probed on ``h``; the form components coincide because the R-term
    is vector-valued and the form slot reads only forms). Corollary
    (with [C'1] of the untwisted structure under the Poisson
    condition, :func:`~jacopy.packages.generalized.poisson_generalized.
    prove_theta_jacobi`): the R-twisted structure is a Courant
    algebroid ⟺ ``d_θR = 0`` — the R-flux dual of ``dH = 0``."""
    from jacopy.core.multi_eval import MultiEval
    from jacopy.central.tangent.exterior import d
    from jacopy.packages.generalized.poisson_generalized import theta_dorfman

    _require_poisson(N)
    x, y, z = (U, omega), (V, eta), (W, zeta)

    def BR(p, q):
        return r_twisted_theta_dorfman(N, R, *p, *q)

    def B0(p, q):
        return theta_dorfman(N, *p, *q)

    def jac(B, k):
        return Sum(B(x, B(y, z))[k], Neg(B(B(x, y), z)[k]), Neg(B(y, B(x, z))[k]))

    vec_defect = Act(Sum(jac(BR, 0), Neg(jac(B0, 0))), h)
    form_defect = Sum(jac(BR, 1), Neg(jac(B0, 1)))
    dtheta = MultiEval(
        lichnerowicz_d(N, R, registry), omega, eta, zeta, d(h),
        alternating=True, slot_kind="covector",
    )
    engine = _lichnerowicz_engine(N, registry)

    def nf(node):
        from jacopy.algorithms.product_rule import product_rule
        from jacopy.algorithms.simplify import simplify

        cur = node
        for _ in range(14):
            expanded, _s = engine.expand(cur, max_steps=max_steps)
            reduced = simplify(product_rule(expanded, registry), registry)
            if reduced == cur:
                break
            cur = reduced
        return cur

    steps: List[ProofStep] = []
    for label, node in (
        ("vector Jacobi defect minus (d_θR)(ω,η,ζ,dh) normalizes to 0",
         Sum(vec_defect, Neg(dtheta))),
        ("form Jacobi defect normalizes to 0 (the R-term has no form part)",
         form_defect),
    ):
        residual = nf(node)
        if residual != Integer(0):
            raise ProofFailure(
                f"r_twisted_jacobi_defect: {label} FAILS — residual "
                + residual._repr_inner()[:160]
            )
        steps.append(ProofStep(node, Integer(0), rule=label, justification="engine normal form"))
    chain = ProofChain(steps)
    theorem = Theorem(
        name="r_twisted_jacobi_defect_is_dtheta_r",
        statement=(
            "J_R − J_θ = (ι̃_ζ ι̃_η ι̃_ω d_θR, 0): the Leibniz-Jacobi defect of "
            "the R-twisted tilde-Dorfman bracket is the Lichnerowicz-Poisson "
            "differential d_θR = [θ,R]_SN contracted with the three forms "
            "(any bivector, any trivector) — hence, with [C'1] under the "
            "Poisson condition, R-twisted Courant ⟺ d_θR = 0 (Watamura §3)"
        ),
        lhs=vec_defect,
        rhs=dtheta,
        proof=chain,
        generality="generic-function",
        from_axioms=(
            "R-twisted / tilde-Dorfman + Lichnerowicz d_θ (intrinsic Palais) definitions",
            "tilde-contraction = evaluation (definitional)",
        ),
        notes="PDF 14f.iii / Watamura §3 — the R-flux dual of dH = 0",
    )
    return chain, theorem


def prove_r_twisted_jacobi(
    N: NambuPoissonStructure,
    R: Expr,
    U: Expr,
    omega: Expr,
    V: Expr,
    eta: Expr,
    W: Expr,
    zeta: Expr,
    h: Expr,
    X: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    cite_theta_jacobi: bool = True,
) -> Tuple[ProofChain, Theorem]:
    """[C'1] for the R-twisted bracket under the DECLARED Poisson
    condition AND the DECLARED closure ``d_θR = 0``. The recorded
    target IS the Jacobiator (``lhs = J_R(h)``, ``rhs = 0``; 2026-09-10
    audit, F5) and the chain shows the route:

        J_R(h) → J_θ(h) + (d_θR)(ω,η,ζ,dh)     (defect theorem, sub-proof attached)
        (d_θR)(ω,η,ζ,dh) → 0                   (declared closure instance)
        J_θ(h) → 0                             ([C'1] vector component: cited, or proven)
        form(J_R) → form(J_θ) → 0              (defect theorem; [C'1] form component)

    ``cite_theta_jacobi=False`` runs [C'1] in full (about two minutes)
    and attaches its steps. Both declarations are recorded; nothing is
    inferred."""
    from jacopy.core.multi_eval import MultiEval
    from jacopy.central.tangent.exterior import d
    from jacopy.packages.generalized.poisson_generalized import (
        prove_theta_jacobi,
        theta_dorfman,
    )

    chain_d, thm_d = prove_r_twisted_jacobi_defect_is_dtheta_r(
        N, R, U, omega, V, eta, W, zeta, h, registry=registry
    )
    x, y, z = (U, omega), (V, eta), (W, zeta)

    def BR(p, q):
        return r_twisted_theta_dorfman(N, R, *p, *q)

    def B0(p, q):
        return theta_dorfman(N, *p, *q)

    def jac(B, k):
        return Sum(B(x, B(y, z))[k], Neg(B(B(x, y), z)[k]), Neg(B(y, B(x, z))[k]))

    jr_vec, j0_vec = Act(jac(BR, 0), h), Act(jac(B0, 0), h)
    jr_form, j0_form = jac(BR, 1), jac(B0, 1)
    dtheta = MultiEval(
        lichnerowicz_d(N, R, registry), omega, eta, zeta, d(h),
        alternating=True, slot_kind="covector",
    )
    if cite_theta_jacobi:
        c1_vec = c1_form = None
        assumptions = ("declared Poisson condition ([C'1] CITED)", "declared d_θR = 0")
        cite = " (cited library theorem: prove_theta_jacobi)"
    else:
        chain_t, _ = prove_theta_jacobi(
            N, U, omega, V, eta, W, zeta, h, X, registry=registry
        )
        c1_vec, c1_form = [chain_t.steps[0]], [chain_t.steps[1]]
        assumptions = ("declared Poisson condition ([C'1] proven)", "declared d_θR = 0")
        cite = " (proven: prove_theta_jacobi, steps attached)"
    steps = [
        ProofStep(
            jr_vec, Sum(j0_vec, dtheta),
            rule="J_R(h) = J_θ(h) + (d_θR)(ω,η,ζ,dh) — the R-side Ševera defect theorem",
            justification="engine normal form of the difference (sub-proof attached)",
            children=[chain_d.steps[0]], provenance_tag="theorem",
        ),
        ProofStep(
            dtheta, Integer(0),
            rule="declared closure d_θR = 0 (instance at (ω,η,ζ,dh))",
            justification="axiom instance of the opt-in R-flux closure",
            provenance_tag="axiom",
        ),
        ProofStep(
            j0_vec, Integer(0),
            rule="[C'1] of (TM)₀ ⊕ (T*M)_θ under the declared Poisson condition, "
            "vector component on the probe" + cite,
            justification="cited" if c1_vec is None else "proven",
            children=c1_vec, provenance_tag="theorem",
        ),
        ProofStep(
            jr_form, j0_form,
            rule="form(J_R) = form(J_θ) — the R-term has no form part (defect theorem)",
            justification="engine normal form of the difference (sub-proof attached)",
            children=[chain_d.steps[1]], provenance_tag="theorem",
        ),
        ProofStep(
            j0_form, Integer(0),
            rule="[C'1] of (TM)₀ ⊕ (T*M)_θ under the declared Poisson condition, "
            "form component (paired with every X)" + cite,
            justification="cited" if c1_form is None else "proven",
            children=c1_form, provenance_tag="theorem",
        ),
    ]
    chain = ProofChain(steps)
    theorem = Theorem(
        name="r_twisted_leibniz_jacobi",
        statement=(
            "[x,[y,z]]_R = [[x,y]_R,z]_R + [y,[x,z]_R]_R for the R-twisted "
            "tilde-Dorfman bracket — [C'1] under the declared Poisson condition "
            "and the declared R-flux closure d_θR = 0 (Watamura §3)"
        ),
        lhs=jr_vec,
        rhs=Integer(0),
        proof=chain,
        generality="generic-function",
        from_axioms=assumptions + (thm_d.name,),
        notes="PDF 14f.iii / Watamura §3; lhs is the vector Jacobiator on the probe, "
        "the form Jacobiator is carried in the chain",
    )
    return chain, theorem


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
