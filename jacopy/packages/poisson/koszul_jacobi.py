"""
The GENERAL Koszul Jacobi identity (5.E.2b closure, 2026-09-07) —
the oldest recorded deferral of the project:

    ⟨ [α,[β,γ]_π]_π + [β,[γ,α]_π]_π + [γ,[α,β]_π]_π , X ⟩ = 0

on GENERAL (non-exact) 1-forms, under the declared ``[π,π]_SN = 0``.

Mechanism (the 6.J programme completed): a GREEDY stall-time
difference-test over a rich instance-seed space, followed by an
exact ℚ-LINEAR phase (Gaussian elimination over the monomial
vectorization) for the multi-seed combinations greedy cannot reach.
Soundness: every seed is a proven/declared zero —

* ``S_br`` — pairing-Leibniz bridges
  ``⟨c,[a♯,b♯]⟩ = a♯(π(b,c)) − π(b, ℒ_{a♯}c)`` (mechanical);
* ``2NF`` — two-normal-form sharp/interior instances (the 6.B
  recipe: one node, full-engine leg vs sharp-only leg);
* ``pairD`` — the declared ``[π,π] = 0`` slices
  ``⟨c, D(a,b)⟩ = ½[π,π](a,b,c) = 0`` with X-, bracket- and
  d-pairing lifts (congruence);
* ``jac`` — cited vector-field Jacobi combinations;
* ``blift`` / ``symD`` — pairing-Leibniz and sharp-antisymmetry
  congruences —

so any rational linear combination is zero, and each accepted
rewrite is "registered-rule normalization + cited zero". The linear
phase's result is verified by the ENGINE (NF of the residual minus
the combination must be literally 0). Honest-fail without the
declaration.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Product,
    Rational,
    Sum,
)
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure
from jacopy.central.objects.interior import Interior
from jacopy.central.tangent.exterior import CARTAN_TM, d
from jacopy.central.tangent.lie_bracket import (
    jacobi_combination_theorems,
    lie_bracket,
)
from jacopy.packages.poisson.core import (
    SharpEvaluationDefinition,
    SharpVF,
)
from jacopy.packages.poisson.koszul import (
    KoszulBracket,
    koszul_bracket,
)


def _node_size(x: Expr) -> int:
    if x.is_atom:
        n = 1
        slots = getattr(x, "rewritable_slots", None)
        if slots:
            for sl in slots:
                n += _node_size(sl)
        return n
    n = 1
    for c in x.children:
        n += _node_size(c)
    return n


def _rich_engine(P, registry, *, declare: bool):
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        InteriorVectorLinearityDefinition,
        MultiEvalArgLinearityDefinition,
    )
    from jacopy.packages.drinfeld.twist import (
        MagicFormulaDefinition,
        PairingArgSplitDefinition,
    )
    from jacopy.packages.poisson.showcase import showcase_engine

    e = showcase_engine(
        P, registry=registry, declare_poisson=declare
    )
    e.register(MagicFormulaDefinition(registry))
    e.register(InteriorVectorLinearityDefinition(registry))
    e.register(ActExpansionDefinition(registry))
    e.register(MultiEvalArgLinearityDefinition(registry))
    e.register(PairingArgSplitDefinition())
    return e


def _normalize(engine, seed: Expr, registry) -> Expr:
    from jacopy.packages.poisson.tilde import _normalized_by

    return _normalized_by(engine, seed, registry)


def _to_vec(expr: Expr, acc=None, coeff=Fraction(1)):
    """Monomial vectorization with multilinearity bookkeeping."""
    if acc is None:
        acc = {}
    if expr == Integer(0):
        return acc
    if isinstance(expr, Sum):
        for c in expr.children:
            _to_vec(c, acc, coeff)
        return acc
    if isinstance(expr, Neg):
        return _to_vec(expr.arg, acc, -coeff)
    if isinstance(expr, Product) and expr.children:
        head = expr.children[0]
        num = None
        if isinstance(head, Integer):
            num = Fraction(int(head._repr_inner()))
        elif isinstance(head, Rational):
            num = Fraction(head._repr_inner())
        if num is not None:
            rest = expr.children[1:]
            core = (
                rest[0] if len(rest) == 1 else Product(*rest)
            )
            return _to_vec(core, acc, coeff * num)
    if isinstance(expr, Pairing):
        if isinstance(expr.X, Sum):
            for t in expr.X.children:
                _to_vec(Pairing(expr.alpha, t), acc, coeff)
            return acc
        if isinstance(expr.X, Neg):
            return _to_vec(
                Pairing(expr.alpha, expr.X.arg), acc, -coeff
            )
        if isinstance(expr.alpha, Sum):
            for t in expr.alpha.children:
                _to_vec(Pairing(t, expr.X), acc, coeff)
            return acc
        if isinstance(expr.alpha, Neg):
            return _to_vec(
                Pairing(expr.alpha.arg, expr.X), acc, -coeff
            )
    if isinstance(expr, MultiEval):
        for k, sl in enumerate(expr.args):
            if isinstance(sl, (Sum, Neg)):
                if isinstance(sl, Neg):
                    new_args = (
                        expr.args[:k]
                        + (sl.arg,)
                        + expr.args[k + 1 :]
                    )
                    return _to_vec(
                        expr.with_args(*new_args), acc, -coeff
                    )
                for t in sl.children:
                    new_args = (
                        expr.args[:k]
                        + (t,)
                        + expr.args[k + 1 :]
                    )
                    _to_vec(
                        expr.with_args(*new_args), acc, coeff
                    )
                return acc
    if isinstance(expr, Act):
        if isinstance(expr.arg, Sum):
            for t in expr.arg.children:
                _to_vec(Act(expr.op, t), acc, coeff)
            return acc
        if isinstance(expr.arg, Neg):
            return _to_vec(
                Act(expr.op, expr.arg.arg), acc, -coeff
            )
    key = expr._repr_inner()
    acc[key] = acc.get(key, Fraction(0)) + coeff
    return acc


def _gauss_solve(rows, target, keys):
    m = len(rows)
    A = [
        [rows[i].get(k, Fraction(0)) for i in range(m)]
        + [target.get(k, Fraction(0))]
        for k in keys
    ]
    piv_cols = []
    r = 0
    for col in range(m):
        p = None
        for rr in range(r, len(A)):
            if A[rr][col] != 0:
                p = rr
                break
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][col]
        A[r] = [x / pv for x in A[r]]
        for rr in range(len(A)):
            if rr != r and A[rr][col] != 0:
                fac = A[rr][col]
                A[rr] = [
                    x - fac * y for x, y in zip(A[rr], A[r])
                ]
        piv_cols.append(col)
        r += 1
        if r == len(A):
            break
    for row in A:
        if all(x == 0 for x in row[:m]) and row[m] != 0:
            return None
    coeffs = [Fraction(0)] * m
    for ridx, col in enumerate(piv_cols):
        coeffs[col] = A[ridx][m]
    return coeffs


def _seed_instances(
    P, engine, alpha, beta, gamma, X, f, registry
) -> List[Expr]:
    pi = P.pi

    def sharp(a):
        return SharpVF(pi, a)

    def PP(x, y):
        return MultiEval(
            pi, x, y, alternating=True, slot_kind="covector"
        )

    def L(V, x):
        return Act(CARTAN_TM.lie(V), x)

    def D(a, b):
        return Sum(
            lie_bracket(sharp(a), sharp(b)),
            Neg(sharp(koszul_bracket(P, a, b))),
        )

    trip = (alpha, beta, gamma)
    pairs = [
        (a, b) for a in trip for b in trip if a is not b
    ]
    sharp_only = ExpansionEngine(
        [SharpEvaluationDefinition(P)]
    )
    seeds: List[Tuple[str, Expr]] = []
    for a in trip:
        for b in trip:
            if a is b:
                continue
            for c in trip:
                sbr = Sum(
                    Pairing(
                        c,
                        lie_bracket(sharp(a), sharp(b)),
                    ),
                    Neg(Act(sharp(a), PP(b, c))),
                    PP(b, L(sharp(a), c)),
                )
                seeds.append(("Sbr", sbr))
                seeds.append(("XSbr", Act(X, sbr)))
    for a in trip:
        for (b, c) in pairs:
            node = Pairing(
                Act(Interior(sharp(c)), d(a)), sharp(b)
            )
            n_full = _normalize(engine, node, registry)
            n_sharp = _normalize(sharp_only, node, registry)
            two_nf = Sum(n_sharp, Neg(n_full))
            seeds.append(("2NF", two_nf))
            seeds.append(("X2NF", Act(X, two_nf)))
    for (a, b) in pairs:
        Dv = D(a, b)
        for cc in trip:
            seeds.append(("pairD", Pairing(cc, Dv)))
            seeds.append(
                ("XpairD", Act(X, Pairing(cc, Dv)))
            )
            seeds.append(
                ("brD", Pairing(cc, lie_bracket(X, Dv)))
            )
            seeds.append(
                ("dpairD", Pairing(d(Pairing(cc, X)), Dv))
            )
    for a in trip:
        for b in trip:
            V = sharp(b)
            seeds.append(("Xblift", Sum(
                Pairing(a, lie_bracket(X, V)),
                Neg(Act(X, Pairing(a, V))),
                Pairing(L(X, a), V),
            )))
    for (a, b) in pairs:
        for t in jacobi_combination_theorems(
            X, sharp(a), sharp(b), f, registry=registry
        ):
            seeds.append(("jac", t.lhs))
            for cc in trip:
                seeds.append(("pjac", Pairing(cc, t.lhs)))
    for (a, b) in pairs:
        ssym = Sum(
            sharp(d(PP(a, b))), sharp(d(PP(b, a)))
        )
        for cc in trip:
            seeds.append(
                ("symD", Pairing(cc, lie_bracket(X, ssym)))
            )
            seeds.append(("symD2", Pairing(cc, ssym)))
    out: List[Expr] = []
    seen = set()
    for _tag, S in seeds:
        for sgn in ((lambda x: x), Neg):
            nf = _normalize(engine, sgn(S), registry)
            if nf == Integer(0):
                continue
            k = nf._repr_inner()
            if k in seen:
                continue
            seen.add(k)
            out.append(nf)
    return out


def prove_general_koszul_jacobi(
    P,
    alpha: Expr,
    beta: Expr,
    gamma: Expr,
    X: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    declare_poisson: bool = True,
    max_rounds: int = 40,
) -> ProofChain:
    """THE 5.E.2b theorem: the Koszul bracket satisfies the Jacobi
    identity on GENERAL 1-forms, paired against the probe ``X``,
    under the declared ``[π,π]_SN = 0`` — greedy stall-difference +
    exact ℚ-linear citation phase, engine-verified. Honest-fail with
    ``declare_poisson=False``."""
    pi = P.pi
    engine = _rich_engine(P, registry, declare=declare_poisson)
    jac = Sum(
        Pairing(
            KoszulBracket(
                pi, alpha, KoszulBracket(pi, beta, gamma)
            ),
            X,
        ),
        Pairing(
            KoszulBracket(
                pi, beta, KoszulBracket(pi, gamma, alpha)
            ),
            X,
        ),
        Pairing(
            KoszulBracket(
                pi, gamma, KoszulBracket(pi, alpha, beta)
            ),
            X,
        ),
    )
    if not declare_poisson:
        # Without the declaration the defect slices are NOT zero:
        # normalization alone must (and does) leave a residual.
        residual = _normalize(engine, jac, registry)
        raise ProofFailure(
            "general Koszul Jacobi needs the declared [π,π] = 0; "
            f"residual: {residual._repr_inner()[:120]}"
        )
    lhss = _seed_instances(
        P, engine, alpha, beta, gamma, X, f, registry
    )
    steps: List[ProofStep] = []
    residual = _normalize(engine, jac, registry)
    steps.append(ProofStep(
        jac, residual,
        rule="normalize (engine fixpoint + simplify)",
        justification="registered rules",
    ))
    for _ in range(max_rounds):
        if residual == Integer(0):
            break
        progressed = False
        for lhs in lhss:
            cand = _normalize(
                engine, Sum(residual, Neg(lhs)), registry
            )
            if _node_size(cand) < _node_size(residual):
                steps.append(ProofStep(
                    residual, cand,
                    rule="cite instance: subtract lhs = 0",
                    justification=(
                        "proven/declared zero + normalization"
                    ),
                    provenance_tag="axiom",
                ))
                residual = cand
                progressed = True
                break
        if not progressed:
            break
    if residual != Integer(0):
        # ℚ-LINEAR phase.
        target = _to_vec(residual)
        rows = [_to_vec(l) for l in lhss]
        keys = sorted(
            set(target)
            | {k for row in rows for k in row}
        )
        coeffs = _gauss_solve(rows, target, keys)
        if coeffs is not None:
            nz = [
                (i, c)
                for i, c in enumerate(coeffs)
                if c != 0
            ]

            def times(c, e):
                if c == 1:
                    return e
                if c == -1:
                    return Neg(e)
                n_, d_ = c.numerator, c.denominator
                base = e if n_ > 0 else Neg(e)
                ce = (
                    Integer(abs(n_))
                    if d_ == 1
                    else Rational(abs(n_), d_)
                )
                return Product(ce, base)

            combo = Sum(
                residual,
                *(times(-c, lhss[i]) for i, c in nz),
            )
            check = _normalize(engine, combo, registry)
            if check == Integer(0):
                steps.append(ProofStep(
                    residual, Integer(0),
                    rule=(
                        "cite ℚ-linear combination of "
                        f"{len(nz)} zero-instances "
                        "(engine-verified)"
                    ),
                    justification=(
                        "each instance is a proven/declared "
                        "zero; the combination is subtracted "
                        "and the engine confirms literal 0"
                    ),
                    provenance_tag="axiom",
                ))
                residual = Integer(0)
    if residual != Integer(0):
        raise ProofFailure(
            "general Koszul Jacobi: residual survives — "
            f"{residual._repr_inner()[:140]}"
        )
    return ProofChain(steps)
