"""
The Lie bracket of vector fields ``[·,·]_Lie`` (PDF item 9a) — the TM
case of the central code.

**Canonical definition (definition policy, Phase 2).** The Lie bracket
is the commutator of derivations:

    [U, V] := U∘V − V∘U,   i.e.   [U, V](f) = U(V(f)) − V(U(f)).

Every other characterization is a *theorem* proved from this
definition:

* antisymmetry ``[U, V] = −[V, U]``,
* the Jacobi identity
  ``[U, [V, W]] + [V, [W, U]] + [W, [U, V]] = 0``,
* the Leibniz property in the second slot
  ``[U, fV] = U(f)·V + f·[U, V]``.

Representation: the bracket node is the opaque
:class:`~jacopy.algebra.lie_bracket_vf.LieBracketVF` — itself a
degree-0 :class:`~jacopy.algebra.derivation.Derivation`, so ``[U,V]``
is again a vector field that can act on functions and nest inside
further brackets. Its *definitional expansion* is the engine rule
:class:`LieBracketActionDefinition`, which fires on ``[U,V](f)`` and
produces ``U(V(f)) − V(U(f))``.

The proofs act on a generic scalar function ``f``: two vector fields
are equal iff their actions on all functions agree, so an operator
equation ``A = B`` is witnessed by ``A(f) = B(f)`` for a generic
``f`` (the agreement-on-generators route). Each helper returns the
closing :class:`~jacopy.proof.chain.ProofChain`.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition
from jacopy.proof.strategies import ExpandAndSimplify


def _engine(registry: Optional[PropertyRegistry]):
    # Late import: the engine module aggregates definitional rules from
    # this module and from definitions.py, so importing it eagerly at
    # the top would cycle.
    from jacopy.central.tangent.engine import tangent_engine

    return tangent_engine(registry=registry)


def lie_bracket(X: Expr, Y: Expr) -> LieBracketVF:
    """``[X, Y]`` — the Lie bracket of two vector fields.

    Returns the opaque :class:`LieBracketVF` node (a degree-0
    derivation). Its action on functions expands definitionally via
    :class:`LieBracketActionDefinition` in a
    :func:`tangent_engine`-driven proof.
    """
    if not isinstance(X, Expr) or not isinstance(Y, Expr):
        raise TypeError("lie_bracket arguments must be Expr")
    return LieBracketVF(X, Y)


# --------------------------------------------------------------------- #
# Definitional engine rules (TM case)                                    #
# --------------------------------------------------------------------- #


class LieBracketActionDefinition(Definition):
    """``[X, Y](f) → X(Y(f)) − Y(X(f))`` — the canonical definition.

    This is the *definitional* rule of the Lie bracket (commutator of
    derivations); it is always on in the tangent engine. It fires on
    any ``Act`` whose operator is a :class:`LieBracketVF`, including
    nested brackets (``[X, [Y, Z]](f)``), which unfold recursively
    under the engine fix-point.
    """

    name = "Lie bracket definition: [X, Y](f) = X(Y(f)) - Y(X(f))"
    anchor = Act

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Act) and isinstance(expr.op, LieBracketVF)

    def rewrite(self, expr: Expr) -> Expr:
        br = expr.op
        f = expr.arg
        return Sum(Act(br.X, Act(br.Y, f)), Neg(Act(br.Y, Act(br.X, f))))


class ScalarActAsMultiplicationDefinition(Definition):
    """``s(x) → s·x`` for a degree-0 scalar ``s`` used in operator
    position.

    The module structure of vector fields (``fV`` as an operator) puts
    scalars into compositions: ``(f·V)(g)`` unfolds to ``f(V(g))``
    under the composition pass, and this rule finishes the job as
    plain multiplication ``f·V(g)``. It only fires when the operator
    is *not* a derivation and its degree is determinably 0 (via the
    supplied registry) — genuine operators are untouched.
    """

    name = "scalar action: s(x) = s*x  (s a degree-0 scalar)"
    anchor = Act

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        op = expr.op
        if isinstance(op, Integer):
            # Numeric literals ARE degree-0 scalars: 2(x) = 2·x. The
            # zero operator is short-circuited by product_rule already,
            # and Product(0, x) folds to 0 in canonicalize anyway.
            return True
        if isinstance(op, (Derivation, Sum, Product, Neg)):
            # Derivations act; Sum/Product/Neg operator heads are
            # handled by ActOverSumOp / product_rule composition.
            return False
        try:
            return degree_of(op, self._registry) == Degree.const(0)
        except ValueError:
            return False

    def rewrite(self, expr: Expr) -> Expr:
        return Product(expr.op, expr.arg)


class BracketOrientationDefinition(Definition):
    """Canonical bracket orientation: ``[B, A] → −[A, B]`` when the
    operands are out of the structural sort order, and ``[A, A] → 0``.

    A **theorem-backed canonical form** (same family as the alternating
    argument sort and the γ index order): antisymmetry is the 2.A
    theorem :func:`prove_antisymmetry`, whose proof this rule carries
    as its ``theorem_proof_builder`` — in foundational mode the flip
    step inlines the full derivation. Canonical orientation is what
    lets ``[Z,X] ∧ Y + [X,Z] ∧ Y``-style pairs cancel structurally
    (the Schouten-Nijenhuis antisymmetry proofs live on this).
    """

    anchor = LieBracketVF
    name = "bracket orientation: [B, A] = −[A, B], [A, A] = 0"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        return expr.X == expr.Y or (
            expr.X._repr_inner() > expr.Y._repr_inner()
        )

    def rewrite(self, expr: Expr) -> Expr:
        if expr.X == expr.Y:
            return Integer(0)
        return Neg(LieBracketVF(expr.Y, expr.X))

    def theorem_proof_builder(self):
        def _builder(matched: Expr) -> "ProofChain":
            # Antisymmetry proved on a generic auxiliary function
            # (agreement on generators), for the matched operands.
            from jacopy.core.properties import Graded
            from jacopy.core.expr import Symbol

            reg = PropertyRegistry()
            aux = Symbol("f_aux")
            reg.declare(aux, Graded(degree=0))
            return prove_antisymmetry(
                matched.X, matched.Y, aux, registry=reg
            )

        return _builder


class LieBracketLeibnizDefinition(Definition):
    """The ``C^∞``-module structure of the Lie bracket — THEOREMS from
    the commutator definition, packaged as one derived rule:

    * ``[fX, Y] → f·[X,Y] − Y(f)·X``  (left Leibniz),
    * ``[X, fY] → f·[X,Y] + X(f)·Y``  (right Leibniz),
    * ``[X + X', Y] → [X,Y] + [X',Y]`` (and in the second slot),
    * ``Neg`` pull-out, ``[0, Y] → 0``.

    Each is provable by acting on a generic function and agreeing on
    generators; the proof builder derives exactly that from an engine
    WITHOUT this rule (non-circular). NOT registered in the tangent
    engine (Phase 2 proofs are pinned to bracket atoms staying inert
    at section level); packages opt in — the metric-affine engine
    needs it for the torsion/curvature tensoriality theorems
    (Phase 4.B).
    """

    anchor = LieBracketVF
    name = (
        "Lie bracket module structure (derived): [fX,Y] = f[X,Y] − "
        "Y(f)X, [X,fY] = f[X,Y] + X(f)Y, bilinear over sums"
    )

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        from jacopy.central.calculus.scalars import is_scalar_function

        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    def _splittable(self, slot: Expr) -> bool:
        if isinstance(slot, (Sum, Neg)) or slot == Integer(0):
            return True
        return self._scalar_split(slot) is not None

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, LieBracketVF) and (
            self._splittable(expr.X) or self._splittable(expr.Y)
        )

    def rewrite(self, expr: Expr) -> Expr:
        X, Y = expr.X, expr.Y
        if X == Integer(0) or Y == Integer(0):
            return Integer(0)
        if isinstance(X, Sum):
            return Sum(*(LieBracketVF(c, Y) for c in X.children))
        if isinstance(X, Neg):
            return Neg(LieBracketVF(X.arg, Y))
        split = self._scalar_split(X)
        if split is not None:
            f, core = split
            return Sum(
                Product(f, LieBracketVF(core, Y)),
                Neg(Product(Act(Y, f), core)),
            )
        if isinstance(Y, Sum):
            return Sum(*(LieBracketVF(X, c) for c in Y.children))
        if isinstance(Y, Neg):
            return Neg(LieBracketVF(X, Y.arg))
        f, core = self._scalar_split(Y)
        return Sum(
            Product(f, LieBracketVF(X, core)),
            Product(Act(X, f), core),
        )

    def theorem_proof_builder(self):
        registry = self._registry

        def _builder(matched: Expr) -> ProofChain:
            from jacopy.core.properties import Graded
            from jacopy.core.expr import Symbol
            from jacopy.proof.expansion import (
                ActOverSumOpDefinition,
                ExpansionEngine,
            )
            from jacopy.proof.step import ProofStep

            after = self.rewrite(matched)
            reg = registry if registry is not None else PropertyRegistry()
            aux = Symbol("h_aux")
            reg.declare(aux, Graded(degree=0))
            bare = ExpansionEngine(
                [
                    ActOverSumOpDefinition(),
                    LieBracketActionDefinition(),
                    ScalarActAsMultiplicationDefinition(reg),
                ]
            )
            sub = ExpandAndSimplify().prove(
                Act(matched, aux),
                Act(after, aux),
                registry=reg,
                engine=bare,
            )
            step = ProofStep(
                matched,
                after,
                rule="agreement on generators (generic function)",
                justification=(
                    "both sides act equally on the generic function "
                    "h_aux; vector fields agreeing on all functions "
                    "are equal"
                ),
            )
            for st in sub:
                step.add_child(st)
            return ProofChain([step])

        return _builder


# --------------------------------------------------------------------- #
# Theorems (PDF item 9c) — proved from the canonical definition        #
# --------------------------------------------------------------------- #


def _bare_commutator_engine(registry: Optional[PropertyRegistry]):
    """The commutator definition and nothing else — used by the proofs
    of theorems that themselves BACK engine rules (antisymmetry backs
    the orientation rule), so their derivations stay non-circular."""
    from jacopy.proof.expansion import (
        ActOverSumOpDefinition,
        ExpansionEngine,
    )

    return ExpansionEngine(
        [
            ActOverSumOpDefinition(),
            LieBracketActionDefinition(),
            ScalarActAsMultiplicationDefinition(registry),
        ]
    )


def prove_antisymmetry(
    X: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``[X, Y](f) = −[Y, X](f)`` (antisymmetry on a generic
    function) — from the commutator definition ONLY: this theorem
    backs the orientation rule, so its derivation must not use the
    full engine (the orientation rule would fire on ``[Y, X]`` and
    the proof would be circular; caught by the 4.C audit)."""
    lhs = Act(lie_bracket(X, Y), f)
    rhs = Neg(Act(lie_bracket(Y, X), f))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_bare_commutator_engine(registry)
    )


def prove_jacobi(
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove the Jacobi identity on a generic function:

    ``[X, [Y, Z]](f) + [Y, [Z, X]](f) + [Z, [X, Y]](f) = 0``.
    """
    lhs = Sum(
        Act(lie_bracket(X, lie_bracket(Y, Z)), f),
        Act(lie_bracket(Y, lie_bracket(Z, X)), f),
        Act(lie_bracket(Z, lie_bracket(X, Y)), f),
    )
    return ExpandAndSimplify().prove(
        lhs,
        Integer(0),
        registry=registry,
        engine=_engine(registry),
    )


def jacobi_combination_theorems(
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    engine=None,
) -> Tuple["Theorem", "Theorem"]:
    """The SECTION-LEVEL Jacobi instance for ``(X, Y, Z)`` — the
    canonical-form combination
    ``[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]]`` (engine-normalized) equals
    zero — as a citable :class:`Theorem` pair (both signs, since a
    residual may surface negated).

    Proof: the combination acts as zero on the generic function ``f``
    (:func:`prove_jacobi`, mechanical), and a vector field vanishing
    on all functions is zero (agreement on generators — the explicit
    synthetic step). The tangent analogue of the algebroid
    Jacobiator-instance theorems (Phase 3.F); torsionful Bianchi
    identities consume it (Phase 4.C).

    ``engine`` chooses the NORMALIZATION engine (default: the tangent
    engine); pass the engine the citation will run under, so the
    theorem's left-hand side matches that engine's normal form
    structurally. The proof leg is always tangent-pure.
    """
    from jacopy.algorithms.product_rule import product_rule
    from jacopy.algorithms.simplify import simplify
    from jacopy.proof.step import ProofStep
    from jacopy.proof.theorems import Theorem

    combination = Sum(
        lie_bracket(X, lie_bracket(Y, Z)),
        lie_bracket(Y, lie_bracket(Z, X)),
        lie_bracket(Z, lie_bracket(X, Y)),
    )
    # Normalize to the engine's canonical form (orientation sort) so
    # the cited rule matches residuals structurally.
    eng = engine if engine is not None else _engine(registry)
    current = combination
    for _ in range(16):
        nxt, step = eng.expand_once(current)
        if step is None:
            break
        current = nxt
    normalized = simplify(product_rule(current, registry), registry)

    sub = prove_jacobi(X, Y, Z, f, registry=registry)
    step = ProofStep(
        normalized,
        Integer(0),
        rule="agreement on generators (generic function)",
        justification=(
            "the Jacobi combination kills the generic function "
            f"{f._repr_inner()}, and a vector field vanishing on all "
            "functions is zero"
        ),
    )
    for s in sub:
        step.add_child(s)
    chain = ProofChain([step])
    names = (X._repr_inner(), Y._repr_inner(), Z._repr_inner())
    thm = Theorem(
        name="lie_jacobi_" + "_".join(names),
        statement=(
            f"[{names[0]},[{names[1]},{names[2]}]] + "
            f"[{names[1]},[{names[2]},{names[0]}]] + "
            f"[{names[2]},[{names[0]},{names[1]}]] = 0"
        ),
        lhs=normalized,
        rhs=Integer(0),
        proof=chain,
        generality="instance",
        from_axioms=("Lie bracket definition",),
        notes="section-level Jacobi via agreement on generators",
    )
    neg_chain = ProofChain(list(chain.steps))
    neg_chain.append(
        ProofStep(
            simplify(Neg(normalized), registry),
            Integer(0),
            rule="negate both sides",
            justification="the combination vanishes, so does its negation",
        )
    )
    thm_neg = Theorem(
        name=thm.name + "_neg",
        statement=f"−({thm.statement})",
        lhs=simplify(Neg(normalized), registry),
        rhs=Integer(0),
        proof=neg_chain,
        generality="instance",
        from_axioms=thm.from_axioms,
    )
    return thm, thm_neg


def prove_leibniz_second_slot(
    X: Expr,
    f: Expr,
    Y: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``[X, fY](g) = X(f)·Y(g) + f·[X, Y](g)`` (the Leibniz /
    ``C^∞(M)``-module property in the second slot, on a generic
    function ``g``).

    ``fY`` enters as the operator ``Product(f, Y)``; its action
    unfolds through the composition pass and
    :class:`ScalarActAsMultiplicationDefinition`.
    """
    lhs = Act(lie_bracket(X, Product(f, Y)), g)
    rhs = Sum(
        Product(Act(X, f), Act(Y, g)),
        Product(f, Act(lie_bracket(X, Y), g)),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_additivity_second_slot(
    X: Expr,
    Y: Expr,
    Z: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``[X, Y + Z](f) = [X, Y](f) + [X, Z](f)`` (ℝ-linearity,
    additive part; PDF item 9c)."""
    lhs = Act(lie_bracket(X, Sum(Y, Z)), f)
    rhs = Sum(
        Act(lie_bracket(X, Y), f),
        Act(lie_bracket(X, Z), f),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_scalar_homogeneity(
    c: int,
    X: Expr,
    Y: Expr,
    f: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``[c·X, Y](f) = c·[X, Y](f)`` for a numeric constant
    ``c`` (ℝ-linearity, homogeneous part; PDF item 9c).

    Constants pass through the bracket — unlike smooth functions,
    whose first-slot behaviour picks up the extra ``−Y(f)·X`` term
    (see :func:`prove_first_slot_function_linearity`).
    """
    if not isinstance(c, int):
        raise TypeError("prove_scalar_homogeneity expects an int constant")
    coeff = Integer(c)
    lhs = Act(lie_bracket(Product(coeff, X), Y), f)
    rhs = Product(coeff, Act(lie_bracket(X, Y), f))
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )


def prove_first_slot_function_linearity(
    f: Expr,
    X: Expr,
    Y: Expr,
    g: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``[fX, Y](g) = f·[X, Y](g) − Y(f)·X(g)`` (the mirrored
    first-slot property; equivalent to Leibniz + antisymmetry)."""
    lhs = Act(lie_bracket(Product(f, X), Y), g)
    rhs = Sum(
        Product(f, Act(lie_bracket(X, Y), g)),
        Neg(Product(Act(Y, f), Act(X, g))),
    )
    return ExpandAndSimplify().prove(
        lhs, rhs, registry=registry, engine=_engine(registry)
    )
