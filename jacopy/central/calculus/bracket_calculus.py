"""
The exterior derivative derived from a bracket — generic machinery
(PDF item 9d; definition policy §1 and §4).

**Canonical definition (Palais intrinsic formula, no combinatorial
prefactor).** Given an anchor ``ρ`` and a bracket ``[·,·]``, the
exterior derivative of a p-form ``ω`` evaluated on ``p+1`` arguments
is *defined* as

    dω(X_0, …, X_p) = Σ_i (−1)^i ρ(X_i)( ω(X_0, …, X̂_i, …, X_p) )
                    + Σ_{i<j} (−1)^{i+j} ω([X_i, X_j], X_0, …, X̂_i, …, X̂_j, …, X_p)

with the special cases ``df(X) = ρ(X)(f)`` for functions and
``dω(X, Y) = ρ(X)(ω(Y)) − ρ(Y)(ω(X)) − ω([X, Y])`` for 1-forms. This
matches the determinant wedge convention fixed in Phase 1
(``ω∧η(X,Y) = ω(X)η(Y) − ω(Y)η(X)``).

Everything else about ``d`` — ``d² = 0``, the wedge Leibniz rule, the
Cartan relations — is a **theorem** proved from this definition.

The machinery is *generic in the pair* ``(ρ, [·,·])``
(:class:`BracketCalculus`): the TM case takes the identity anchor and
the Lie bracket (:mod:`jacopy.central.tangent.exterior`), a general
algebroid takes ``(ρ_E, [·,·]_E)`` (Phase 3), the Poisson cotangent
case takes ``(π^♯, [·,·]_Kos)`` giving ``d̃`` (Phase 5). Per PDF item
8a, instantiating with ``(id, Lie)`` *is* the usual exterior
derivative — the TM case is an instantiation, not a separate code
path.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


class ExteriorDerivative(Derivation):
    """``d`` — the degree ``+1`` derivation of a bracket calculus.

    Structural identity is ``(name, degree, calculus name)``: the ``d``
    of the TM Cartan calculus and the ``d̃`` of a Poisson calculus are
    distinct operators even if displayed similarly.
    """

    __slots__ = ("_calculus_name",)

    def __init__(self, calculus_name: str, *, name: str = "d") -> None:
        super().__init__(name, degree=1)
        self._calculus_name = calculus_name

    @property
    def calculus_name(self) -> str:
        return self._calculus_name

    def _key(self) -> Any:
        return (self._name, self._degree, self._calculus_name)


class LieDerivative(Derivation):
    """``L_X`` — the degree-0 Lie derivative of a bracket calculus in
    the direction ``X`` (PDF item 9e).

    Canonical definition (definition policy §1): the core cases

        L_X f = ρ(X)(f),      L_X Y = [X, Y],

    plus the evaluation (Leibniz-extension) formula on covariant slots

        (L_X ω)(Y_1, …, Y_k) = ρ(X)(ω(Y_1, …, Y_k))
                              − Σ_i ω(Y_1, …, [X, Y_i], …, Y_k).

    The Cartan magic formula ``L_X = d ι_X + ι_X d`` is a THEOREM
    proved from this definition, never a definition itself.
    """

    __slots__ = ("_calculus_name", "_vector")

    def __init__(
        self,
        calculus_name: str,
        X: Expr,
        *,
        op_name: str = "L",
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(X, Expr):
            raise TypeError("LieDerivative requires an Expr direction")
        display = (
            name if name is not None else f"{op_name}_{X._repr_inner()}"
        )
        super().__init__(display, degree=0)
        self._calculus_name = calculus_name
        self._vector = X

    @property
    def calculus_name(self) -> str:
        return self._calculus_name

    @property
    def vector(self) -> Expr:
        return self._vector

    def _key(self) -> Any:
        return (self._name, self._degree, self._calculus_name, self._vector)


class BracketCalculus:
    """An ``(anchor, bracket)`` pair with its derived operators.

    Parameters
    ----------
    name
        Identity of the calculus (e.g. ``"Cartan-TM"``); the derived
        ``d`` carries it.
    anchor
        ``ρ``: maps a section to the vector field that acts on
        functions. The TM case passes the identity.
    bracket
        ``[·,·]``: maps two sections to their bracket (an Expr that can
        occupy an argument slot — e.g. the opaque ``LieBracketVF``).
    d_name
        Display name of the derived exterior derivative (``"d"``;
        Poisson will use ``"d̃"``).
    lie_name
        Display name of the derived Lie derivative (``"L"``; Poisson
        will use ``"L̃"``). Display only — identity is carried by the
        calculus name (audit 3 fix: the ``L_`` pattern was hardcoded,
        the same bug family as the fixed ``∇_`` connection display).
    """

    __slots__ = (
        "_name",
        "_anchor",
        "_bracket",
        "_d",
        "_lie_name",
        "_form_degree",
        "_section_test",
    )

    def __init__(
        self,
        name: str,
        *,
        anchor: Callable[[Expr], Expr],
        bracket: Callable[[Expr, Expr], Expr],
        d_name: str = "d",
        lie_name: str = "L",
        form_degree=None,
        section_test=None,
    ) -> None:
        """``form_degree``/``section_test`` are the calculus' grading
        hooks (Phase 5.D): the TILDE calculus on ``T*M`` swaps the
        roles — its sections are 1-FORMS and its forms are
        MULTIVECTORS — so the intrinsic d/L rules must ask the
        calculus, not assume the TM grading. ``form_degree(expr,
        registry) -> Degree`` resolves the calculus-form degree
        (default: the ordinary form grading); ``section_test(expr) ->
        bool`` recognizes the calculus' sections (default: the
        wedge-degree-1 derivation protocol)."""
        if not isinstance(name, str) or not name:
            raise ValueError("BracketCalculus name must be a non-empty str")
        if not callable(anchor) or not callable(bracket):
            raise TypeError("anchor and bracket must be callables")
        self._name = name
        self._anchor = anchor
        self._bracket = bracket
        self._d = ExteriorDerivative(name, name=d_name)
        self._lie_name = lie_name
        self._form_degree = form_degree
        self._section_test = section_test

    def form_degree(self, expr: Expr, registry) -> Degree:
        """The calculus-form degree of ``expr`` (raises
        :class:`ValueError` when undetermined)."""
        if self._form_degree is not None:
            return self._form_degree(expr, registry)
        return degree_of(expr, registry)

    def is_section(self, expr: Expr) -> bool:
        """Is ``expr`` a section of this calculus' bundle?"""
        if self._section_test is not None:
            return self._section_test(expr)
        from jacopy.algebra.lie_bracket_vf import LieBracketVF
        from jacopy.central.objects.vector_field import VectorField

        if isinstance(expr, (VectorField, LieBracketVF)):
            return True
        return (
            isinstance(expr, Derivation)
            and getattr(expr, "wedge_degree", None) == Degree.const(1)
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def d(self) -> ExteriorDerivative:
        """The derived exterior derivative of this calculus."""
        return self._d

    def anchor(self, X: Expr) -> Expr:
        """``ρ(X)`` — the acting vector field of the section ``X``."""
        return self._anchor(X)

    def bracket(self, X: Expr, Y: Expr) -> Expr:
        """``[X, Y]`` — the bracket of this calculus."""
        return self._bracket(X, Y)

    def lie(self, X: Expr) -> LieDerivative:
        """``L_X`` — the derived Lie derivative in the direction ``X``."""
        return LieDerivative(self._name, X, op_name=self._lie_name)

    def __repr__(self) -> str:
        return f"BracketCalculus({self._name!r})"


def _slot_ok(
    calc: BracketCalculus, slot: Expr, registry
) -> bool:
    """Evaluation slots must hold SECTIONS of the calculus' bundle.

    A slot that is determinably a *form* of this calculus (concrete
    calculus-degree ≥ 1) is ill-typed — the intrinsic rules must stay
    inert instead of unrolling it into nonsense like ``[X, γ]_VF``
    with a 1-form ``γ`` in a vector-field bracket (type hole caught by
    the 5.D.2 mixed-world audit). Undeterminable degrees stay
    caller-asserted (arity policy)."""
    try:
        return calc.form_degree(slot, registry).as_int() < 1
    except ValueError:
        return True


def _evaluate(omega: Expr, args: Tuple[Expr, ...]) -> Expr:
    """``ω(args)`` in node form: 0 args → ``ω`` itself, 1 arg → the
    canonical pairing, ≥2 args → an alternating MultiEval."""
    if not args:
        return omega
    if len(args) == 1:
        return Pairing(omega, args[0])
    return MultiEval(omega, *args, alternating=True, slot_kind="vector")


class IntrinsicDDefinition(Definition):
    """The Palais formula as the definitional expansion of ``dω(…)``.

    Fires on two shapes rooted at this calculus' ``d``:

    * ``⟨dω, X⟩`` (a :class:`Pairing` whose form slot is ``Act(d, ω)``)
      — the arity-1 evaluation. Requires ``ω`` to be determinably a
      0-form; rewrites to ``ρ(X)(ω)``.
    * ``(dω)(X_0, …, X_k)`` (a :class:`MultiEval` with head
      ``Act(d, ω)``) — the general unroll for ``k+1`` slots. When
      ``|ω|`` is a determinable integer it must equal ``k`` (otherwise
      the node is left inert for diagnostics); a symbolic ``|ω|`` is
      unrolled as declared by the caller (same arity policy as
      ``Form.__call__``).

    Nested occurrences (``d(dω)`` inner evaluations) unfold recursively
    under the engine fix-point.
    """

    anchor = (Pairing, MultiEval)

    def __init__(
        self,
        calculus: BracketCalculus,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(calculus, BracketCalculus):
            raise TypeError("IntrinsicDDefinition expects a BracketCalculus")
        self._calc = calculus
        self._registry = registry
        self.name = (
            f"intrinsic d ({calculus.name}): Palais formula "
            "dω(X_0,…,X_p) = Σ(−1)^i ρ(X_i)(ω(…)) + Σ(−1)^{i+j} ω([X_i,X_j],…)"
        )

    # ---- helpers ----------------------------------------------------- #

    def _is_this_d(self, head: Expr) -> bool:
        return (
            isinstance(head, Act)
            and isinstance(head.op, ExteriorDerivative)
            and head.op.calculus_name == self._calc.name
        )

    def _degree_of_arg(self, omega: Expr) -> Optional[int]:
        try:
            return self._calc.form_degree(omega, self._registry).as_int()
        except ValueError:
            return None

    # ---- Definition API ---------------------------------------------- #

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Pairing):
            if not self._is_this_d(expr.alpha):
                return False
            if not _slot_ok(self._calc, expr.X, self._registry):
                return False
            return self._degree_of_arg(expr.alpha.arg) == 0
        if isinstance(expr, MultiEval):
            if not self._is_this_d(expr.head):
                return False
            if not all(
                _slot_ok(self._calc, a, self._registry)
                for a in expr.args
            ):
                return False
            p = self._degree_of_arg(expr.head.arg)
            return p is None or p == expr.arity - 1
        return False

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            omega = expr.alpha.arg
            return Act(self._calc.anchor(expr.X), omega)
        omega = expr.head.arg
        args = expr.args
        terms = []
        for i, Xi in enumerate(args):
            rest = args[:i] + args[i + 1:]
            inner = _evaluate(omega, rest)
            term: Expr = Act(self._calc.anchor(Xi), inner)
            terms.append(Neg(term) if i % 2 else term)
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                br = self._calc.bracket(args[i], args[j])
                rest = tuple(
                    a for idx, a in enumerate(args) if idx not in (i, j)
                )
                inner = _evaluate(omega, (br,) + rest)
                terms.append(Neg(inner) if (i + j) % 2 else inner)
        return Sum(*terms)


class IntrinsicLDefinition(Definition):
    """The canonical definition of ``L_X`` as engine rules.

    Three shapes rooted at this calculus' :class:`LieDerivative`:

    * ``L_X f → ρ(X)(f)`` for a certain scalar function (``Act`` with a
      :class:`LieDerivative` operator);
    * ``L_X Y → [X, Y]`` for a section in the argument slot;
    * evaluation on covariant slots: ``⟨L_X ω, Y⟩`` and
      ``(L_X ω)(Y_1, …, Y_k)`` unroll to
      ``ρ(X)(ω(…)) − Σ_i ω(…, [X, Y_i], …)``, with the same arity
      policy as the intrinsic ``d`` (concrete degree must match the
      slot count; symbolic degree is caller-asserted).
    """

    anchor = (Act, Pairing, MultiEval)

    def __init__(
        self,
        calculus: BracketCalculus,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(calculus, BracketCalculus):
            raise TypeError("IntrinsicLDefinition expects a BracketCalculus")
        self._calc = calculus
        self._registry = registry
        self.name = (
            f"intrinsic L ({calculus.name}): L_X f = ρ(X)(f), "
            "L_X Y = [X,Y], (L_X ω)(Y…) = ρ(X)(ω(Y…)) − Σ ω(…,[X,Y_i],…)"
        )

    # ---- helpers ----------------------------------------------------- #

    def _is_this_L(self, head: Expr) -> bool:
        return (
            isinstance(head, Act)
            and isinstance(head.op, LieDerivative)
            and head.op.calculus_name == self._calc.name
        )

    def _is_section(self, expr: Expr) -> bool:
        # Delegated to the calculus (Phase 5.D): the TM default is the
        # wedge-degree-1 derivation protocol; the tilde calculus
        # recognizes 1-FORMS as its sections.
        return self._calc.is_section(expr)

    def _degree_int(self, omega: Expr) -> Optional[int]:
        try:
            return self._calc.form_degree(omega, self._registry).as_int()
        except ValueError:
            return None

    # ---- Definition API ---------------------------------------------- #

    def matches(self, expr: Expr) -> bool:
        from jacopy.central.calculus.scalars import is_scalar_function

        if isinstance(expr, Act) and isinstance(expr.op, LieDerivative):
            if expr.op.calculus_name != self._calc.name:
                return False
            return is_scalar_function(expr.arg, self._registry) or (
                self._is_section(expr.arg)
            )
        if isinstance(expr, Pairing):
            if not self._is_this_L(expr.alpha):
                return False
            if not _slot_ok(self._calc, expr.X, self._registry):
                return False
            return self._degree_int(expr.alpha.arg) == 1
        if isinstance(expr, MultiEval):
            if not self._is_this_L(expr.head):
                return False
            if not all(
                _slot_ok(self._calc, a, self._registry)
                for a in expr.args
            ):
                return False
            p = self._degree_int(expr.head.arg)
            return p is None or p == expr.arity
        return False

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.calculus.scalars import is_scalar_function

        if isinstance(expr, Act) and isinstance(expr.op, LieDerivative):
            X = expr.op.vector
            if is_scalar_function(expr.arg, self._registry):
                return Act(self._calc.anchor(X), expr.arg)
            return self._calc.bracket(X, expr.arg)
        if isinstance(expr, Pairing):
            X = expr.alpha.op.vector
            omega = expr.alpha.arg
            Y = expr.X
            return Sum(
                Act(self._calc.anchor(X), Pairing(omega, Y)),
                Neg(Pairing(omega, self._calc.bracket(X, Y))),
            )
        X = expr.head.op.vector
        omega = expr.head.arg
        args = expr.args
        terms = [Act(self._calc.anchor(X), _evaluate(omega, args))]
        for i in range(len(args)):
            replaced = (
                args[:i] + (self._calc.bracket(X, args[i]),) + args[i + 1:]
            )
            terms.append(Neg(_evaluate(omega, replaced)))
        return Sum(*terms)


class HeadSumDefinition(Definition):
    """Linearity in the *head* slot: ``⟨α + β, Y⟩ → ⟨α, Y⟩ + ⟨β, Y⟩``
    and ``(ω + η)(Y…) → ω(Y…) + η(Y…)``.

    Only the head/form slot is split. The argument slots are
    deliberately NOT split here: the collecting pass
    (:func:`~jacopy.algorithms.factor_terms.collect_pairings`) groups
    argument slots, and an engine-side split would ping-pong against
    it. Head sums are never produced by the collector, so this
    direction is safe.
    """

    name = "slot linearity: sums in the head slot distribute"
    anchor = (Pairing, MultiEval)

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Pairing):
            return isinstance(expr.alpha, Sum)
        if isinstance(expr, MultiEval):
            return isinstance(expr.head, Sum)
        return False

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            return Sum(
                *(Pairing(term, expr.X) for term in expr.alpha.children)
            )
        return Sum(
            *(
                MultiEval(
                    term,
                    *expr.args,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
                for term in expr.head.children
            )
        )


class HeadScalarDefinition(Definition):
    """``C^∞``-linearity in the head slot: ``⟨f·α, Y⟩ → f·⟨α, Y⟩``
    (and the :class:`MultiEval` analogue).

    Definitional — evaluation of a form is ``C^∞``-linear in the form.
    Certainly-scalar factors are pulled out of a Product head wherever
    they sit (canonical sorting may have reordered them); the rule
    only fires when a non-scalar factor remains to stay the head.
    Introduced in Phase 3.E.1 (the coboundary-Leibniz derivation pairs
    ``f·Dg`` heads against generic sections); generic, not
    algebroid-specific.
    """

    name = "slot linearity: scalar factors pull out of the head slot"
    anchor = (Pairing, MultiEval)

    def __init__(self, registry=None) -> None:
        self._registry = registry

    def _split(self, head: Expr):
        from jacopy.central.calculus.scalars import is_scalar_function

        if not (isinstance(head, Product) and len(head.children) >= 2):
            return None
        scalars = []
        rest = []
        for c in head.children:
            (scalars if is_scalar_function(c, self._registry) else rest).append(c)
        if not scalars or not rest:
            return None
        return scalars, (rest[0] if len(rest) == 1 else Product(*rest))

    def matches(self, expr: Expr) -> bool:
        head = expr.alpha if isinstance(expr, Pairing) else expr.head
        return self._split(head) is not None

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            scalars, core = self._split(expr.alpha)
            return Product(*scalars, Pairing(core, expr.X))
        scalars, core = self._split(expr.head)
        return Product(
            *scalars,
            MultiEval(
                core,
                *expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            ),
        )


class SlotNegDefinition(Definition):
    """Multilinearity: a ``Neg`` in an evaluation slot pulls out.

    ``⟨α, −V⟩ → −⟨α, V⟩``, ``⟨−α, X⟩ → −⟨α, X⟩`` and the analogous
    single-Neg pull-out for :class:`MultiEval` slots/heads. Definitional
    (linearity of the evaluation primitives); keeps canonical-form
    rewrites (e.g. bracket orientation) composable with slot rules.
    """

    name = "slot linearity: Neg pulls out of evaluation slots"
    anchor = (Pairing, MultiEval)

    def matches(self, expr: Expr) -> bool:
        if isinstance(expr, Pairing):
            return isinstance(expr.alpha, Neg) or isinstance(expr.X, Neg)
        if isinstance(expr, MultiEval):
            return isinstance(expr.head, Neg) or any(
                isinstance(a, Neg) for a in expr.args
            )
        return False

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr, Pairing):
            sign = 0
            alpha, X = expr.alpha, expr.X
            while isinstance(alpha, Neg):
                sign ^= 1
                alpha = alpha.arg
            while isinstance(X, Neg):
                sign ^= 1
                X = X.arg
            node: Expr = Pairing(alpha, X)
            return Neg(node) if sign else node
        sign = 0
        head = expr.head
        while isinstance(head, Neg):
            sign ^= 1
            head = head.arg
        args = []
        for a in expr.args:
            while isinstance(a, Neg):
                sign ^= 1
                a = a.arg
            args.append(a)
        node = MultiEval(
            head, *args,
            alternating=expr.alternating, slot_kind=expr.slot_kind,
        )
        return Neg(node) if sign else node


class SlotZeroDefinition(Definition):
    """Multilinearity in the zero slot: ``⟨α, 0⟩ = 0``, ``⟨0, X⟩ = 0``,
    and a :class:`MultiEval` with any zero slot (or zero head) is 0.

    Definitional (linearity of the pairing / evaluation primitives);
    what lets a cited bracket identity ``V → 0`` finish the job inside
    an evaluation slot.
    """

    name = "slot linearity: evaluation with a zero slot vanishes"
    anchor = (Pairing, MultiEval)

    def matches(self, expr: Expr) -> bool:
        from jacopy.core.expr import Integer as _Int

        zero = _Int(0)
        if isinstance(expr, Pairing):
            return expr.alpha == zero or expr.X == zero
        if isinstance(expr, MultiEval):
            return expr.head == zero or any(a == zero for a in expr.args)
        return False

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Integer as _Int

        return _Int(0)


class MultiEvalArgLinearityDefinition(Definition):
    """Multilinearity of :class:`MultiEval` ARGUMENT slots
    (definitional — the evaluation is multilinear):

        ω(…, f·X, …) → f·ω(…, X, …),
        ω(…, X + Y, …) → ω(…, X, …) + ω(…, Y, …).

    Phase 5.E.3: the Nambu sharp produces ``f·Πη`` and sum-shaped
    vectors inside evaluation slots. The PAIRING argument slot is
    deliberately NOT covered (the known collect_pairings ping-pong);
    no rule collects MultiEval arguments, so this direction is
    cycle-free."""

    name = (
        "slot linearity: scalars and sums split out of MultiEval "
        "argument slots"
    )
    anchor = MultiEval

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        # Scalars in ANY factor position pull out (2026-08-01: the
        # Nambu Jacobi residual carries ``d(Y g)·Π(…)`` slots with
        # the scalar SECOND).
        from jacopy.core.expr import Product
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )

        if isinstance(expr, Product) and len(expr.children) >= 2:
            scalars = [
                c
                for c in expr.children
                if is_scalar_function(c, self._registry)
            ]
            rest = [
                c
                for c in expr.children
                if not is_scalar_function(c, self._registry)
            ]
            if scalars and len(rest) == 1:
                return (
                    scalars[0]
                    if len(scalars) == 1
                    else Product(*scalars)
                ), rest[0]
        return None

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        return any(
            isinstance(a, Sum) or self._scalar_split(a) is not None
            for a in expr.args
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Product

        args = list(expr.args)
        for i, a in enumerate(args):
            if isinstance(a, Sum):
                return Sum(
                    *(
                        expr.with_args(
                            *(args[:i] + [t] + args[i + 1 :])
                        )
                        for t in a.children
                    )
                )
            split = self._scalar_split(a)
            if split is not None:
                scalar, rest = split
                return Product(
                    scalar,
                    expr.with_args(
                        *(args[:i] + [rest] + args[i + 1 :])
                    ),
                )
        raise AssertionError("rewrite without splittable arg")


class PairingVectorScalarDefinition(Definition):
    """``⟨α, f·Y⟩ → f·⟨α, Y⟩`` — certainly-scalar factors pull out of
    the pairing's VECTOR slot (definitional C∞-bilinearity).

    Cycle analysis (2026-08-01, admitted after the user-bracket API
    prototype surfaced the gap): ``collect_pairings`` collects only
    MULTI-term groups (``Σ sᵢ⟨α,Vᵢ⟩ → ⟨α, Σ sᵢVᵢ⟩``) under a
    strictly-smaller acceptance, so a split singleton is never
    re-collected; a collected slot is a ``Sum``, which this rule does
    not touch. The historical guard on
    ``SharpPairingLinearityDefinition`` concerned the SUM-slot split,
    which stays excluded."""

    name = "pairing slot linearity: ⟨α, f·Y⟩ = f·⟨α, Y⟩"
    anchor = Pairing

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def _split(self, expr: Expr):
        from jacopy.core.expr import Product
        from jacopy.central.calculus.scalars import (
            is_scalar_function,
        )

        v = expr.X
        if not (
            isinstance(v, Product) and len(v.children) >= 2
        ):
            return None
        scalars = [
            c
            for c in v.children
            if is_scalar_function(c, self._registry)
        ]
        rest = [
            c
            for c in v.children
            if not is_scalar_function(c, self._registry)
        ]
        if scalars and len(rest) == 1:
            return scalars, rest[0]
        return None

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Pairing)
            and self._split(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Product

        scalars, core = self._split(expr)
        return Product(*scalars, Pairing(expr.alpha, core))


class ActExpansionDefinition(Definition):
    """Slot-reachable graded Leibniz / linearity of a single ``Act``
    node — delegates to the product-rule algorithm's ``_expand_act``
    (single source of truth; Phase 6.D).

    The engine's slot protocol fires Definitions INSIDE operator
    atoms (a Nambu sharp's form slot, a bracket's sections), but the
    algorithm passes (``product_rule``, ``simplify``) only walk
    ``children`` — so shapes like ``Π(d(ι_U(f·η)))`` stalled with the
    Leibniz expansion out of reach. This rule makes the same
    expansion available as an engine rewrite. Definitional: it IS
    the product rule (ℝ-linearity + graded Leibniz), step-tagged as
    one axiom application."""

    name = "graded Leibniz / Act linearity (slot-reachable product rule)"
    anchor = Act

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def _expanded(self, expr: Expr) -> Expr:
        from jacopy.algorithms.product_rule import _expand_act

        return _expand_act(expr.op, expr.arg, self._registry)

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Act) and self._expanded(expr) != expr

    def rewrite(self, expr: Expr) -> Expr:
        return self._expanded(expr)


class HeadFormProductLiftDefinition(Definition):
    """``(α · β)(Y…) → (α ∧ β)(Y…)`` — a bare :class:`Product` of
    form-degree factors in an evaluation head IS their wedge (v3's
    canonical graded product of forms is :class:`Wedge`; raw products
    of forms only arise transiently from Leibniz splits like
    ``d(f·x) = d(f)·x + f·d(x)`` once the scalar prefix is gone).

    Guarded: every factor must have determinable degree ≥ 1 and the
    degrees must sum to the evaluation arity — scalar prefixes are
    left to :class:`HeadScalarDefinition` (which strips them first).
    Phase 6.D: the slot-reachable product rule surfaced these shapes
    ahead of the Palais evaluation."""

    name = "head lift: a product of forms in an evaluation head is their wedge"
    anchor = MultiEval

    def __init__(
        self, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def _form_factors(self, expr: "MultiEval"):
        from jacopy.core.expr import Product as _Product

        head = expr.head
        if not (
            isinstance(head, _Product) and len(head.children) >= 2
        ):
            return None
        total = 0
        for c in head.children:
            try:
                k = degree_of(c, self._registry).as_int()
            except ValueError:
                return None
            if k is None or k < 1:
                return None
            total += k
        if total != expr.arity:
            return None
        return head.children

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and self._form_factors(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.wedge import Wedge as _Wedge

        factors = self._form_factors(expr)
        return MultiEval(
            _Wedge(*factors),
            *expr.args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )
