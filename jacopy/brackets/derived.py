"""
Derived bracket construction, the mathematical heart of the package.

Given a graded bracket ``[·, ·]`` on a graded module and a generator
``Q`` of odd (or appropriate) degree, the *derived bracket* is

    {a, b}_Q := [[a, Q], b]

which is automatically a graded Leibniz bracket regardless of any
conditions on ``Q``. Over a graded LIE base with an ODD (shifted)
generator, the condition

    [Q, Q] = 0

makes the derived bracket a LODAY (left-Leibniz-Jacobi) bracket
[Kosmann-Schwarzbach 1996]. It is NOT automatically graded
antisymmetric, and the cyclic Lie-form Jacobi holds only where the
bracket is additionally skew (e.g. on suitable abelian subalgebras)
— the proof layer's :class:`DerivedBracketStrategy` checks exactly
these preconditions before certifying anything (2026-09-08 audit).
This conditionality is the point of the construction: Poisson,
Courant, and Koszul brackets arise as derived brackets of a
Schouten-type base, each Jacobi reducing to one equation on ``Q``
*on the slots where the extra hypotheses hold*.

The :class:`DerivedBracket` class captures this construction at the
algorithmic level:

* :meth:`expand` produces ``[[a, Q], b]`` using the base bracket's own
  :meth:`~jacopy.brackets.base.GradedBracket.expand` on each layer.
* :meth:`jacobi_obstruction` returns ``[Q, Q]_base``, the Expr whose
  vanishing is the remaining hypothesis for the derived bracket's
  Jacobi under the theorem's preconditions.
* :func:`derived_bracket` is a light factory helper.

The *Derived Bracket Theorem* itself — Loday Jacobi from a
vanishing obstruction under the stated preconditions — is registered
as a package-level theorem in
Faz 9 and consulted from the proof layer. This module only provides
the object; the theorem lives elsewhere so that each new
``DerivedBracket`` instance automatically inherits the result rather
than re-proving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree


LieDerivativeFactory = Callable[[Expr], Derivation]


@dataclass(frozen=True)
class VanishingCondition:
    """A proof-level claim of the form ``obstruction = 0``.

    A :class:`VanishingCondition` is the typed handle the proof layer
    consumes when a construction's correctness rests on a single
    equation. :meth:`DerivedBracket.jacobi_condition` returns one whose
    :attr:`obstruction` is ``[Q, Q]_base``: under the Derived Bracket
    Theorem's preconditions (graded-Lie base, odd shifted generator —
    plus per-slot skewness for the cyclic Lie form) its vanishing is
    the remaining hypothesis for Jacobi on ``{·, ·}_Q``, so passing
    the condition around is passing that hypothesis.

    The class is intentionally minimal, it is data, not a proof
    tactic. :meth:`holds` runs :func:`simplify` against the
    ``obstruction`` and reports whether the canonical form is the
    literal :class:`Integer` ``0``. Symbolic residues that the
    canonical pipeline can't settle return ``False`` even if the
    underlying math would eventually close them; the expectation is
    that the proof layer (e.g.
    :class:`jacopy.proof.strategies.DerivedBracketStrategy`) is the
    right tool for non-trivial discharges.
    """

    obstruction: Expr
    name: str = "vanishing condition"

    def holds(self, registry: Optional[PropertyRegistry] = None) -> bool:
        """True iff the obstruction simplifies to :class:`Integer` ``0``."""
        return simplify(self.obstruction, registry) == Integer(0)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name}: {self.obstruction._repr_inner()} = 0"


class DerivedBracket(GradedBracket):
    """``{a, b}_Q := [[a, Q]_base, b]_base``.

    Parameters
    ----------
    base
        The underlying graded bracket.
    Q
        The generator, an :class:`Expr` of degree ``degree_Q``.
    degree_Q
        Explicit generator degree. Defaults to ``0`` when unspecified.
        The derived bracket's own degree is ``degree_Q + 2·|base|``
        per the derived-bracket degree formula (each base layer
        contributes its degree).
    name
        Optional display name. Defaults to a structured tag derived
        from the base bracket and generator.
    acting_on
        Optional :class:`~jacopy.algebra.derivation.Derivation` that
        lifts operands into the domain of the base bracket. The
        canonical use is the Koszul bracket on 1-forms: with
        ``base=sn``, ``Q=π``, and ``acting_on=π^♯`` (an
        :class:`~jacopy.calculus.anchor.Anchor` or musical
        :class:`~jacopy.calculus.musical.Sharp`), :meth:`expand`
        emits the classical Koszul formula
        ``L_{ρa} b − L_{ρb} a − d⟨ρa, b⟩`` rather than the literal
        ``[[a, Q]_base, b]_base``. The literal form is unreachable
        when operands are 1-forms and the base bracket (SN) takes
        multivectors, the anchor bridges the two bundles.
    d, lie_derivative
        Cartan operators used *only* when ``acting_on`` is set. Default
        to the smooth-manifold singletons.
    """

    def __init__(
        self,
        base: GradedBracket,
        Q: Expr,
        *,
        degree_Q: DegreeLike = 0,
        name: Optional[str] = None,
        acting_on: Optional[Derivation] = None,
        d: Optional[Derivation] = None,
        lie_derivative: Optional[LieDerivativeFactory] = None,
    ) -> None:
        if not isinstance(base, GradedBracket):
            raise TypeError("DerivedBracket 'base' must be a GradedBracket")
        if not isinstance(Q, Expr):
            raise TypeError("DerivedBracket generator 'Q' must be an Expr")
        if acting_on is not None and not isinstance(acting_on, Derivation):
            raise TypeError(
                f"DerivedBracket acting_on must be a Derivation, "
                f"got {type(acting_on).__name__}"
            )
        self._base = base
        self._Q = Q
        self._degree_Q = as_degree(degree_Q)
        self._acting_on = acting_on
        # Cartan ops resolved lazily to avoid import-time cycles, only
        # used when acting_on is set.
        self._d_override = d
        self._lie_derivative_override = lie_derivative
        display = name or f"{{·,·}}_{Q._repr_inner()}"
        # Derived-bracket degree formula: |[[a,Q],b]| = |a| + |b| +
        # |Q| + 2·|base| — each base-bracket layer contributes its own
        # degree (2026-09-07 audit, finding 2: the previous
        # ``|Q| − 2`` hard-coded the Schouten-type base |base| = −1).
        # Graded Leibniz always holds; antisymmetry is NOT automatic
        # (a derived bracket is Loday, skew only on suitable slots —
        # 2026-09-08 re-audit, finding 2) and Jacobi is conditional
        # on [Q, Q]_base = 0 + the theorem's preconditions, checked
        # by the proof layer. Both surfaced as non-claims here.
        super().__init__(
            display,
            degree=self._degree_Q + base.degree + base.degree,
            is_graded_antisymmetric=False,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=None,
        )

    @property
    def base(self) -> GradedBracket:
        return self._base

    @property
    def Q(self) -> Expr:
        return self._Q

    @property
    def degree_Q(self) -> Degree:
        return self._degree_Q

    @property
    def acting_on(self) -> Optional[Derivation]:
        return self._acting_on

    # ---- core expansion -------------------------------------------- #

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``{a, b}_Q = [[a, Q]_base, b]_base`` (generic) or the
        anchor-lifted Koszul formula when ``acting_on`` is set.

        With ``acting_on=ρ``:
        ``{a, b}_{Q, ρ} = L_{ρa} b − L_{ρb} a − d⟨ρa, b⟩``,
        matching :class:`~jacopy.brackets.koszul.KoszulBracket(ρ)`,
        the structural equality is the content of the
        classical/derived equivalence theorem on Poisson manifolds.
        """
        if self._acting_on is not None:
            return self._koszul_expand(a, b, registry)
        inner = self._base.expand(a, self._Q, registry)
        return self._base.expand(inner, b, registry)

    def _koszul_expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """Emit the Koszul 3-term formula via the anchor lift.

        Delegated out of :meth:`expand` to keep the default derived
        path clean. Cartan operators are resolved from the
        constructor overrides or from the smooth-manifold singletons;
        imports are deferred to avoid a top-level cycle between
        ``brackets`` and ``calculus``.
        """
        # v3 Cartan layer (the pre-reorg ``jacopy.calculus.*`` paths
        # died in the v3 reorg; 2026-09-07 audit, finding 7).
        from jacopy.central.tangent.exterior import CARTAN_TM, d as d_fn
        from jacopy.core.pairing import Pairing

        lie_factory = (
            self._lie_derivative_override
            if self._lie_derivative_override is not None
            else CARTAN_TM.lie
        )
        rho_a = Act(self._acting_on, a)
        rho_b = Act(self._acting_on, b)
        paired = Pairing(b, rho_a)
        d_term = (
            Act(self._d_override, paired)
            if self._d_override is not None
            else d_fn(paired)
        )
        return Sum(
            Act(lie_factory(rho_a), b),
            Neg(Act(lie_factory(rho_b), a)),
            Neg(d_term),
        )

    def expand_definition(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """Return ``[[a, Q]_base, b]_base`` *before* base-expansion.

        Useful for proofs that want to show the derived definition
        literally, without collapsing the inner base-bracket nodes. The
        result is a :class:`BracketApply` of the base bracket applied
        to a base-bracket node.
        """
        inner = BracketApply(self._base, a, self._Q)
        return BracketApply(self._base, inner, b)

    # ---- Jacobi obstruction ---------------------------------------- #

    def jacobi_obstruction(
        self, registry: Optional[PropertyRegistry] = None
    ) -> Expr:
        """Return ``[Q, Q]_base``, the expression whose vanishing is
        the remaining hypothesis for the derived bracket's Jacobi.

        Under the Derived Bracket Theorem's PRECONDITIONS (graded-Lie
        base, odd shifted generator — and per-slot skewness for the
        cyclic Lie form, checked by the proof layer) the vanishing of
        this single expression yields Jacobi; without them it claims
        nothing (2026-09-08 audits).
        """
        return self._base.expand(self._Q, self._Q, registry)

    def jacobi_obstruction_raw(self) -> BracketApply:
        """Return the unexpanded ``[Q, Q]_base`` :class:`BracketApply`
        node, leaving the base-bracket shape intact for display or
        further pattern matching."""
        return BracketApply(self._base, self._Q, self._Q)

    def jacobi_condition(
        self, registry: Optional[PropertyRegistry] = None
    ) -> VanishingCondition:
        """Return the :class:`VanishingCondition` controlling Jacobi.

        The returned condition wraps ``[Q, Q]_base`` (expanded by the
        base bracket via :meth:`jacobi_obstruction`) together with a
        display name tied to this derived bracket. The proof layer's
        :class:`~jacopy.proof.strategies.DerivedBracketStrategy`
        consumes exactly this shape when it discharges Jacobi: one
        theorem step reduces triple-cyclic Jacobi to the
        condition's vanishing, then simplifies.
        """
        return VanishingCondition(
            obstruction=self.jacobi_obstruction(registry),
            name=f"Jacobi condition on {self.name}",
        )

    # ---- identity -------------------------------------------------- #

    def _identity_key(self) -> Any:
        # Extend the base key with the base-bracket reference, the
        # generator, and the generator's degree so that two derived
        # brackets with identical parameters compare equal and hash
        # alike. ``acting_on`` + Cartan overrides also participate,
        # two derived brackets with different anchors are distinct.
        return super()._identity_key() + (
            self._base,
            self._Q,
            self._degree_Q,
            self._acting_on,
            self._d_override,
            self._lie_derivative_override,
        )


# --------------------------------------------------------------------- #
# Factory                                                                #
# --------------------------------------------------------------------- #


def derived_bracket(
    base: GradedBracket,
    Q: Expr,
    *,
    degree_Q: DegreeLike = 0,
    name: Optional[str] = None,
) -> DerivedBracket:
    """Construct a :class:`DerivedBracket`, mirror of the class
    constructor with a friendlier functional name."""
    return DerivedBracket(base, Q, degree_Q=degree_Q, name=name)
