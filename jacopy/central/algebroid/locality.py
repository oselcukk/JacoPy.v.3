"""
The locality projector (Phase 3.E.4) [MC Def 3.10].

On a REGULAR anchored bundle (``ρ`` of locally constant rank, so
``ker(ρ)`` is a subbundle [MC Def 3.9]), a **locality projector** is a
``C^∞``-linear map ``𝒫: 𝔛(E) → 𝔛(E)`` such that ``L̂ := 𝒫 ∘ L`` stays
in the locality structure ``[L̃]`` and ``im(L̂) ⊂ ker(ρ)``. It is the
ingredient that upgrades the pseudo-E-curvature to a genuine
``C^∞``-multilinear E-curvature operator [MC Cor 3.3] — that operator
itself is Phase 4 material; here we mechanize the projector's two
defining properties:

* **Absorption** (``L̂ ∈ [L̃]``): on coboundary forms the projected
  operator agrees with the original — ``𝒫(L(Df, u, v)) → L(Df, u, v)``.
* **Kernel** (``im(L̂) ⊂ ker(ρ)``): ``ρ(𝒫(L(ω, u, v))) → 0`` for an
  arbitrary form slot ``ω``.

For ``ω = Df`` the two compose to ``ρ(L(Df,u,v)) = 0`` — which is NOT
an extra assumption: on a local pre-Leibniz algebroid it is a theorem
[MC §3], mechanized as
:func:`~jacopy.central.algebroid.theorems.prove_locality_anchor_annihilation`
and enterable by citation. The projector context therefore requires
the ``regular`` declaration (its literal precondition) and nothing
else; consistency of the ``Df`` case is the cited theorem's job.

The projector reuses the :class:`SectionMap`/:class:`MappedSection`
machinery, so ``C^∞``-linearity of ``𝒫`` is already definitional.
"""

from __future__ import annotations

from jacopy.core.expr import Expr, Integer
from jacopy.proof.expansion import Definition
from jacopy.central.algebroid.context import (
    Algebroid,
    AnchoredVF,
    CoboundaryForm,
    LocalityOperator,
)
from jacopy.central.algebroid.operators import MappedSection, SectionMap


class LocalityProjector(SectionMap):
    """``𝒫`` — a locality projector on a regular algebroid
    [MC Def 3.10]; a ``C^∞``-linear ``E → E`` map context whose
    defining rules are supplied to the engine via
    ``algebroid_engine(alg, projectors=(P,))``.

    DEFINITION VARIANTS (literature audit 2026-07-31): the
    metric-connection paper [MC Def 3.10] requires only absorption +
    kernel; the admissible-connections paper (Dereli-Doğan, JGP 186)
    Def 3.8 additionally requires ``𝒫|_{ker ρ} = id``. The only
    ker-ρ membership the engine can certify structurally is
    ``im(𝒫L)`` itself, so the mechanizable content of the extra
    condition is IDEMPOTENCE on projected locality values,
    ``𝒫(𝒫(L(ω,u,v))) → 𝒫(L(ω,u,v))``. Pass ``admissible=True`` to
    adopt the stronger definition; the default stays [MC Def 3.10]
    (the extra rule is NOT derivable from it, so defaulting it on
    would smuggle an axiom)."""

    __slots__ = ("_algebroid", "_admissible")

    def __init__(
        self,
        alg: Algebroid,
        name: str = "P",
        *,
        admissible: bool = False,
    ) -> None:
        if not isinstance(alg, Algebroid):
            raise TypeError("LocalityProjector expects an Algebroid")
        if alg.is_tangent:
            raise ValueError(
                "the tangent algebroid has no locality operator, hence "
                "no locality projector"
            )
        if not alg.declares("regular"):
            raise ValueError(
                "a locality projector needs ker(ρ) to be a subbundle: "
                "declare 'regular' (anchor of locally constant rank, "
                "MC Def 3.9) on the algebroid first"
            )
        super().__init__(name, alg.bundle, alg.bundle)
        self._algebroid = alg
        self._admissible = bool(admissible)

    @property
    def algebroid(self) -> Algebroid:
        return self._algebroid

    @property
    def admissible(self) -> bool:
        return self._admissible

    def rules(self):
        """The projector's defining engine rules (two for
        [MC Def 3.10]; plus idempotence under the admissible-paper
        definition)."""
        base = (
            LocalityProjectorAbsorptionDefinition(self),
            LocalityProjectorKernelDefinition(self),
        )
        if self._admissible:
            return base + (
                LocalityProjectorIdempotenceDefinition(self),
            )
        return base


def locality_projector(
    alg: Algebroid, name: str = "P", *, admissible: bool = False
) -> LocalityProjector:
    """Create a locality projector context on a regular algebroid."""
    return LocalityProjector(alg, name, admissible=admissible)


class LocalityProjectorAbsorptionDefinition(Definition):
    """``𝒫(L(Df, u, v)) → L(Df, u, v)`` — the projected operator is
    locally equivalent to the original (``L̂ ∈ [L̃]``: agreement on
    coboundary forms is exactly the equivalence [MC Def 3.6])."""

    anchor = MappedSection

    def __init__(self, projector: LocalityProjector) -> None:
        self._projector = projector
        alg = projector.algebroid
        self.name = (
            f"locality projector {projector.name} ({alg.name}): "
            "P(L(Df,u,v)) = L(Df,u,v)  (L̂ ∈ [L̃])"
        )

    def matches(self, expr: Expr) -> bool:
        alg = self._projector.algebroid
        if not (
            isinstance(expr, MappedSection)
            and expr.map_name == self._projector.name
        ):
            return False
        inner = expr.section
        return (
            isinstance(inner, LocalityOperator)
            and inner.algebroid_name == alg.name
            and isinstance(inner.form, CoboundaryForm)
            and inner.form.algebroid_name == alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        return expr.section


class LocalityProjectorIdempotenceDefinition(Definition):
    """``𝒫(𝒫(L(ω,u,v))) → 𝒫(L(ω,u,v))`` — the mechanizable face of
    ``𝒫|_{ker ρ} = id`` (admissible-connections paper, Def 3.8):
    projected locality values are certified to lie in ``ker ρ``, and
    the stronger definition makes ``𝒫`` fix them. Registered only
    for projectors created with ``admissible=True``."""

    anchor = MappedSection

    def __init__(self, projector: LocalityProjector) -> None:
        self._projector = projector
        alg = projector.algebroid
        self.name = (
            f"locality projector {projector.name} ({alg.name}): "
            "P(P(L(ω,u,v))) = P(L(ω,u,v))  (P|_ker ρ = id)"
        )

    def matches(self, expr: Expr) -> bool:
        alg = self._projector.algebroid
        if not (
            isinstance(expr, MappedSection)
            and expr.map_name == self._projector.name
        ):
            return False
        inner = expr.section
        return (
            isinstance(inner, MappedSection)
            and inner.map_name == self._projector.name
            and isinstance(inner.section, LocalityOperator)
            and inner.section.algebroid_name == alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        return expr.section


class LocalityProjectorKernelDefinition(Definition):
    """``ρ(𝒫(L(ω, u, v))) → 0`` — the projected locality values lie in
    the anchor's kernel (``im(L̂) ⊂ ker(ρ)``), for an ARBITRARY form
    slot ``ω``. (For ``ω = Df`` absorption fires first and the
    vanishing is the cited pre-Leibniz theorem instead.)"""

    anchor = AnchoredVF

    def __init__(self, projector: LocalityProjector) -> None:
        self._projector = projector
        alg = projector.algebroid
        self.name = (
            f"locality projector {projector.name} ({alg.name}): "
            "ρ(P(L(ω,u,v))) = 0  (im(L̂) ⊂ ker ρ)"
        )

    def matches(self, expr: Expr) -> bool:
        alg = self._projector.algebroid
        if not (
            isinstance(expr, AnchoredVF)
            and expr.algebroid_name == alg.name
        ):
            return False
        sec = expr.section
        return (
            isinstance(sec, MappedSection)
            and sec.map_name == self._projector.name
            and isinstance(sec.section, LocalityOperator)
            and sec.section.algebroid_name == alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)
