"""
The generic axiom-suite façade (research interface, Phase 8 brick 2):
hand it a bracket, an anchor, a pairing and a coboundary ``D`` on a
:class:`~jacopy.research.sections.SectionType`, and it runs the
Courant-type conditions — right-Leibniz, symmetric part, anchor
morphism, Leibniz–Jacobi — *component by component*, choosing for each
slot how to evaluate it (a vector on a probe function, a ``k``-form on
``k`` fresh vector slots), and reports what closed, what left a
residual and what that residual is.

The Ψ-transport of [2409.11973 §7] is one method:
:meth:`AlgebroidData.transport` builds the primed data

    [·,·]' = Ψ⁻¹[Ψ·,Ψ·],   ρ' = ρ∘Ψ,   ⟨·,·⟩' = ⟨Ψ·,Ψ·⟩,   D' = Ψ⁻¹∘D,

so the same suite runs on the twisted structure unchanged.

Nothing is assumed silently: the engine is assembled from the
expressions actually checked (:func:`~jacopy.research.engine_assembly.
assemble_engine`), a Poisson/fundamental-identity declaration is an
explicit flag, and a residual is reported as a residual.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.research.block_matrix import BlockMatrix
from jacopy.research.sections import (
    GeneralizedSection,
    SectionType,
    sum_components,
)


# ------------------------------------------------------------------ #
# Brackets and algebroid data                                         #
# ------------------------------------------------------------------ #


class Bracket:
    """A bracket on sections of one :class:`SectionType`."""

    def __init__(
        self,
        type_: SectionType,
        fn: Callable[[GeneralizedSection, GeneralizedSection], GeneralizedSection],
        *,
        name: str = "[·,·]",
    ) -> None:
        self._type = type_
        self._fn = fn
        self.name = name

    @property
    def type(self) -> SectionType:
        return self._type

    def __call__(
        self, e1: GeneralizedSection, e2: GeneralizedSection
    ) -> GeneralizedSection:
        for e in (e1, e2):
            if e.type != self._type:
                raise TypeError(
                    f"{self.name} lives on {self._type}, got {e.type}"
                )
        out = self._fn(e1, e2)
        if not isinstance(out, GeneralizedSection):
            out = GeneralizedSection(self._type, *out)
        return out

    @classmethod
    def from_components(
        cls, type_: SectionType, fn, *, name: str = "[·,·]"
    ) -> "Bracket":
        """Wrap a component-level function ``fn(*e1, *e2) -> tuple``
        (the shape of the library's bracket formulas)."""
        return cls(
            type_,
            lambda e1, e2: GeneralizedSection(type_, *fn(*e1, *e2)),
            name=name,
        )

    def twist(
        self,
        psi: BlockMatrix,
        psi_inverse: Optional[BlockMatrix] = None,
        *,
        name: Optional[str] = None,
    ) -> "Bracket":
        """Eq. (7.3): ``[e₁,e₂]_Ψ = Ψ⁻¹[Ψe₁, Ψe₂]``."""
        inv = psi_inverse if psi_inverse is not None else psi.inverse()
        return Bracket(
            self._type,
            lambda e1, e2: inv(self(psi(e1), psi(e2))),
            name=name or f"{self.name}_{psi.name}",
        )


@dataclass
class AlgebroidData:
    """A bracket with its anchor, pairing and coboundary.

    * ``anchor(e) -> Expr`` — the vector field ``ρ(e)``;
    * ``pairing(e1, e2) -> tuple[Expr, ...]`` — the pairing's
      components (one scalar for ``TM ⊕ T*M``; a 1-form and a 4-form
      for the exceptional bundle);
    * ``D(pairing_components) -> GeneralizedSection`` — the coboundary
      applied to the pairing, as a section (``D⟨e₁,e₂⟩``).
    """

    bracket: Bracket
    anchor: Callable[[GeneralizedSection], Expr]
    pairing: Optional[Callable[..., Tuple[Expr, ...]]] = None
    D: Optional[Callable[..., GeneralizedSection]] = None
    name: str = "E"

    @property
    def type(self) -> SectionType:
        return self.bracket.type

    def transport(
        self,
        psi: BlockMatrix,
        psi_inverse: Optional[BlockMatrix] = None,
        *,
        name: Optional[str] = None,
    ) -> "AlgebroidData":
        """The primed data of the Ψ-twist (13j): ``ρ' = ρ∘Ψ``,
        ``⟨·,·⟩' = ⟨Ψ·,Ψ·⟩``, ``D' = Ψ⁻¹∘D`` and the twisted bracket."""
        inv = psi_inverse if psi_inverse is not None else psi.inverse()
        pairing = None
        if self.pairing is not None:
            pairing = lambda e1, e2: self.pairing(psi(e1), psi(e2))
        D = None
        if self.D is not None:
            D = lambda *parts: inv(self.D(*parts))
        return AlgebroidData(
            bracket=self.bracket.twist(psi, inv),
            anchor=lambda e: self.anchor(psi(e)),
            pairing=pairing,
            D=D,
            name=name or f"{self.name}_{psi.name}",
        )


# ------------------------------------------------------------------ #
# Reports                                                             #
# ------------------------------------------------------------------ #


@dataclass
class CheckResult:
    axiom: str
    component: str
    status: str  # CLOSED | RESIDUAL | FAILED
    seconds: float
    steps: Optional[int] = None
    residual: Optional[Expr] = None
    note: str = ""

    @property
    def closed(self) -> bool:
        return self.status == "CLOSED"


@dataclass
class SuiteReport:
    title: str
    results: List[CheckResult] = field(default_factory=list)

    @property
    def all_closed(self) -> bool:
        return all(r.closed for r in self.results)

    def extend(self, other: "SuiteReport") -> "SuiteReport":
        self.results.extend(other.results)
        return self

    def __repr__(self) -> str:
        lines = [f"{self.title}"]
        for r in self.results:
            extra = f" {r.steps} steps" if r.steps is not None else ""
            lines.append(
                f"  {r.status:<8} {r.axiom:<22} {r.component:<14}"
                f" ({r.seconds:5.2f}s{extra})"
            )
            if r.residual is not None:
                lines.append(
                    f"           residual: {r.residual._repr_inner()[:140]}"
                )
            if r.note:
                lines.append(f"           note: {r.note}")
        verdict = "ALL CLOSED" if self.all_closed else "NOT ALL CLOSED"
        lines.append(f"  → {verdict}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# The suite                                                           #
# ------------------------------------------------------------------ #


class AxiomSuite:
    """Run the Courant-type conditions of an :class:`AlgebroidData`.

    ``probe`` is the scalar function vector-valued identities act on;
    form-valued identities are evaluated on fresh vector fields named
    ``X₁, X₂, …``. The engine is assembled per check from the
    expressions involved unless one is supplied.
    """

    def __init__(
        self,
        data: AlgebroidData,
        *,
        registry: Optional[PropertyRegistry] = None,
        probe: Optional[Expr] = None,
        engine=None,
        structures: Sequence = (),
        declare_fi: bool = False,
        extra_rules: Sequence = (),
    ) -> None:
        from jacopy.central.objects import functions, vector_fields

        self.data = data
        self.registry = registry if registry is not None else PropertyRegistry()
        if probe is None:
            (probe,) = functions("h_probe", registry=self.registry)
        self.probe = probe
        self._engine = engine
        self.structures = tuple(structures)
        self.declare_fi = declare_fi
        self.extra_rules = tuple(extra_rules)
        top = max(
            (s.degree for s in data.type if s.kind == "form"), default=0
        )
        self._slots = vector_fields(
            " ".join(f"X{_sub(i)}" for i in range(1, top + 1))
        ) if top else ()
        self.last_engine = None

    @property
    def last_report(self) -> List[str]:
        """Assembly report of the most recently built engine."""
        from jacopy.research.engine_assembly import assembly_report

        return assembly_report(self.last_engine) if self.last_engine is not None else []

    # ---- machinery ------------------------------------------------ #

    def engine_for(self, *exprs: Expr):
        if self._engine is not None:
            return self._engine
        from jacopy.research.engine_assembly import assemble_engine

        eng = assemble_engine(
            *exprs,
            registry=self.registry,
            structures=self.structures,
            declare_fi=self.declare_fi,
            extra=self.extra_rules,
        )
        self.last_engine = eng
        return eng

    def _evaluate(self, comp: Expr, slot) -> Expr:
        """The scalar/evaluated face of one component."""
        if slot.kind == "vector":
            return Act(comp, self.probe)
        if slot.kind == "function":
            return comp
        k = slot.degree
        if k == 0:
            return comp
        if k == 1:
            return Pairing(comp, self._slots[0])
        return MultiEval(
            comp, *self._slots[:k], alternating=True, slot_kind="vector"
        )

    def _zero_by_components(
        self, axiom: str, diff: GeneralizedSection
    ) -> SuiteReport:
        from jacopy.packages.poisson.tilde import _normalized_by

        rep = SuiteReport(axiom)
        for comp, slot in zip(diff, diff.type):
            node = self._evaluate(comp, slot)
            eng = self.engine_for(node)
            t = time.time()
            residual = _normalized_by(eng, node, self.registry)
            ok = residual == Integer(0)
            rep.results.append(
                CheckResult(
                    axiom,
                    slot.label,
                    "CLOSED" if ok else "RESIDUAL",
                    time.time() - t,
                    residual=None if ok else residual,
                )
            )
        return rep

    # ---- the axioms ----------------------------------------------- #

    def right_leibniz(
        self, e1: GeneralizedSection, e2: GeneralizedSection, f: Expr
    ) -> SuiteReport:
        """``[e₁, f·e₂] = f·[e₁,e₂] + (ρ(e₁)f)·e₂``."""
        br, rho = self.data.bracket, self.data.anchor
        diff = (
            br(e1, e2.scale(f))
            - br(e1, e2).scale(f)
            - e2.scale(Act(rho(e1), f))
        )
        return self._zero_by_components("right-Leibniz", diff)

    def symmetric_part(
        self, e1: GeneralizedSection, e2: GeneralizedSection
    ) -> SuiteReport:
        """``[e₁,e₂] + [e₂,e₁] = D⟨e₁,e₂⟩``."""
        if self.data.pairing is None or self.data.D is None:
            raise ValueError("symmetric part needs a pairing and D")
        br = self.data.bracket
        diff = br(e1, e2) + br(e2, e1) - self.data.D(
            *self.data.pairing(e1, e2)
        )
        return self._zero_by_components("symmetric part", diff)

    def anchor_morphism(
        self, e1: GeneralizedSection, e2: GeneralizedSection
    ) -> SuiteReport:
        """``ρ([e₁,e₂]) = [ρe₁, ρe₂]_Lie`` (probed); a residual here
        is the *obstruction* (the Poisson condition's analogue)."""
        from jacopy.central.tangent.lie_bracket import lie_bracket
        from jacopy.packages.poisson.tilde import _normalized_by

        br, rho = self.data.bracket, self.data.anchor
        node = Act(
            Sum(rho(br(e1, e2)), Neg(lie_bracket(rho(e1), rho(e2)))),
            self.probe,
        )
        eng = self.engine_for(node)
        t = time.time()
        residual = _normalized_by(eng, node, self.registry)
        ok = residual == Integer(0)
        rep = SuiteReport("anchor morphism")
        rep.results.append(
            CheckResult(
                "anchor morphism",
                "ρ on probe",
                "CLOSED" if ok else "RESIDUAL",
                time.time() - t,
                residual=None if ok else residual,
            )
        )
        return rep

    def jacobi(
        self,
        e1: GeneralizedSection,
        e2: GeneralizedSection,
        e3: GeneralizedSection,
        *,
        max_steps: int = 20000,
        components: Optional[Sequence[int]] = None,
    ) -> SuiteReport:
        """Leibniz–Jacobi ``[e₁,[e₂,e₃]] = [[e₁,e₂],e₃] + [e₂,[e₁,e₃]]``
        through the bracket-identity repair loop (bounded by
        ``max_steps``; FAILED is reported honestly)."""
        from jacopy.central.tangent.cartan import (
            prove_with_bracket_identities,
        )
        from jacopy.proof.strategies import ProofFailure

        br = self.data.bracket
        diff = br(e1, br(e2, e3)) - br(br(e1, e2), e3) - br(e2, br(e1, e3))
        rep = SuiteReport("Leibniz-Jacobi")
        for i, (comp, slot) in enumerate(zip(diff, diff.type)):
            if components is not None and i not in components:
                continue
            node = self._evaluate(comp, slot)
            eng = self.engine_for(node)
            t = time.time()
            try:
                chain, _ = prove_with_bracket_identities(
                    node,
                    Integer(0),
                    self.probe,
                    registry=self.registry,
                    engine=eng,
                    max_steps=max_steps,
                )
                rep.results.append(
                    CheckResult(
                        "Leibniz-Jacobi", slot.label, "CLOSED",
                        time.time() - t, steps=len(chain.steps),
                    )
                )
            except ProofFailure as exc:
                rep.results.append(
                    CheckResult(
                        "Leibniz-Jacobi", slot.label, "FAILED",
                        time.time() - t, note=str(exc)[:160],
                    )
                )
        return rep

    def run(
        self,
        e1: GeneralizedSection,
        e2: GeneralizedSection,
        f: Expr,
        *,
        e3: Optional[GeneralizedSection] = None,
        jacobi_max_steps: int = 20000,
    ) -> SuiteReport:
        """Right-Leibniz + symmetric part (if pairing/D given) +
        anchor morphism (+ Jacobi when ``e3`` is given)."""
        rep = SuiteReport(f"axiom suite for {self.data.name} on {self.data.type}")
        rep.extend(self.right_leibniz(e1, e2, f))
        if self.data.pairing is not None and self.data.D is not None:
            rep.extend(self.symmetric_part(e1, e2))
        rep.extend(self.anchor_morphism(e1, e2))
        if e3 is not None:
            rep.extend(self.jacobi(e1, e2, e3, max_steps=jacobi_max_steps))
        return rep


def _sub(n: int) -> str:
    return "".join("₀₁₂₃₄₅₆₇₈₉"[int(c)] for c in str(n))
