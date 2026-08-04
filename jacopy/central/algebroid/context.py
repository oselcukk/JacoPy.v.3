"""
The algebroid context: anchored vector bundles, the opaque algebroid
bracket, and the coboundary map (PDF item 10a-c; Phase 3.A).

Source papers (in-repo): ``metric-connection.pdf`` (MC; Dereli-Doğan,
arXiv:2006.05957) and ``pre-metric-bourbaki.pdf`` (B; Çatal-Özer-
Dereli-Doğan, arXiv:2210.00548). Definitions follow them exactly.

An **anchored vector bundle** is a pair ``(E, ρ)`` with a bundle
morphism ``ρ: E → TM`` [MC Def before 3.2]. Being a bundle morphism,
``ρ`` is ``C^∞(M)``-linear **by definition** — that linearity is a
definitional rule here, not a declared axiom. The anchor lets
sections of ``E`` act on functions:

    u(f) := ρ(u)(f),

and induces the **coboundary** ``D := ρ* ∘ d`` with
``(Df)(u) = ρ(u)(f)`` [MC Def 3.5] — the first-order operator of the
symmetric-part decomposition in the Bourbaki hierarchy
(``D_E := ρ*d`` [B §4]).

The bracket ``[·,·]_E`` is an **ℝ-bilinear** map on sections [B Def
4.1] with NO canonical formula — hence the opaque
:class:`AlgebroidBracket` atom. Everything beyond ℝ-bilinearity
(right-Leibniz, anchor morphism, Leibniz-Jacobi, …) belongs to the
declared hierarchy (Phase 3.B): unlike the TM case, algebroid proofs
run from *assumptions*, and this module encodes only what holds by
definition.

**The E = TM reduction (PDF item 8a).** :func:`tangent_algebroid`
returns the context with the identity anchor and the Lie bracket; its
anchor is the identity *literally* (``anchor(u) is u``), its bracket
IS :func:`~jacopy.central.tangent.lie_bracket.lie_bracket`, and its
coboundary is the usual ``df``. The algebroid engine delegates to the
tangent engine for it — the usual calculus is the instantiation, not
a parallel code path.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Atom, Expr
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects.bundle import Bundle, TM, TangentBundle
from jacopy.central.objects.vector_field import VectorField, vector_fields


class AnchoredVF(Derivation):
    """``ρ(u)`` — the anchored image of a section, a genuine vector
    field (degree-0 derivation on functions).

    Opaque, like :class:`~jacopy.algebra.lie_bracket_vf.LieBracketVF`:
    the anchor has no formula, so ``ρ(u)`` is an atom carrying the
    algebroid identity and the section. Its ``C^∞``-linearity
    (``ρ(fu) = f·ρ(u)``) is definitional and lives in
    :mod:`jacopy.central.algebroid.rules`.
    """

    __slots__ = ("_algebroid_name", "_section", "_anchor_name")

    def __init__(
        self,
        algebroid_name: str,
        section: Expr,
        *,
        anchor_name: str = "ρ",
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(section, Expr):
            raise TypeError("AnchoredVF requires an Expr section")
        display = (
            name
            if name is not None
            else f"{anchor_name}({section._repr_inner()})"
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._section = section
        self._anchor_name = anchor_name

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def section(self) -> Expr:
        return self._section

    @property
    def wedge_degree(self) -> Degree:
        """``ρ(u)`` is a 1-vector in the multivector grading."""
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._section,)

    def with_slots(self, section: Expr) -> "AnchoredVF":
        return AnchoredVF(
            self._algebroid_name, section, anchor_name=self._anchor_name
        )

    def _key(self) -> Any:
        return (self._name, self._degree, self._algebroid_name, self._section)


class AlgebroidBracket(Derivation):
    """``[u, v]_E`` — the opaque algebroid bracket of two sections.

    Again a section of ``E`` (so again a degree-0 atom that the anchor
    can be applied to); ℝ-bilinear by definition [B Def 4.1], with all
    further structure declared (Phase 3.B). Note the contrast with
    :class:`LieBracketVF`: an algebroid bracket does NOT act on
    functions directly — only through the anchor.
    """

    __slots__ = ("_algebroid_name", "_u", "_v")

    def __init__(
        self,
        algebroid_name: str,
        u: Expr,
        v: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(u, Expr) or not isinstance(v, Expr):
            raise TypeError("AlgebroidBracket requires Expr sections")
        display = (
            name
            if name is not None
            else f"[{u._repr_inner()},{v._repr_inner()}]_{algebroid_name}"
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._u = u
        self._v = v

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def u(self) -> Expr:
        return self._u

    @property
    def v(self) -> Expr:
        return self._v

    @property
    def wedge_degree(self) -> Degree:
        """A bracket of sections is again a section (1-vector)."""
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._u, self._v)

    def with_slots(self, u: Expr, v: Expr) -> "AlgebroidBracket":
        return AlgebroidBracket(self._algebroid_name, u, v)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._algebroid_name,
            self._u,
            self._v,
        )


class CoboundaryForm(Atom):
    """``Df`` — the coboundary of a function, an E-1-form.

    ``D := ρ* ∘ d`` with the defining evaluation
    ``⟨Df, u⟩ = ρ(u)(f)`` [MC Def 3.5]; that evaluation is the
    definitional rule :class:`CoboundaryPairingDefinition`. On the
    tangent algebroid ``D`` is the usual ``d`` and this atom is never
    built (see :meth:`Algebroid.D`).
    """

    __slots__ = ("_algebroid_name", "_function", "_bundle")

    def __init__(
        self, algebroid_name: str, f: Expr, bundle: Bundle
    ) -> None:
        if not isinstance(f, Expr):
            raise TypeError("CoboundaryForm requires an Expr function")
        self._algebroid_name = algebroid_name
        self._function = f
        self._bundle = bundle

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._function,)

    def with_slots(self, f: Expr) -> "CoboundaryForm":
        return CoboundaryForm(self._algebroid_name, f, self._bundle)

    def _key(self) -> Any:
        return (self._algebroid_name, self._function, self._bundle)

    def _repr_inner(self) -> str:
        return f"D{self._function._repr_inner()}"


class LocalityOperator(Derivation):
    """``L(ω, u, v)`` — the locality operator [MC Def 3.5], itself a
    section of ``E``.

    The general node: the first slot is an arbitrary ``E``-1-form
    (a :class:`CoboundaryForm` ``Df`` in the left-Leibniz rule — build
    that case with :func:`locality_term`). ``L`` is ``C^∞``-multilinear
    in all three slots BY DEFINITION [MC Def 3.5]; the corresponding
    rules live in :class:`~jacopy.central.algebroid.rules.\
LocalityMultilinearityDefinition`. The locality *structure* (the
    ``Df``-equivalence class) and the projector are Phase 3.E.4.
    """

    __slots__ = ("_algebroid_name", "_form", "_u", "_v")

    def __init__(
        self,
        algebroid_name: str,
        form: Expr,
        u: Expr,
        v: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        for s in (form, u, v):
            if not isinstance(s, Expr):
                raise TypeError("LocalityOperator requires Expr arguments")
        display = (
            name
            if name is not None
            else (
                f"L({form._repr_inner()},{u._repr_inner()},"
                f"{v._repr_inner()})"
            )
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._form = form
        self._u = u
        self._v = v

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def form(self) -> Expr:
        return self._form

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._form, self._u, self._v)

    def with_slots(self, form: Expr, u: Expr, v: Expr) -> "LocalityOperator":
        return LocalityOperator(self._algebroid_name, form, u, v)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        """Slot-walking substitution (Phase 4.E.2 coframe
        recombination reaches the form slot)."""
        if self == dummy:
            return target
        new = tuple(
            s.substitute_atom(dummy, target)
            for s in (self._form, self._u, self._v)
        )
        if new == (self._form, self._u, self._v):
            return self
        return LocalityOperator(self._algebroid_name, *new)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._algebroid_name,
            self._form,
            self._u,
            self._v,
        )


def locality_term(alg: "Algebroid", f: Expr, u: Expr, v: Expr) -> LocalityOperator:
    """``L(Df, u, v)`` — the locality correction of the left-Leibniz
    rule, with the coboundary of ``f`` in the form slot."""
    if not isinstance(alg, Algebroid):
        raise TypeError("locality_term expects an Algebroid")
    if alg.is_tangent:
        raise ValueError(
            "the tangent algebroid has no locality operator: the Lie "
            "bracket's left-Leibniz rule needs no correction term"
        )
    return LocalityOperator(alg.name, alg.D(f), u, v)


class EMetric(Atom):
    """``g(u, v)`` — evaluation of the fibre E-metric on two sections,
    a ``C^∞(M)`` function [MC Def 3.1; B §4].

    An E-metric is BY DEFINITION symmetric and non-degenerate
    [MC Def 3.1]; symmetry is the canonical-slot-order rule
    :class:`~jacopy.central.algebroid.rules.MetricSymmetryDefinition`,
    ``C^∞``-bilinearity (g is a (0,2)-tensor) is
    :class:`~jacopy.central.algebroid.rules.MetricBilinearityDefinition`,
    and non-degeneracy licenses the generic-section agreement step of
    the Phase 3.E.3 tactics (it is never a rewrite).
    """

    __slots__ = ("_algebroid_name", "_metric_name", "_u", "_v")

    def __init__(
        self,
        algebroid_name: str,
        u: Expr,
        v: Expr,
        *,
        metric_name: str = "g",
    ) -> None:
        for s in (u, v):
            if not isinstance(s, Expr):
                raise TypeError("EMetric requires Expr sections")
        self._algebroid_name = algebroid_name
        self._metric_name = metric_name
        self._u = u
        self._v = v

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def u(self) -> Expr:
        return self._u

    @property
    def v(self) -> Expr:
        return self._v

    @property
    def degree(self) -> Degree:
        """A metric evaluation is a scalar function."""
        return Degree.const(0)

    @property
    def rewritable_slots(self):
        return (self._u, self._v)

    def with_slots(self, u: Expr, v: Expr) -> "EMetric":
        return EMetric(
            self._algebroid_name, u, v, metric_name=self._metric_name
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        """Slot-walking substitution (Phase 4.F.2b: Kronecker
        contraction must reach ``∇_{e_s}`` inside a metric slot)."""
        if self == dummy:
            return target
        new_u = self._u.substitute_atom(dummy, target)
        new_v = self._v.substitute_atom(dummy, target)
        if new_u is self._u and new_v is self._v:
            return self
        return self.with_slots(new_u, new_v)

    def _key(self) -> Any:
        return (
            self._algebroid_name,
            self._metric_name,
            self._u,
            self._v,
        )

    def _repr_inner(self) -> str:
        return (
            f"{self._metric_name}({self._u._repr_inner()},"
            f"{self._v._repr_inner()})"
        )


class MetricSharp(Derivation):
    """``g⁻¹(ω)`` — the inverse-metric image of an E-1-form, itself a
    section of ``E`` (the metric analogue of a musical sharp).

    Defining evaluation: ``g(g⁻¹(ω), w) = ⟨ω, w⟩``
    (:class:`~jacopy.central.algebroid.rules.MetricSharpEvaluationDefinition`);
    well-defined because the E-metric is non-degenerate by definition.
    Appears in the symmetric-part axiom ``[u,v] + [v,u] = g⁻¹D g(u,v)``
    [B Def 4.2-4.3].
    """

    __slots__ = ("_algebroid_name", "_metric_name", "_form")

    def __init__(
        self,
        algebroid_name: str,
        form: Expr,
        *,
        metric_name: str = "g",
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(form, Expr):
            raise TypeError("MetricSharp requires an Expr form")
        display = (
            name
            if name is not None
            else f"{metric_name}⁻¹({form._repr_inner()})"
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._metric_name = metric_name
        self._form = form

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def form(self) -> Expr:
        return self._form

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._form,)

    def with_slots(self, form: Expr) -> "MetricSharp":
        return MetricSharp(
            self._algebroid_name,
            form,
            metric_name=self._metric_name,
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        """Slot-walking substitution (Phase 4.F.2: the coframe
        recombination reaches ``g⁻¹(e^s)``'s form slot — the
        ``E = g⁻¹`` instance of the recombination pattern)."""
        if self == dummy:
            return target
        new_form = self._form.substitute_atom(dummy, target)
        if new_form is self._form:
            return self
        return self.with_slots(new_form)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._algebroid_name,
            self._metric_name,
            self._form,
        )


#: Atomic declarable properties (the paper hierarchy's axioms, plus
#: the bundle-level regularity assumption). ``regular`` [MC Def 3.9]
#: states that the anchor has locally constant rank — it carries NO
#: rewrite rule; it licenses structure that needs ``ker(ρ)`` to be a
#: subbundle (the locality projector, Phase 3.E.4).
DECLARATIONS = (
    "right-leibniz",
    "left-leibniz",
    "anchor-morphism",
    "jacobi",
    "antisymmetric",
    "symmetric-part",
    "metric-invariance",
    "regular",
)

#: Hierarchy levels bundling their axioms [B §4, MC §3]. Derivable
#: properties are deliberately ABSENT from the levels that prove them
#: (definition policy: a derived property enters by proof or explicit
#: declaration, never silently):
#:
#: * "leibniz" does NOT include "anchor-morphism" — it follows from
#:   right-Leibniz + Jacobi (the Phase 3.D theorem).
#: * "metric" / "pre-courant" / "courant" do NOT include
#:   "right-leibniz" — it follows from metric invariance + the
#:   E-metric's definitional non-degeneracy [B 4.11, Phase 3.E.3].
#: * "courant" does NOT include "anchor-morphism" — it follows via
#:   [B 4.11] + the 3.D theorem.
#:
#: The metric family [B Def 4.2-4.4]: almost-Courant = almost-Leibniz
#: + E-metric + symmetric part; metric = anchored + ℝ-bilinear +
#: symmetric part + metric invariance; pre-Courant (resp. Courant) =
#: metric + pre-Leibniz (resp. Leibniz), minus the derivable axioms.
LEVELS = {
    "almost-leibniz": ("right-leibniz",),
    "local": ("right-leibniz", "left-leibniz"),
    "pre-leibniz": ("right-leibniz", "anchor-morphism"),
    "leibniz": ("right-leibniz", "jacobi"),
    # Lie algebroid: the antisymmetric Leibniz algebroid. The
    # ``antisymmetric`` axiom ([u,v] = −[v,u]) is what upgrades the
    # conditional Cartan structure to the classical one — d_E² = 0
    # beyond functions genuinely needs it (Phase 3.F).
    "lie": ("right-leibniz", "antisymmetric", "jacobi"),
    "almost-courant": ("right-leibniz", "symmetric-part"),
    "metric": ("symmetric-part", "metric-invariance"),
    "pre-courant": (
        "symmetric-part",
        "metric-invariance",
        "anchor-morphism",
    ),
    "courant": ("symmetric-part", "metric-invariance", "jacobi"),
}


def _normalize_declarations(declare) -> frozenset:
    out = set()
    for item in declare:
        if item in LEVELS:
            out.update(LEVELS[item])
        elif item in DECLARATIONS:
            out.add(item)
        else:
            raise ValueError(
                f"unknown declaration {item!r}; atomic: {DECLARATIONS}, "
                f"levels: {tuple(LEVELS)}"
            )
    return frozenset(out)


class Algebroid:
    """The algebroid context ``(E, ρ, [·,·]_E)`` (PDF item 10a-c).

    A context object (like Bundle/Frame/Connection, not an Expr). At
    this layer it carries only the *definitional* structure: sections,
    the anchor, the ℝ-bilinear bracket, the induced action
    ``u(f) := ρ(u)(f)`` and the coboundary ``D``. The property
    hierarchy (almost-Leibniz → … → Courant) is declared per instance
    in Phase 3.B.

    Parameters
    ----------
    name
        Identity of the algebroid (used in displays: ``[u,v]_E``).
    bundle
        The underlying vector bundle ``E``.
    anchor_name
        Display name of the anchor (``"ρ"``).
    """

    __slots__ = ("_name", "_bundle", "_anchor_name", "_declarations")

    def __init__(
        self,
        name: str,
        bundle: Bundle,
        *,
        anchor_name: str = "ρ",
        declare=(),
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Algebroid name must be a non-empty str")
        if not isinstance(bundle, Bundle):
            raise TypeError("Algebroid bundle must be a Bundle")
        self._name = name
        self._bundle = bundle
        self._anchor_name = anchor_name
        self._declarations = _normalize_declarations(declare)

    # ---- identity ----------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @property
    def anchor_name(self) -> str:
        return self._anchor_name

    @property
    def declarations(self) -> frozenset:
        """The declared axiom set (normalized: levels expanded)."""
        return self._declarations

    def declares(self, prop: str) -> bool:
        return prop in self._declarations

    def with_declarations(self, *declare) -> "Algebroid":
        """A copy of this context with additional declarations."""
        return Algebroid(
            self._name,
            self._bundle,
            anchor_name=self._anchor_name,
            declare=tuple(self._declarations) + tuple(declare),
        )

    @property
    def is_tangent(self) -> bool:
        """The ``E = TM, ρ = id, [·,·] = Lie`` reduction (PDF 8a)."""
        return isinstance(self._bundle, TangentBundle)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Algebroid)
            and self._name == other._name
            and self._bundle == other._bundle
            and self._declarations == other._declarations
        )

    def __hash__(self) -> int:
        return hash(
            ("algebroid", self._name, self._bundle, self._declarations)
        )

    def __repr__(self) -> str:
        return f"Algebroid({self._name!r}, bundle={self._bundle!r})"

    # ---- structure ---------------------------------------------------- #

    def sections(self, names: str) -> Tuple[VectorField, ...]:
        """Sections of ``E`` (paper convention: ``u v w``)."""
        return vector_fields(names, bundle=self._bundle)

    def anchor(self, u: Expr) -> Expr:
        """``ρ(u)`` — the acting vector field of the section ``u``.

        On the tangent algebroid the anchor is the identity map, so
        ``anchor(u) is u`` — the reduction is literal, not simulated.
        """
        if not isinstance(u, Expr):
            raise TypeError("anchor expects an Expr section")
        if self.is_tangent:
            return u
        return AnchoredVF(self._name, u, anchor_name=self._anchor_name)

    def bracket(self, u: Expr, v: Expr) -> Expr:
        """``[u, v]_E`` — the Lie bracket on the tangent algebroid,
        the opaque :class:`AlgebroidBracket` otherwise."""
        if not isinstance(u, Expr) or not isinstance(v, Expr):
            raise TypeError("bracket expects Expr sections")
        if self.is_tangent:
            from jacopy.central.tangent.lie_bracket import lie_bracket

            return lie_bracket(u, v)
        return AlgebroidBracket(self._name, u, v)

    def act(self, u: Expr, f: Expr) -> Act:
        """``u(f) := ρ(u)(f)`` — the anchored action (the Phase 1
        deferral of :mod:`~jacopy.central.objects.vector_field`,
        finally wired)."""
        if not isinstance(f, Expr):
            raise TypeError("act expects an Expr function")
        return Act(self.anchor(u), f)

    def metric(self, u: Expr, v: Expr) -> EMetric:
        """``g(u, v)`` — the fibre E-metric evaluated on two sections
        (a scalar function). One canonical metric ``g`` per context;
        symmetric and non-degenerate BY DEFINITION [MC Def 3.1]."""
        if not isinstance(u, Expr) or not isinstance(v, Expr):
            raise TypeError("metric expects Expr sections")
        if self.is_tangent:
            raise ValueError(
                "the tangent algebroid carries no fibre E-metric at "
                "this layer; Riemannian structure is the Phase 4 "
                "package"
            )
        return EMetric(self._name, u, v)

    def sharp(self, form: Expr) -> MetricSharp:
        """``g⁻¹(ω)`` — the inverse-metric image of an E-1-form, a
        section (well-defined by non-degeneracy)."""
        if not isinstance(form, Expr):
            raise TypeError("sharp expects an Expr form")
        if self.is_tangent:
            raise ValueError(
                "the tangent algebroid carries no fibre E-metric at "
                "this layer; musical maps live in the Phase 4/5 "
                "packages"
            )
        return MetricSharp(self._name, form)

    def D(self, f: Expr) -> Expr:
        """The coboundary ``Df`` (``D = ρ* ∘ d``); the usual ``df`` on
        the tangent algebroid."""
        if not isinstance(f, Expr):
            raise TypeError("D expects an Expr function")
        if self.is_tangent:
            from jacopy.central.tangent.exterior import d

            return d(f)
        return CoboundaryForm(self._name, f, self._bundle)


def algebroid(
    name: str,
    bundle: Optional[Bundle] = None,
    *,
    anchor_name: str = "ρ",
    declare=(),
) -> Algebroid:
    """Create an algebroid context on ``bundle`` (default: a fresh
    abstract bundle named after the algebroid).

    ``declare`` opts into the property hierarchy: atomic axioms
    (:data:`DECLARATIONS`) or bundled levels (:data:`LEVELS`), e.g.
    ``declare=("leibniz",)`` for right-Leibniz + Leibniz-Jacobi.
    """
    if bundle is None:
        bundle = Bundle(name)
    return Algebroid(name, bundle, anchor_name=anchor_name, declare=declare)


def tangent_algebroid() -> Algebroid:
    """``(TM, id, [·,·]_Lie)`` — the PDF item 8a reduction.

    Carries no declarations: on TM every hierarchy property is a
    *proved theorem* of Phase 2, not an assumption.
    """
    return Algebroid("TM", TM)
