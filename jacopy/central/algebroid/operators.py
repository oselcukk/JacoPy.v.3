"""
Algebroid defect operators (PDF items 10e-g; Phases 3.B-3.C).

Three operators, each measuring the failure of a structural property
(so ``≡ 0`` characterizes the property):

* **Jacobiator** (item 10e; Leibniz form):
  ``J^E(u,v,w) := [u,[v,w]]_E − [[u,v]_E,w]_E − [v,[u,w]_E]_E``;
  ``J^E ≡ 0`` ⟺ Leibniz-Jacobi (the carrier of the ``jacobi``
  declaration — a bracket-level Jacobi rewrite would not terminate).
* **Derivator** (item 10f; also MC's tool for the ``C^∞``-module
  analysis and the Midterm Q3 duality conditions):
  ``D^E_Φ(u,v) := Φ[u,v]_E − [Φu,v]_E − [u,Φv]_E`` for a
  ``C^∞``-linear ``Φ: E → E``; ``D^E_Φ ≡ 0`` ⟺ Φ derives the bracket.
* **Predator** (item 10g):
  ``P^E_Φ(u,v) := Φ[u,v]_E − [Φu, Φv]_Lie`` for a ``C^∞``-linear
  ``Φ: E → TM``; ``P^E_Φ ≡ 0`` ⟺ Φ is a morphism onto the Lie
  bracket. **In particular ``P^E_ρ ≡ 0`` ⟺ the pre-Leibniz
  (anchor-morphism) property** — proved as the 3.C closing theorem.

``C^∞``-linear maps are modelled like the anchor: a :class:`SectionMap`
context produces opaque :class:`MappedSection` atoms ``Φ(u)`` (again
sections, so again degree-0 derivation atoms), with linearity as a
definitional rule — a bundle-morphism is ``C^∞``-linear by definition.

Every operator node has a definitional expansion into brackets; the
declared vanishings (``jacobi``) register at higher precedence.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Derivation
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition
from jacopy.central.calculus.scalars import is_scalar_function
from jacopy.central.algebroid.context import Algebroid
from jacopy.central.objects.bundle import Bundle, TangentBundle


class MappedSection(Derivation):
    """``Φ(u)`` — the image of a section under a ``C^∞``-linear map,
    itself a section (of the map's target bundle). The exact pattern of
    :class:`~jacopy.central.algebroid.context.AnchoredVF` (the anchor
    is the special case ``Φ = ρ: E → TM``)."""

    __slots__ = ("_map_name", "_section")

    def __init__(
        self,
        map_name: str,
        section: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(section, Expr):
            raise TypeError("MappedSection requires an Expr section")
        display = (
            name
            if name is not None
            else f"{map_name}({section._repr_inner()})"
        )
        super().__init__(display, degree=0)
        self._map_name = map_name
        self._section = section

    @property
    def map_name(self) -> str:
        return self._map_name

    @property
    def section(self) -> Expr:
        return self._section

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._section,)

    def with_slots(self, section: Expr) -> "MappedSection":
        return MappedSection(self._map_name, section)

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        """Slot-walking substitution (Phase 4.E.2)."""
        if self == dummy:
            return target
        new_section = self._section.substitute_atom(dummy, target)
        if new_section is self._section:
            return self
        return MappedSection(self._map_name, new_section)

    def _key(self) -> Any:
        return (self._name, self._degree, self._map_name, self._section)


class SectionMap:
    """A named ``C^∞``-linear bundle map ``Φ: source → target``
    (context object; the derivator wants ``E → E``, the predator
    ``E → TM``)."""

    __slots__ = ("_name", "_source", "_target")

    def __init__(self, name: str, source: Bundle, target: Bundle) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("SectionMap name must be a non-empty str")
        if not isinstance(source, Bundle) or not isinstance(target, Bundle):
            raise TypeError("SectionMap source/target must be Bundles")
        self._name = name
        self._source = source
        self._target = target

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> Bundle:
        return self._source

    @property
    def target(self) -> Bundle:
        return self._target

    @property
    def targets_tangent(self) -> bool:
        return isinstance(self._target, TangentBundle)

    def __call__(self, u: Expr) -> MappedSection:
        """``Φ(u)`` — the opaque mapped section."""
        if not isinstance(u, Expr):
            raise TypeError("SectionMap expects an Expr section")
        return MappedSection(self._name, u)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SectionMap)
            and self._name == other._name
            and self._source == other._source
            and self._target == other._target
        )

    def __hash__(self) -> int:
        return hash(("section-map", self._name, self._source, self._target))

    def __repr__(self) -> str:
        return (
            f"SectionMap({self._name!r}: {self._source!r} → "
            f"{self._target!r})"
        )


def section_map(name: str, source: Bundle, target: Bundle) -> SectionMap:
    """Create a named ``C^∞``-linear bundle map."""
    return SectionMap(name, source, target)


class MappedSectionLinearityDefinition(Definition):
    """``C^∞``-linearity of a bundle map (definitional):
    ``Φ(fu + v) = f·Φ(u) + Φ(v)``, ``Φ(−u) = −Φ(u)``, ``Φ(0) = 0``."""

    name = "section-map linearity: Φ(fu + v) = f·Φ(u) + Φ(v)"
    anchor = MappedSection

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _scalar_split(self, expr: Expr):
        if isinstance(expr, Product) and len(expr.children) >= 2:
            head = expr.children[0]
            if is_scalar_function(head, self._registry):
                rest = expr.children[1:]
                return head, (rest[0] if len(rest) == 1 else Product(*rest))
        return None

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MappedSection):
            return False
        sec = expr.section
        if isinstance(sec, (Sum, Neg)) or sec == Integer(0):
            return True
        return self._scalar_split(sec) is not None

    def rewrite(self, expr: Expr) -> Expr:
        m = expr.map_name
        sec = expr.section

        def phi(s: Expr) -> Expr:
            return MappedSection(m, s)

        if sec == Integer(0):
            return Integer(0)
        if isinstance(sec, Sum):
            return Sum(*(phi(c) for c in sec.children))
        if isinstance(sec, Neg):
            return Neg(phi(sec.arg))
        scalar, rest = self._scalar_split(sec)
        return Product(scalar, phi(rest))


class Jacobiator(Derivation):
    """``J^E(u, v, w)`` — the Leibniz-Jacobi defect, itself a section
    of ``E`` (degree-0 atom acting through the anchor only)."""

    __slots__ = ("_algebroid_name", "_u", "_v", "_w")

    def __init__(
        self,
        algebroid_name: str,
        u: Expr,
        v: Expr,
        w: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        for s in (u, v, w):
            if not isinstance(s, Expr):
                raise TypeError("Jacobiator requires Expr sections")
        display = (
            name
            if name is not None
            else (
                f"J_{algebroid_name}({u._repr_inner()},"
                f"{v._repr_inner()},{w._repr_inner()})"
            )
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._u = u
        self._v = v
        self._w = w

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def sections(self):
        return (self._u, self._v, self._w)

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._u, self._v, self._w)

    def with_slots(self, u: Expr, v: Expr, w: Expr) -> "Jacobiator":
        return Jacobiator(self._algebroid_name, u, v, w)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._algebroid_name,
            self._u,
            self._v,
            self._w,
        )


def jacobiator(alg: Algebroid, u: Expr, v: Expr, w: Expr) -> Jacobiator:
    """``J^E(u, v, w)`` for the given algebroid context."""
    if not isinstance(alg, Algebroid):
        raise TypeError("jacobiator expects an Algebroid")
    return Jacobiator(alg.name, u, v, w)


class JacobiatorExpansionDefinition(Definition):
    """Definitional expansion
    ``J^E(u,v,w) → [u,[v,w]] − [[u,v],w] − [v,[u,w]]``."""

    anchor = Jacobiator

    def __init__(self, alg: Algebroid) -> None:
        if not isinstance(alg, Algebroid):
            raise TypeError(
                "JacobiatorExpansionDefinition expects an Algebroid"
            )
        self._alg = alg
        self.name = (
            f"Jacobiator definition ({alg.name}): "
            "J(u,v,w) = [u,[v,w]] − [[u,v],w] − [v,[u,w]]"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Jacobiator)
            and expr.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        u, v, w = expr.sections
        br = self._alg.bracket
        return Sum(
            br(u, br(v, w)),
            Neg(br(br(u, v), w)),
            Neg(br(v, br(u, w))),
        )


class Derivator(Derivation):
    """``D^E_Φ(u, v)`` — the bracket-derivation defect of a
    ``C^∞``-linear ``Φ: E → E`` (item 10f), itself a section of E."""

    __slots__ = ("_algebroid_name", "_map_name", "_u", "_v")

    def __init__(
        self,
        algebroid_name: str,
        map_name: str,
        u: Expr,
        v: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        for s in (u, v):
            if not isinstance(s, Expr):
                raise TypeError("Derivator requires Expr sections")
        display = (
            name
            if name is not None
            else (
                f"D_{map_name}({u._repr_inner()},{v._repr_inner()})"
            )
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._map_name = map_name
        self._u = u
        self._v = v

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def map_name(self) -> str:
        return self._map_name

    @property
    def sections(self):
        return (self._u, self._v)

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._u, self._v)

    def with_slots(self, u: Expr, v: Expr) -> "Derivator":
        return Derivator(self._algebroid_name, self._map_name, u, v)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._algebroid_name,
            self._map_name,
            self._u,
            self._v,
        )


def derivator(
    alg: Algebroid, phi: SectionMap, u: Expr, v: Expr
) -> Derivator:
    """``D^E_Φ(u, v)`` — requires ``Φ: E → E``."""
    if not isinstance(alg, Algebroid):
        raise TypeError("derivator expects an Algebroid")
    if not isinstance(phi, SectionMap):
        raise TypeError("derivator expects a SectionMap")
    if phi.source != alg.bundle or phi.target != alg.bundle:
        raise ValueError(
            f"derivator needs Φ: E → E on {alg.bundle!r}; got "
            f"{phi.source!r} → {phi.target!r}"
        )
    return Derivator(alg.name, phi.name, u, v)


class DerivatorExpansionDefinition(Definition):
    """``D_Φ(u,v) → Φ[u,v]_E − [Φu, v]_E − [u, Φv]_E``."""

    anchor = Derivator

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"Derivator definition ({alg.name}): "
            "D_Φ(u,v) = Φ[u,v] − [Φu,v] − [u,Φv]"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Derivator)
            and expr.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        u, v = expr.sections
        m = expr.map_name
        br = self._alg.bracket
        return Sum(
            MappedSection(m, br(u, v)),
            Neg(br(MappedSection(m, u), v)),
            Neg(br(u, MappedSection(m, v))),
        )


class Predator(Derivation):
    """``P^E_Φ(u, v) := Φ[u,v]_E − [Φu, Φv]_Lie`` — the
    Lie-morphism defect of ``Φ: E → TM`` (item 10g). ``Φ = ρ`` gives
    the pre-Leibniz defect: ``P^E_ρ ≡ 0 ⟺ anchor-morphism``."""

    __slots__ = ("_algebroid_name", "_map_name", "_u", "_v")

    def __init__(
        self,
        algebroid_name: str,
        map_name: str,
        u: Expr,
        v: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        for s in (u, v):
            if not isinstance(s, Expr):
                raise TypeError("Predator requires Expr sections")
        display = (
            name
            if name is not None
            else (
                f"P_{map_name}({u._repr_inner()},{v._repr_inner()})"
            )
        )
        super().__init__(display, degree=0)
        self._algebroid_name = algebroid_name
        self._map_name = map_name
        self._u = u
        self._v = v

    @property
    def algebroid_name(self) -> str:
        return self._algebroid_name

    @property
    def map_name(self) -> str:
        return self._map_name

    @property
    def sections(self):
        return (self._u, self._v)

    @property
    def wedge_degree(self) -> Degree:
        return Degree.const(1)

    @property
    def rewritable_slots(self):
        return (self._u, self._v)

    def with_slots(self, u: Expr, v: Expr) -> "Predator":
        return Predator(self._algebroid_name, self._map_name, u, v)

    def _key(self) -> Any:
        return (
            self._name,
            self._degree,
            self._algebroid_name,
            self._map_name,
            self._u,
            self._v,
        )


def predator(
    alg: Algebroid, phi: SectionMap, u: Expr, v: Expr
) -> Predator:
    """``P^E_Φ(u, v)`` — requires ``Φ: E → TM``."""
    if not isinstance(alg, Algebroid):
        raise TypeError("predator expects an Algebroid")
    if not isinstance(phi, SectionMap):
        raise TypeError("predator expects a SectionMap")
    if phi.source != alg.bundle or not phi.targets_tangent:
        raise ValueError(
            f"predator needs Φ: E → TM; got {phi.source!r} → "
            f"{phi.target!r}"
        )
    return Predator(alg.name, phi.name, u, v)


def anchor_predator(alg: Algebroid, u: Expr, v: Expr) -> Predator:
    """``P^E_ρ(u, v)`` — the predator of the anchor itself; its
    vanishing is exactly the pre-Leibniz property."""
    if not isinstance(alg, Algebroid):
        raise TypeError("anchor_predator expects an Algebroid")
    return Predator(alg.name, alg.anchor_name, u, v)


class PredatorExpansionDefinition(Definition):
    """``P_Φ(u,v) → Φ[u,v]_E − [Φu, Φv]_Lie``; for ``Φ = ρ`` (the
    anchor's own name) the images are built with the algebroid's
    anchor, so the pre-Leibniz declaration can act on them."""

    anchor = Predator

    def __init__(self, alg: Algebroid) -> None:
        self._alg = alg
        self.name = (
            f"Predator definition ({alg.name}): "
            "P_Φ(u,v) = Φ[u,v] − [Φu, Φv]_Lie"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Predator)
            and expr.algebroid_name == self._alg.name
        )

    def rewrite(self, expr: Expr) -> Expr:
        u, v = expr.sections
        m = expr.map_name
        if m == self._alg.anchor_name:
            image = self._alg.anchor
        else:
            def image(s: Expr) -> Expr:
                return MappedSection(m, s)
        return Sum(
            image(self._alg.bracket(u, v)),
            Neg(LieBracketVF(image(u), image(v))),
        )
