"""
Engine auto-assembly (research interface, Phase 8 brick 3): look at the
expressions a user wants normalised, detect which node families occur
(Cartan operators, Nambu sharps of which structures, multivector
interiors, wedges, evaluations, …) and build the rewrite engine that
knows about exactly those — with a human-readable *assembly report*
saying what was detected and which rules were added, so the engine
explains itself.

Honest by construction: a sharp map of a structure the caller did not
hand over is an error (the engine cannot guess that structure's
rules), and the Poisson/fundamental-identity declaration is an explicit
``declare_fi=True`` — never inferred.

Also home of the two slot-reachable wedge rules first written by hand
in ``examples/bracket_twist_walkthrough.ipynb`` (now library rules):

* :class:`WedgeGradedOrderDefinition` — ``α ∧ β = (−1)^{|α||β|} β ∧ α``
  as a canonical ordering, ONLY when every factor has a concrete
  degree (so the sign is exact);
* :class:`WedgeScalarFactorDefinition` — ``(f·α) ∧ β → f·(α ∧ β)``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

from jacopy.core.expr import Expr, Neg, Product
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wedge import Wedge
from jacopy.proof.expansion import Definition, ExpansionEngine


def _wedge_degree(c: Expr, registry) -> Optional[int]:
    """The EXTERIOR-algebra degree of a wedge factor as a concrete
    int, or ``None`` when it is not certain. A vector field has
    operator degree 0 but wedge degree 1 — the ``wedge_degree`` lift
    wins whenever a node carries one, and a derivation without a lift
    is refused rather than graded as a scalar (2026-09-09 audit,
    finding F1: sorting ``Y ∧ X`` as if both were even proved a false
    equality)."""
    from jacopy.algebra.derivation import Derivation, degree_of
    from jacopy.core.symbolic_degree import Degree

    lift = getattr(c, "wedge_degree", None)
    if isinstance(lift, Degree):
        try:
            return lift.as_int()
        except ValueError:
            return None
    if isinstance(c, Derivation):
        return None
    if isinstance(c, Product):
        total = 0
        for k in c.children:
            kk = _wedge_degree(k, registry)
            if kk is None:
                return None
            total += kk
        return total
    if isinstance(c, Neg):
        return _wedge_degree(c.arg, registry)
    try:
        return degree_of(c, registry).as_int()
    except ValueError:
        return None


class ZeroProductDefinition(Definition):
    """``0 · x → 0`` — slot-reachable: the simplifier does not enter
    operator argument slots, so ``Π(0·0)`` used to survive there
    (2026-09-09 audit, finding F3)."""

    name = "zero product: 0·x = 0 (slot-reachable)"
    anchor = Product

    def matches(self, expr: Expr) -> bool:
        from jacopy.core.expr import Integer

        return isinstance(expr, Product) and any(
            c == Integer(0) for c in expr.children
        )

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.core.expr import Integer

        return Integer(0)


# ------------------------------------------------------------------ #
# The two promoted wedge rules                                        #
# ------------------------------------------------------------------ #


class WedgeGradedOrderDefinition(Definition):
    """Canonical graded order of wedge factors (sound only with
    concrete degrees — the rule stays inert otherwise)."""

    name = "wedge graded commutativity: canonical order (concrete degrees)"
    anchor = Wedge

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._r = registry

    def _sorted(self, e: Wedge):
        from jacopy.algorithms.normalize_alternating import _sort_key

        items = []
        for c in e.children:
            k = _wedge_degree(c, self._r)
            if k is None:
                return None
            items.append((c, k))
        # the SAME key as the simplifier's odd-degree sort
        # (normalize_alternating._sort_key): two canonical orders on
        # the same wedge would rewrite each other forever (2026-09-09
        # non-convergence catch on the exceptional 5-form Jacobi).
        target = sorted(items, key=lambda it: _sort_key(it[0]))
        if [c for c, _ in target] == list(e.children):
            return None
        cur, sign = items[:], 0
        for pos in range(len(target)):
            j = next(i for i in range(pos, len(cur)) if cur[i] == target[pos])
            while j > pos:
                # adjacent swap of degrees a, b costs (−1)^{ab}
                sign ^= (cur[j][1] * cur[j - 1][1]) & 1
                cur[j], cur[j - 1] = cur[j - 1], cur[j]
                j -= 1
        return sign, [c for c, _ in cur]

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Wedge) and self._sorted(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        sign, kids = self._sorted(expr)
        out = Wedge(*kids)
        return Neg(out) if sign else out


class WedgeScalarFactorDefinition(Definition):
    """``(f·α) ∧ β → f·(α ∧ β)`` — scalar factors of a wedge factor
    pull out in front (scalars are central)."""

    name = "wedge factor scalar pull-out: (f·α) ∧ β = f·(α ∧ β)"
    anchor = Wedge

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._r = registry

    def _site(self, e: Wedge):
        from jacopy.central.calculus.scalars import is_scalar_function

        for i, c in enumerate(e.children):
            if isinstance(c, Product) and any(
                is_scalar_function(k, self._r) for k in c.children
            ):
                return i
        return None

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Wedge) and self._site(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.central.calculus.scalars import is_scalar_function

        i = self._site(expr)
        c = expr.children[i]
        scalars = [k for k in c.children if is_scalar_function(k, self._r)]
        rest = [k for k in c.children if not is_scalar_function(k, self._r)]
        kids = list(expr.children)
        kids[i] = rest[0] if len(rest) == 1 else Product(*rest)
        return Product(*scalars, Wedge(*kids))


# ------------------------------------------------------------------ #
# Scanning                                                            #
# ------------------------------------------------------------------ #


def _walk(expr: Expr, seen: Set[int], out: List[Expr]) -> None:
    if id(expr) in seen:
        return
    seen.add(id(expr))
    out.append(expr)
    for c in expr.children:
        _walk(c, seen, out)
    for s in getattr(expr, "rewritable_slots", ()) or ():
        if isinstance(s, Expr):
            _walk(s, seen, out)


class Scan:
    """What occurs in a set of expressions."""

    def __init__(self, exprs: Iterable[Expr]) -> None:
        from jacopy.algebra.lie_bracket_vf import LieBracketVF
        from jacopy.core.indexed_sum import IndexedSum
        from jacopy.core.multi_eval import MultiEval
        from jacopy.algebra.derivation import Act
        from jacopy.central.calculus.bracket_calculus import (
            ExteriorDerivative,
            LieDerivative,
        )
        from jacopy.central.objects.interior import Interior
        from jacopy.central.objects.multivector_interior import (
            MultivectorInterior,
        )
        from jacopy.packages.poisson.nambu import NambuSharpVF

        nodes: List[Expr] = []
        seen: Set[int] = set()
        for e in exprs:
            _walk(e, seen, nodes)
        self.families: Set[str] = set()
        self.sharp_pis = []
        self.interior_multivectors = []
        for n in nodes:
            for cls, tag in (
                (Act, "act"),
                (MultiEval, "multi_eval"),
                (Wedge, "wedge"),
                (IndexedSum, "indexed_sum"),
                (LieDerivative, "lie_derivative"),
                (ExteriorDerivative, "exterior_derivative"),
                (Interior, "interior"),
                (LieBracketVF, "lie_bracket"),
                (MultivectorInterior, "multivector_interior"),
                (NambuSharpVF, "nambu_sharp"),
            ):
                if isinstance(n, cls):
                    self.families.add(tag)
            if isinstance(n, NambuSharpVF) and n.pi not in self.sharp_pis:
                self.sharp_pis.append(n.pi)
            if isinstance(n, MultivectorInterior):
                mv = n.multivector
                if mv not in self.interior_multivectors:
                    self.interior_multivectors.append(mv)

    def __repr__(self) -> str:
        return (
            f"Scan(families={sorted(self.families)}, "
            f"sharps={[p._repr_inner() for p in self.sharp_pis]})"
        )


# ------------------------------------------------------------------ #
# Assembly                                                            #
# ------------------------------------------------------------------ #


def _has(engine: ExpansionEngine, cls, key=None) -> bool:
    for d in engine.definitions:
        if type(d) is cls:
            if key is None:
                return True
            if getattr(d, "_N", None) is not None and getattr(d._N, "pi", None) == key:
                return True
    return False


def assemble_engine(
    *exprs: Expr,
    registry: Optional[PropertyRegistry] = None,
    structures: Sequence = (),
    declare_fi: bool = False,
    extra: Sequence[Definition] = (),
) -> ExpansionEngine:
    """Build an engine for ``exprs``; :func:`assembly_report` returns
    its one-line explanations of what was detected and added.

    ``structures`` are the Nambu/Poisson structures whose sharps may
    occur (the first one hosts the tilde-calculus rules, the others
    get their linearity rules); ``declare_fi=True`` DECLARES the
    fundamental identity (Poisson condition) of the first structure —
    an explicit assumption, recorded in the report."""
    from jacopy.central.calculus import (
        ActExpansionDefinition,
        MultiEvalArgLinearityDefinition,
        OperatorSlotAdditivityDefinition,
    )
    from jacopy.central.objects.multivector_interior import (
        MultivectorInteriorDecomposableDefinition,
        MultivectorInteriorDecomposableFormDefinition,
        MultivectorInteriorLinearityDefinition,
    )
    from jacopy.central.tangent.engine import tangent_engine
    from jacopy.central.tangent.lie_bracket import (
        LieBracketLeibnizDefinition,
    )
    from jacopy.packages.drinfeld.roytenberg import (
        SharpSlotWedgeNormalizeDefinition,
    )
    from jacopy.packages.drinfeld.tilde_calculus import _tilde_engine
    from jacopy.packages.drinfeld.twist import (
        FormProductToWedgeDefinition,
        WedgeSumScalarDefinition,
    )
    from jacopy.packages.poisson.nambu import (
        NambuSharpLinearityDefinition,
    )

    scan = Scan(exprs)
    report: List[str] = [f"scanned {len(exprs)} expression(s): {scan}"]
    known = {s.pi: s for s in structures}
    unknown = [p for p in scan.sharp_pis if p not in known]
    if unknown:
        raise ValueError(
            "sharp maps of undeclared structure(s) "
            f"{[p._repr_inner() for p in unknown]}: pass them via "
            "structures=(...) — the engine will not guess their rules"
        )

    if structures:
        host = structures[0]
        eng = _tilde_engine(host, registry, declare_fi=declare_fi)
        report.append(
            f"base: tilde-calculus engine hosted by {host.pi._repr_inner()} "
            f"(Cartan calculus + Koszul/tilde rules)"
            + (
                " — WITH the DECLARED fundamental identity (Poisson condition)"
                if declare_fi
                else " — no fundamental-identity declaration"
            )
        )
        for s in structures[1:]:
            if not _has(eng, NambuSharpLinearityDefinition, s.pi):
                eng.register(NambuSharpLinearityDefinition(s, registry))
                report.append(
                    f"+ linearity of the sharp map of {s.pi._repr_inner()}"
                )
    else:
        eng = tangent_engine(registry=registry)
        report.append("base: tangent (Cartan) engine")

    def add(rule: Definition, why: str) -> None:
        if _has(eng, type(rule)):
            return
        eng.register(rule)
        report.append(f"+ {rule.name}  [{why}]")

    fam = scan.families
    add(ZeroProductDefinition(), "always: zero products inside operator slots")
    if "multivector_interior" in fam:
        add(MultivectorInteriorLinearityDefinition(registry), "multivector interiors present")
        add(MultivectorInteriorDecomposableDefinition(), "multivector interiors present")
        add(MultivectorInteriorDecomposableFormDefinition(registry), "multivector interiors present")
    if "wedge" in fam:
        add(WedgeSumScalarDefinition(registry), "wedges present")
        add(FormProductToWedgeDefinition(registry), "wedges present")
        add(WedgeGradedOrderDefinition(registry), "wedges present")
        add(WedgeScalarFactorDefinition(registry), "wedges present")
        if "nambu_sharp" in fam:
            add(SharpSlotWedgeNormalizeDefinition(registry), "wedges inside sharp slots")
    if "nambu_sharp" in fam:
        # evaluations θ(·,·) are BORN inside sharp slots during
        # expansion (not visible in the scanned input), so the
        # slot-reachable alternation rule rides with every sharp
        from jacopy.packages.generalized.r_twisted import (
            SharpSlotAlternatingNormalizeDefinition,
        )

        add(
            SharpSlotAlternatingNormalizeDefinition(registry),
            "sharp slots present (alternation must reach inside them)",
        )
    if "act" in fam:
        add(ActExpansionDefinition(registry), "operator actions present")
    if "multi_eval" in fam:
        add(MultiEvalArgLinearityDefinition(registry), "multi-slot evaluations present")
    if fam & {"lie_derivative", "interior", "nambu_sharp"}:
        add(OperatorSlotAdditivityDefinition(), "operators with additive direction slots")
    if "lie_bracket" in fam:
        add(LieBracketLeibnizDefinition(registry), "Lie brackets of vector fields present")
    for rule in extra:
        add(rule, "requested by the caller")
    eng.assembly_report = report
    return eng


#: Legacy name — the report now lives on the engine itself
#: (``ExpansionEngine.assembly_report``), so nothing is registered here
#: and no engine is kept alive by this module (2026-09-09 audit, F4).
_REPORTS: dict = {}


def assembly_report(engine: ExpansionEngine) -> List[str]:
    """The assembly report of an engine built by
    :func:`assemble_engine` (empty for other engines)."""
    return list(getattr(engine, "assembly_report", None) or [])
