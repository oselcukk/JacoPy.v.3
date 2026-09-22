"""Display coverage over EVERY ``Expr`` subclass in the package (ledger
K1.a, 2026-09-22): each concrete class needs a construction recipe
here; a new node class without one fails this test, so the renderer's
coverage cannot silently rot. Every instance must render to LaTeX and
ASCII without raising and without leaking raw Unicode."""

from __future__ import annotations

import importlib
import pkgutil
import re

import pytest

import jacopy
from jacopy.core.expr import Expr

_NON_ASCII = re.compile(r"[^\x00-\x7f]")

#: Internal helper nodes that are never user-visible.
_INTERNAL = {"_BoundSumToken", "_PropertyMarker", "_SlotAtom", "Atom", "Derivation"}


def _all_expr_subclasses():
    for m in pkgutil.walk_packages(jacopy.__path__, "jacopy."):
        importlib.import_module(m.name)
    seen, out = set(), []

    def walk(c):
        for s in c.__subclasses__():
            if s not in seen:
                seen.add(s)
                # only the library's own nodes (test files define
                # throwaway Expr subclasses of their own)
                if s.__module__.startswith("jacopy."):
                    out.append(s)
                walk(s)

    walk(Expr)
    return out


def _recipes():
    from jacopy.algebra.commutator import Commutator
    from jacopy.algebra.derivation import Act
    from jacopy.algebra.lie_bracket_vf import LieBracketVF
    from jacopy.brackets.base import BracketApply
    from jacopy.central.algebroid import algebroid
    from jacopy.central.algebroid.context import (
        AlgebroidBracket,
        AnchoredVF,
        CoboundaryForm,
        EMetric,
        LocalityOperator,
        MetricSharp,
    )
    from jacopy.central.algebroid.operators import (
        Derivator,
        Jacobiator,
        MappedSection,
        Predator,
    )
    from jacopy.central.calculus.bracket_calculus import (
        ExteriorDerivative,
        LieDerivative,
    )
    from jacopy.central.objects import (
        Bundle,
        PVector,
        forms,
        frame,
        functions,
        vector_fields,
    )
    from jacopy.central.objects.connection import CovariantOp
    from jacopy.central.objects.endomorphism import EndoForm, EndoVF
    from jacopy.central.objects.frame import FrameIndex, KroneckerDelta
    from jacopy.central.objects.interior import Interior
    from jacopy.central.objects.metric import InverseMetric, Metric
    from jacopy.central.objects.multivector_interior import MultivectorInterior
    from jacopy.central.objects.musical import Flat, Sharp
    from jacopy.central.objects.tensor import Tensor
    from jacopy.central.objects.tilde_interior import TildeInterior
    from jacopy.central.tangent.anholonomy import anholonomy_coefficient
    from jacopy.central.tangent.exterior import CARTAN_TM
    from jacopy.central.tangent.schouten import SN
    from jacopy.core.expr import Integer, Neg, Power, Product, Rational, Sum, Symbol
    from jacopy.core.hodge import HodgeStar
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.core.multi_eval import MultiEval
    from jacopy.core.pairing import Pairing
    from jacopy.core.registry import PropertyRegistry
    from jacopy.core.symmetrize import Antisymmetrization, Symmetrization
    from jacopy.core.tensor_product import TensorProduct
    from jacopy.core.wedge import Wedge
    from jacopy.core.wildcards import SeqWildcard, Wildcard
    from jacopy.packages.generalized.bourbaki_exact import ChiSec, PhiSec
    from jacopy.packages.generalized.bourbaki_precalculus import (
        BD,
        BIota,
        BLieR,
        BLieZ,
        BSymbol,
    )
    from jacopy.packages.generalized.bourbaki_structure import (
        BDop,
        BLieRE,
        BMetric,
        BSymbolL,
    )
    from jacopy.packages.generalized.courant_axioms import CourantD
    from jacopy.packages.generalized.general_twist import PsiInvSec, PsiSec
    from jacopy.packages.metric_affine.frame_components import (
        ConnectionCoefficient,
        ConnectionForm,
    )
    from jacopy.packages.metric_affine.metric import (
        InverseMetricComponent,
        MetricValue,
        NonMetricity,
    )
    from jacopy.packages.metric_affine.torsion_curvature import Curvature, Torsion
    from jacopy.packages.poisson.core import (
        HamiltonianVF,
        PoissonBracketValue,
        SharpVF,
        poisson_structure,
    )
    from jacopy.packages.poisson.koszul import KoszulBracket
    from jacopy.packages.poisson.nambu import nambu_structure
    from jacopy.packages.poisson.symplectic import SymplecticHamiltonianVF
    from jacopy.packages.poisson.tilde import TildeAnholonomyCoefficient

    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    X, Y, Z = vector_fields("X Y Z")
    om, et = forms("ω η", degree=1)
    (a2,) = forms("α₂", degree=2)
    fr = frame()
    co = fr.dual()
    P = poisson_structure("π")
    N = nambu_structure("Π₃", p=2)
    E = algebroid("E", Bundle("E"), declare=("lie",))
    u, v, w = E.sections("u v w")

    return {
        "Symbol": Symbol("s"), "Integer": Integer(-3), "Rational": Rational(1, 3),
        "Neg": Neg(f), "Sum": Sum(f, Neg(g)), "Product": Product(f, X),
        "Power": Power(f, Integer(2)), "Wedge": Wedge(om, et),
        "TensorProduct": TensorProduct(om, et), "HodgeStar": HodgeStar(om, 3),
        "IndexedSum": IndexedSum(Symbol("i"), (0, 1), X),
        "MultiEval": MultiEval(a2, X, Y), "Pairing": Pairing(om, X),
        "Symmetrization": Symmetrization(TensorProduct(om, et)),
        "Antisymmetrization": Antisymmetrization(TensorProduct(om, et)),
        "Wildcard": Wildcard("w"), "SeqWildcard": SeqWildcard("ws"),
        "Act": Act(X, f), "Commutator": Commutator(X, Y),
        "LieBracketVF": LieBracketVF(X, Y),
        "BracketApply": BracketApply(SN, X, Y),
        "VectorField": X, "Form": om, "PVector": PVector("R", degree=3),
        "FrameField": fr.field("a"), "CoframeField": co.field("b"),
        "FrameIndex": FrameIndex("a"),
        "KroneckerDelta": KroneckerDelta("a", "b"),
        "AnholonomyCoefficient": anholonomy_coefficient(fr, "c", "a", "b"),
        "Metric": Metric("g"), "InverseMetric": InverseMetric("g"),
        "Tensor": Tensor("T", upper=1, lower=2),
        "Flat": Flat(Metric("g")), "Sharp": Sharp(P.pi),
        "Interior": Interior(X), "TildeInterior": TildeInterior(om),
        "MultivectorInterior": MultivectorInterior(N.pi),
        "LieDerivative": CARTAN_TM.lie(X),
        "ExteriorDerivative": ExteriorDerivative("Cartan-TM"),
        "CovariantOp": CovariantOp("∇", X),
        "EndoVF": EndoVF("A", X), "EndoForm": EndoForm("A", om, inverted=True),
        "AlgebroidBracket": E.bracket(u, v), "AnchoredVF": E.anchor(u),
        "CoboundaryForm": CoboundaryForm("E", f, Bundle("E")),
        "EMetric": EMetric("E", u, v),
        "LocalityOperator": LocalityOperator("E", om, u, v),
        "MetricSharp": MetricSharp("E", om),
        "Jacobiator": Jacobiator("E", u, v, w),
        "Derivator": Derivator("E", "Φ", u, v), "Predator": Predator("E", "Φ", u, v),
        "MappedSection": MappedSection("Φ", u),
        "SharpVF": SharpVF(P.pi, om), "HamiltonianVF": HamiltonianVF(P.pi, f),
        "PoissonBracketValue": PoissonBracketValue(P.pi, f, g),
        "KoszulBracket": KoszulBracket(P.pi, om, et), "NambuSharpVF": N.sharp_vf(a2),
        "SymplecticHamiltonianVF": SymplecticHamiltonianVF(a2, f),
        "TildeAnholonomyCoefficient": TildeAnholonomyCoefficient(P.pi, "e", "a", "b", "c"),
        "MetricValue": MetricValue("g", X, Y),
        "InverseMetricComponent": InverseMetricComponent("g", "a", "b"),
        "NonMetricity": NonMetricity("∇", "g", X, Y, Z),
        "Torsion": Torsion("∇", X, Y), "Curvature": Curvature("∇", X, Y, Z),
        "ConnectionCoefficient": ConnectionCoefficient("∇", "e", "a", "b", "c"),
        "ConnectionForm": ConnectionForm("∇", "e", "a", "b"),
        "CourantD": CourantD("E", f),
        "BD": BD(om), "BIota": BIota(X, om), "BLieR": BLieR(X, om), "BLieZ": BLieZ(Y, om),
        "BSymbol": BSymbol(f, om), "BDop": BDop(om), "BLieRE": BLieRE(u, om),
        "BMetric": BMetric(u, v), "BSymbolL": BSymbolL(f, om),
        "ChiSec": ChiSec(om), "PhiSec": PhiSec(X), "PsiSec": PsiSec(u), "PsiInvSec": PsiInvSec(u),
    }


def test_every_expr_subclass_has_a_recipe():
    classes = {c.__name__ for c in _all_expr_subclasses()} - _INTERNAL
    recipes = _recipes()
    missing = sorted(classes - set(recipes))
    assert not missing, f"add a display recipe for: {missing}"


@pytest.mark.parametrize("name", sorted(_recipes()))
def test_render_latex_and_ascii(name):
    from jacopy.display import to_ascii, to_latex

    expr = _recipes()[name]
    latex = to_latex(expr)
    assert latex and not _NON_ASCII.search(latex), latex
    assert "\x00" not in latex
    assert to_ascii(expr)


def test_section_renders_as_one_oplus_snippet():
    from jacopy.central.objects import forms, vector_fields
    from jacopy.display import display_expr, section_to_latex
    from jacopy.research import SectionType

    (U,) = vector_fields("U")
    (om2,) = forms("ω₂", degree=2)
    (om5,) = forms("ω₅", degree=5)
    s = SectionType.exceptional().section(U, om2, om5)
    assert section_to_latex(s) == r"U \oplus \omega_{2} \oplus \omega_{5}"
    assert display_expr(s)._repr_latex_() == "$" + section_to_latex(s) + "$"
