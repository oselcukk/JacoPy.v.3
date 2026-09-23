"""The display layer (deferred-items ledger K1, 2026-09-22): LaTeX /
ASCII / Jupyter / terminal renderers over v3 nodes, proof chains and
theorems — including a coverage sweep over representative nodes of
every family (no exception, no raw Unicode leak) and, when
``pdflatex`` is installed, an end-to-end compile of a real theorem."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from jacopy.algebra.derivation import Act
from jacopy.central.objects import forms, frame, functions, vector_fields
from jacopy.central.objects.interior import Interior
from jacopy.central.objects.metric import InverseMetric, Metric
from jacopy.central.objects.multivector_interior import MultivectorInterior
from jacopy.central.objects.musical import Flat, Sharp
from jacopy.central.objects.tensor import Tensor
from jacopy.central.objects.tilde_interior import TildeInterior
from jacopy.central.tangent.anholonomy import anholonomy_coefficient
from jacopy.central.tangent.cartan import L, prove_cartan_magic_on_functions
from jacopy.central.tangent.exterior import d
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.core.expr import Integer, Neg, Power, Product, Rational, Sum, Symbol
from jacopy.core.hodge import HodgeStar
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.pairing import Pairing
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symmetrize import Antisymmetrization, Symmetrization
from jacopy.core.tensor_product import TensorProduct
from jacopy.core.wedge import Wedge
from jacopy.display import (
    LatexDisplay,
    chain_to_ascii,
    chain_to_latex,
    chain_to_latex_document,
    chain_to_tikz,
    chain_to_tikz_document,
    display_chain,
    display_chain_collapsible,
    display_expr,
    display_step,
    display_theorem,
    latex_name,
    register,
    render_chain,
    step_to_latex,
    theorem_to_latex,
    theorem_to_latex_document,
    to_ascii,
    to_latex,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.theorems import Theorem

_NON_ASCII = re.compile(r"[^\x00-\x7f]")


@pytest.fixture()
def cast():
    reg = PropertyRegistry()
    f, g = functions("f g", registry=reg)
    X, Y, Z = vector_fields("X Y Z")
    om, et = forms("ω η", degree=1)
    (a2,) = forms("α₂", degree=2)
    (om5,) = forms("ω₅", degree=5)
    return reg, f, g, X, Y, Z, om, et, a2, om5


# ---- name sanitising ------------------------------------------------ #


class TestLatexName:
    def test_greek_and_scripts(self):
        assert latex_name("ω") == r"\omega"
        assert latex_name("ω₂") == r"\omega_{2}"
        assert latex_name("Ψ⁻¹") == r"\Psi^{-1}"
        assert latex_name("Π₃") == r"\Pi_{3}"

    def test_combining_marks(self):
        assert latex_name("ℒ̃") == r"\tilde{\mathcal{L}}"
        assert latex_name("Π̂") == r"\hat{\Pi}"
        assert latex_name("K̃") == r"\tilde{K}"

    def test_subscripts(self):
        assert latex_name("X_f") == "X_f"
        assert latex_name("X_ab") == "X_{ab}"
        assert latex_name("K_X_ι_U") == r"K_{X_{\iota_U}}"

    def test_operators_and_arrows(self):
        assert latex_name("⊛") == r"\circledast"
        assert latex_name("a → b") == r"a \to b"
        assert latex_name("ι_X") == r"\iota_X"

    def test_rejects_non_str(self):
        with pytest.raises(TypeError):
            latex_name(3)  # type: ignore[arg-type]


# ---- expression rendering -------------------------------------------- #


class TestToLatex:
    def test_core_arithmetic(self, cast):
        reg, f, g, X, Y, Z, om, et, a2, om5 = cast
        assert to_latex(Sum(f, Neg(g))) == "f - g"
        assert to_latex(Neg(Sum(X, Y))) == r"-\left(X + Y\right)"
        assert to_latex(Rational(-1, 2)) == r"-\frac{1}{2}"
        assert to_latex(Power(f, Integer(2))) == "{f}^{2}"
        assert to_latex(Product(f, X)) == r"f \, X"
        assert to_latex(Product(Sum(f, g), X)) == r"\left(f + g\right) \, X"

    def test_calculus_operators(self, cast):
        reg, f, g, X, Y, Z, om, et, a2, om5 = cast
        assert to_latex(L(X, om)) == r"\mathcal{L}_{X}\!\left(\omega\right)"
        assert to_latex(d(om)) == r"d\!\left(\omega\right)"
        assert to_latex(Act(Interior(X), a2)) == r"\iota_{X}\!\left(\alpha_{2}\right)"
        assert to_latex(Act(TildeInterior(om), X)) == r"\tilde{\iota}_{\omega}\!\left(X\right)"
        assert to_latex(lie_bracket(X, Y)) == r"\left[X,\, Y\right]"
        assert to_latex(Wedge(om, et)) == r"\omega \wedge \eta"
        assert to_latex(Pairing(om, X)) == r"\langle \omega,\, X \rangle"
        assert to_latex(MultiEval(a2, X, Y)) == r"\alpha_{2}\!\left(X,\, Y\right)"

    def test_poisson_and_nambu_nodes(self, cast):
        from jacopy.packages.poisson.core import (
            HamiltonianVF,
            PoissonBracketValue,
            SharpVF,
            poisson_structure,
        )
        from jacopy.packages.poisson.koszul import KoszulBracket
        from jacopy.packages.poisson.nambu import nambu_structure

        reg, f, g, X, Y, Z, om, et, a2, om5 = cast
        P = poisson_structure("π")
        N = nambu_structure("Π₃", p=2)
        assert to_latex(SharpVF(P.pi, om)) == r"\pi^{\sharp}\!\left(\omega\right)"
        assert to_latex(N.sharp_vf(a2)) == r"\Pi_{3}\!\left(\alpha_{2}\right)"
        assert to_latex(HamiltonianVF(P.pi, f)) == "X_{f}"
        assert to_latex(PoissonBracketValue(P.pi, f, g)) == r"\{f,\, g\}"
        assert to_latex(KoszulBracket(P.pi, om, et)) == r"\left[\omega,\, \eta\right]_{\pi}"

    def test_frame_and_metric_nodes(self, cast):
        from jacopy.central.objects.frame import KroneckerDelta
        from jacopy.packages.metric_affine.torsion_curvature import Curvature, Torsion

        reg, f, g, X, Y, Z, om, et, a2, om5 = cast
        fr = frame()
        assert to_latex(fr.field("a")) == "e_{a}"
        assert to_latex(fr.dual().field("b")) == "e^{b}"
        assert to_latex(KroneckerDelta("a", "b")) == r"\delta^{a}_{b}"
        assert to_latex(anholonomy_coefficient(fr, "c", "a", "b")) == r"\gamma^{c}_{ab}"
        assert to_latex(Metric("g").inverse()) == "g^{-1}"   # the default name carries the inverse once
        assert to_latex(InverseMetric("h")) == "h"             # a custom name is the user's choice
        assert to_latex(Flat(Metric("g"))) == r"g^{\flat}"
        assert to_latex(Torsion("∇", X, Y)) == r"T\!\left(X,\, Y\right)"
        assert to_latex(Curvature("∇", X, Y, Z)) == r"R\!\left(X,\, Y\right)Z"

    def test_algebroid_nodes(self):
        from jacopy.central.algebroid import algebroid
        from jacopy.central.objects import Bundle

        E = algebroid("E", Bundle("E"), declare=("lie",))
        u, v = E.sections("u v")
        assert to_latex(E.bracket(u, v)) == r"\left[u,\, v\right]_{E}"
        assert to_latex(E.anchor(u)) == r"\rho\!\left(u\right)"

    def test_every_family_renders_ascii_clean(self, cast):
        # Coverage sweep: one representative per node family; the
        # output must be pdfLaTeX-safe (no raw Unicode) and never raise.
        from jacopy.packages.drinfeld.double import nambu_double
        from jacopy.packages.drinfeld.examples import boxtimes, exceptional_courant_bracket
        from jacopy.packages.poisson.nambu import nambu_koszul_bracket, nambu_structure

        reg, f, g, X, Y, Z, om, et, a2, om5 = cast
        N = nambu_structure("Π₃", p=2)
        samples = [
            HodgeStar(om, 3), TensorProduct(om, et),
            Symmetrization(TensorProduct(om, et)), Antisymmetrization(TensorProduct(om, et)),
            IndexedSum(Symbol("i"), (0, 1), X), Tensor("T", upper=1, lower=2),
            Sharp(N.pi), Act(MultivectorInterior(N.pi), a2), boxtimes(N, om5),
            nambu_koszul_bracket(N, a2, a2), nambu_double(N, X, a2, Y, a2)[0],
            exceptional_courant_bracket(X, a2, om5, Y, a2, om5)[2],
        ]
        for s in samples:
            t = to_latex(s)
            assert t and not _NON_ASCII.search(t), (type(s).__name__, t)

    def test_user_node_can_register_a_handler(self):
        from jacopy.core.expr import Atom

        class Gizmo(Atom):
            def _key(self):
                return "gizmo"

            def _repr_inner(self):
                return "gizmo"

        @register(Gizmo)
        def _gizmo(expr, ctx):
            return r"\mathrm{G}"

        assert to_latex(Gizmo()) == r"\mathrm{G}"

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            to_latex("x")  # type: ignore[arg-type]


# ---- ascii ---------------------------------------------------------- #


def test_ascii_sign_normalisation(cast):
    reg, f, g, X, Y, Z, om, et, a2, om5 = cast
    assert to_ascii(Sum(f, Neg(g))) == "f - g"
    assert to_ascii(Product(Sum(f, g), X)) == "(f + g) * X"
    assert to_ascii(Wedge(om, et)) == "ω ∧ η"
    # a fraction is a quotient: parenthesised as a power base / exponent
    # (2026-09-23 audit, F5: "3/2**2" reads as 3/4)
    assert to_ascii(Power(Rational(3, 2), Integer(2))) == "(3/2)**2"
    assert to_ascii(Power(f, Rational(1, 2))) == "f**(1/2)"
    assert eval(to_ascii(Power(Rational(3, 2), Integer(2))), {"__builtins__": {}}, {}) == 2.25


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_default_inverse_metric_compiles_with_pdflatex():
    # F4: g⁻¹ rendered ``g^{-1}^{-1}`` was a pdflatex "Double superscript".
    from jacopy.central.objects.metric import Metric as _M

    doc = _document_of(to_latex(_M("g").inverse()))
    run = _pdflatex(doc)
    assert run.returncode == 0 and run.pdf_exists, run.stdout[-1500:]


def _document_of(snippet: str) -> str:
    return (
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
        "\\begin{document}\n$" + snippet + "$\n\\end{document}\n"
    )


# ---- chains, steps, theorems ---------------------------------------- #


@pytest.fixture()
def magic_chain():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    (X,) = vector_fields("X")
    return prove_cartan_magic_on_functions(X, f, registry=reg)


def test_step_and_chain_latex(magic_chain):
    s = step_to_latex(magic_chain.steps[0])
    assert r"\to" in s and r"\text{[" in s and "(axiom)" in s
    body = chain_to_latex(magic_chain)
    assert body.startswith("{\\allowdisplaybreaks") and r"\begin{gather*}" in body
    assert not _NON_ASCII.search(body)
    short = chain_to_latex(magic_chain, max_steps=2)
    assert "more steps" in short and short.count(r"\to") == 2


def test_chain_documents_and_tikz(magic_chain):
    doc = chain_to_latex_document(magic_chain, title="Cartan magic", author="jacopy")
    assert doc.startswith(r"\documentclass{article}") and r"\maketitle" in doc
    tikz = chain_to_tikz(magic_chain)
    assert tikz.count(r"\node[") == len(magic_chain) + 1
    assert r"\usetikzlibrary{positioning}" in chain_to_tikz_document(magic_chain)
    assert chain_to_latex(ProofChain()) == "\\begin{gather*}\n\\text{(empty proof chain)}\n\\end{gather*}"


def test_ascii_and_terminal_chain(magic_chain):
    text = chain_to_ascii(magic_chain, verbosity="compact")
    assert text.count("\n") == len(magic_chain) - 1 and "->" not in text
    assert "->" in chain_to_ascii(magic_chain, verbosity="summary")
    assert isinstance(render_chain(magic_chain), str)
    with pytest.raises(ValueError):
        chain_to_ascii(magic_chain, verbosity="loud")


def test_jupyter_wrappers(magic_chain, cast):
    reg, f, g, X, Y, Z, om, et, a2, om5 = cast
    inline = display_expr(L(X, om))
    assert inline._repr_latex_() == "$" + to_latex(L(X, om)) + "$"
    assert "text/html" in inline._repr_mimebundle_()
    assert display_step(magic_chain.steps[0])._repr_latex_().startswith(r"\begin{gather*}")
    block = display_chain(magic_chain)
    assert block.environment and block._repr_latex_() == chain_to_latex(magic_chain)
    tree = display_chain_collapsible(magic_chain, verbosity="summary")._repr_html_()
    assert "jacopy-proof" in tree and r"\(" in tree
    assert LatexDisplay("x") == LatexDisplay("x") and hash(LatexDisplay("x"))
    with pytest.raises(TypeError):
        display_expr("x")  # type: ignore[arg-type]


def _theorem(chain):
    return Theorem(
        name="cartan_magic_on_functions",
        statement="ℒ_X f = ι_X d f + d ι_X f on functions (ι_X f = 0)",
        lhs=chain.initial,
        rhs=chain.final,
        proof=chain,
        from_axioms=("intrinsic L (Cartan-TM)", "interior product definition"),
        notes="Cartan magic formula, degree 0",
    )


def test_theorem_rendering(magic_chain):
    thm = _theorem(magic_chain)
    t = theorem_to_latex(thm, max_steps=3)
    assert t.startswith(r"\paragraph{Theorem (cartan\_magic\_on\_functions).}")
    assert r"\begin{itemize}" in t and r"\textit{Proof.}" in t
    assert not _NON_ASCII.search(t)
    assert r"\textit{Proof.}" not in theorem_to_latex(thm, with_proof=False)
    doc = theorem_to_latex_document(thm)
    assert r"\title{cartan\_magic\_on\_functions}" in doc
    card = display_theorem(thm)._repr_html_()
    assert "jacopy-theorem" in card and "<details" in card


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_theorem_document_compiles_with_pdflatex(magic_chain):
    doc = theorem_to_latex_document(_theorem(magic_chain))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "thm.tex"
        src.write_text(doc, encoding="utf-8")
        run = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
            cwd=tmp, capture_output=True, text=True, timeout=120,
        )
        assert run.returncode == 0, run.stdout[-2000:]
        assert (Path(tmp) / "thm.pdf").exists()


def _pdflatex(doc: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.tex"
        src.write_text(doc, encoding="utf-8")
        run = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
            cwd=tmp, capture_output=True, text=True, timeout=180,
        )
        run.pdf_exists = (Path(tmp) / "doc.pdf").exists()  # type: ignore[attr-defined]
        return run


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_long_glyph_heavy_theorem_compiles_with_pdflatex():
    # K1.b: the exceptional-double theorem (long statement with ′, ₂₅,
    # ℒ̃, Π̂, ⊛, Λ⁵ → TM) plus its full transcript — every glyph path of
    # the text-mode escaper in one document.
    from jacopy.packages.generalized.exceptional_double import (
        prove_r25_is_lie_tilde_equivariance_defect,
    )
    from jacopy.packages.poisson.nambu import nambu_structure

    reg = PropertyRegistry()
    (h,) = functions("h", registry=reg)
    (om2,) = forms("ω₂", degree=2)
    (et5,) = forms("η₅", degree=5)
    N3, N6 = nambu_structure("Π₃", p=2), nambu_structure("Π₆", p=5)
    chain, thm = prove_r25_is_lie_tilde_equivariance_defect(N3, N6, om2, et5, h, registry=reg)
    run = _pdflatex(theorem_to_latex_document(thm, author="jacopy"))
    assert run.returncode == 0 and run.pdf_exists, run.stdout[-2000:]


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_tikz_chain_document_compiles_with_pdflatex(magic_chain):
    # K1.c: the TikZ diagram is compiled, not just line-counted.
    run = _pdflatex(chain_to_tikz_document(magic_chain, title="Cartan magic (TikZ)"))
    assert run.returncode == 0 and run.pdf_exists, run.stdout[-2000:]
