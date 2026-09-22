r"""
LaTeX renderer for :class:`~jacopy.core.expr.Expr` trees,
:class:`~jacopy.proof.step.ProofStep` / :class:`~jacopy.proof.chain.ProofChain`
transcripts and :class:`~jacopy.proof.theorems.Theorem` records
(PDF item 1, "integrated with LaTeX"; ported from v2's dispatch-based
display layer, 2026-09-22).

The output is raw LaTeX math, no ``$…$`` delimiters, so the caller
chooses the surrounding environment. :func:`chain_to_latex` wraps a
chain in a ``gather*`` body; individual expressions come out as
snippets ready to splice into equations or ``\text{…}`` arguments.

Dispatch is MRO-based: the most specific registered class for
``type(expr)`` wins, so :class:`~jacopy.algebra.derivation.Derivation`
subclasses without a dedicated handler fall through to the generic
name-based one. Handlers for the domain packages (Poisson, metric-
affine, algebroid, generalized) are registered lazily on first use,
so ``import jacopy.display`` stays cheap and the display layer never
forces the package import graph.

Name sanitising (:func:`latex_name`) translates the Unicode glyphs
the code base uses in operator names (``ι_X``, ``ω₂``, ``ℒ̃``, ``⊛``,
``Π₃``) into standard LaTeX commands and braces multi-character and
nested subscripts, so the result is pdfLaTeX-safe.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Type

from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    Power,
    Product,
    Rational,
    Sum,
    Symbol,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.theorems import Theorem


# --------------------------------------------------------------------- #
# Name sanitising                                                        #
# --------------------------------------------------------------------- #


_UNICODE_TO_LATEX: Dict[str, str] = {
    # lowercase Greek
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # uppercase Greek
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    # script / blackboard letters used as operator names
    "ℒ": r"\mathcal{L}", "𝒦": r"\mathcal{K}", "𝒟": r"\mathcal{D}",
    "𝒫": r"\mathcal{P}", "ℝ": r"\mathbb{R}", "ħ": r"\hbar",
    # musical / algebraic
    "♭": r"\flat", "♯": r"\sharp",
    "∧": r"\wedge", "∨": r"\vee", "⋆": r"\star", "⊛": r"\circledast",
    "∘": r"\circ", "⊕": r"\oplus", "⊗": r"\otimes",
    "·": r"\cdot", "∇": r"\nabla", "∂": r"\partial",
    "⟨": r"\langle", "⟩": r"\rangle",
    "∞": r"\infty", "∈": r"\in",
    "′": "'",
    # arrows / ellipsis / Unicode minus that surface in rule names
    "−": "-",
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "↦": r"\mapsto",
    "⇒": r"\Rightarrow",
    "⟹": r"\Longrightarrow",
    "⟺": r"\Longleftrightarrow",
    "⇔": r"\Leftrightarrow",
    "…": r"\dots", "⋯": r"\cdots",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "±": r"\pm", "×": r"\times", "÷": r"\div",
}

_SUBSCRIPT_DIGITS = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
_SUPERSCRIPT_DIGITS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
_SUB_RUN = re.compile("[₀₁₂₃₄₅₆₇₈₉]+")
_SUP_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")

_MULTICHAR_SUB = re.compile(r"_(\w{2,})")

_COMBINING_TILDE_CHAR = "\u0303"
_COMBINING_DOT_CHAR = "\u0307"
_COMBINING_HAT_CHAR = "\u0302"
_MATH_COMBINING_TILDE_RE = re.compile(rf"(.)({_COMBINING_TILDE_CHAR})")
_MATH_COMBINING_DOT_RE = re.compile(rf"(.)({_COMBINING_DOT_CHAR})")
_MATH_COMBINING_HAT_RE = re.compile(rf"(.)({_COMBINING_HAT_CHAR})")


def _brace_nested_subscripts(name: str) -> str:
    """Wrap nested subscripts so pdfLaTeX doesn't see ``Base_a_b``:
    ``K_X_ι_U(μ)`` → ``K_{X_{ι_U(μ)}}``."""
    depth = 0
    positions: list[int] = []
    for i, c in enumerate(name):
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        elif c == "_" and depth == 0:
            positions.append(i)
    if len(positions) < 2:
        return name
    first = positions[0]
    inner = _brace_nested_subscripts(name[first + 1 :])
    return name[: first + 1] + "{" + inner + "}"


def _scripts(name: str) -> str:
    """Unicode sub/superscript digit runs → ``_{…}`` / ``^{…}``."""
    name = _SUB_RUN.sub(
        lambda m: "_{" + "".join(_SUBSCRIPT_DIGITS[c] for c in m.group(0)) + "}",
        name,
    )
    name = _SUP_RUN.sub(
        lambda m: "^{"
        + "".join(
            "-" if c == "⁻" else _SUPERSCRIPT_DIGITS[c] for c in m.group(0)
        )
        + "}",
        name,
    )
    return name


def latex_name(name: str) -> str:
    """Translate a Derivation/Symbol display name into a LaTeX snippet.

    Collapses combining diacritics (``K̃`` → ``\\tilde{K}``, ``Π̂`` →
    ``\\hat{\\Pi}``), converts Unicode sub/superscript digits
    (``ω₂`` → ``\\omega_{2}``, ``Ψ⁻¹`` → ``\\Psi^{-1}``), braces nested
    subscripts, replaces Unicode math glyphs with LaTeX commands, and
    braces multi-character subscripts (``X_ab`` → ``X_{ab}``).
    """
    if not isinstance(name, str):
        raise TypeError("latex_name: expected a str")
    name = _MATH_COMBINING_TILDE_RE.sub(
        lambda m: "\\tilde{" + m.group(1) + "}", name
    )
    name = _MATH_COMBINING_DOT_RE.sub(
        lambda m: "\\dot{" + m.group(1) + "}", name
    )
    name = _MATH_COMBINING_HAT_RE.sub(
        lambda m: "\\hat{" + m.group(1) + "}", name
    )
    name = _scripts(name)
    name = _brace_nested_subscripts(name)
    name = _glyphs(name)
    name = _MULTICHAR_SUB.sub(lambda m: "_{" + m.group(1) + "}", name)
    return name


def _glyphs(name: str) -> str:
    """Replace Unicode math glyphs by LaTeX commands, inserting a
    separating space only where a command would otherwise swallow a
    following letter (``\\omega X`` yes, ``\\omega}`` / ``\\omega_`` no)."""
    for glyph, cmd in _UNICODE_TO_LATEX.items():
        if glyph not in name:
            continue
        spaced = cmd[-1].isalpha()
        if spaced:
            name = re.sub(re.escape(glyph) + r"(?=[A-Za-z])", lambda m, c=cmd: c + " ", name)
        name = name.replace(glyph, cmd)
    return name


# --------------------------------------------------------------------- #
# Dispatch                                                              #
# --------------------------------------------------------------------- #


_P_ATOM = 100
_P_CALL = 90
_P_POWER = 80
_P_PRODUCT = 60
_P_WEDGE = 55
_P_NEG = 50
_P_SUM = 40


Handler = Callable[[Expr, int], str]
_HANDLERS: Dict[Type[Expr], Handler] = {}
_PACKAGE_HANDLERS_LOADED = False


def register(cls: Type[Expr]):
    """Decorator: register a LaTeX handler ``fn(expr, ctx_precedence)``
    for ``cls`` (public, so user-defined nodes can plug in)."""

    def decorator(fn: Handler) -> Handler:
        _HANDLERS[cls] = fn
        return fn

    return decorator


_register = register


def to_latex(expr: Expr, ctx_precedence: int = 0) -> str:
    """Render ``expr`` as a LaTeX math snippet (no ``$`` delimiters)."""
    if not isinstance(expr, Expr):
        raise TypeError("to_latex: expected an Expr")
    _load_package_handlers()
    for cls in type(expr).__mro__:
        h = _HANDLERS.get(cls)
        if h is not None:
            return h(expr, ctx_precedence)
    return latex_name(expr._repr_inner())


def _wrap(text: str, own_prec: int, ctx_prec: int) -> str:
    if own_prec < ctx_prec:
        return f"\\left({text}\\right)"
    return text


def _call(head: str, *args: Expr) -> str:
    arglist = ",\\, ".join(to_latex(a, 0) for a in args)
    return f"{head}\\!\\left({arglist}\\right)"


# --------------------------------------------------------------------- #
# Core expression types                                                 #
# --------------------------------------------------------------------- #


@_register(Symbol)
def _sym(expr: Symbol, _ctx: int) -> str:
    return latex_name(expr.name)


@_register(Integer)
def _int(expr: Integer, ctx: int) -> str:
    v = expr.value
    if v < 0:
        return _wrap(str(v), _P_NEG, ctx)
    return str(v)


@_register(Rational)
def _rat(expr: Rational, ctx: int) -> str:
    p, q = expr.p, expr.q
    if p < 0:
        return _wrap(f"-\\frac{{{-p}}}{{{q}}}", _P_NEG, ctx)
    return f"\\frac{{{p}}}{{{q}}}"


@_register(Neg)
def _neg(expr: Neg, ctx: int) -> str:
    inner = to_latex(expr.arg, _P_NEG + 1)
    return _wrap(f"-{inner}", _P_NEG, ctx)


@_register(Sum)
def _sum(expr: Sum, ctx: int) -> str:
    parts: list[str] = []
    for i, child in enumerate(expr.children):
        if isinstance(child, Neg):
            inner = to_latex(child.arg, _P_NEG + 1)
            parts.append(("- " if i > 0 else "-") + inner)
        else:
            rendered = to_latex(child, _P_SUM + 1)
            parts.append(("+ " if i > 0 else "") + rendered)
    text = " ".join(parts) if len(parts) > 1 else parts[0] if parts else "0"
    return _wrap(text, _P_SUM, ctx)


@_register(Product)
def _prod(expr: Product, ctx: int) -> str:
    parts = [to_latex(c, _P_PRODUCT + 1) for c in expr.children]
    text = " \\, ".join(parts) if parts else "1"
    return _wrap(text, _P_PRODUCT, ctx)


@_register(Power)
def _pow(expr: Power, ctx: int) -> str:
    base = to_latex(expr.base, _P_POWER + 1)
    exp = to_latex(expr.exp, 0)
    return _wrap(f"{{{base}}}^{{{exp}}}", _P_POWER, ctx)


def _load_core_handlers() -> None:
    from jacopy.algebra.commutator import Commutator
    from jacopy.algebra.derivation import Act, Derivation
    from jacopy.algebra.lie_bracket_vf import LieBracketVF
    from jacopy.brackets.base import BracketApply
    from jacopy.core.hodge import HodgeStar
    from jacopy.core.indexed_sum import IndexedSum
    from jacopy.core.multi_eval import MultiEval
    from jacopy.core.pairing import Pairing
    from jacopy.core.symmetrize import Antisymmetrization, Symmetrization
    from jacopy.core.tensor_product import TensorProduct
    from jacopy.core.wedge import Wedge
    from jacopy.core.wildcards import SeqWildcard, Wildcard

    @_register(Wedge)
    def _wedge(expr, ctx):
        parts = [to_latex(c, _P_WEDGE + 1) for c in expr.children]
        return _wrap(" \\wedge ".join(parts), _P_WEDGE, ctx)

    @_register(TensorProduct)
    def _tp(expr, ctx):
        parts = [to_latex(c, _P_WEDGE + 1) for c in expr.children]
        return _wrap(" \\otimes ".join(parts), _P_WEDGE, ctx)

    @_register(Symmetrization)
    def _symm(expr, _ctx):
        return _call("\\operatorname{Sym}", expr.arg)

    @_register(Antisymmetrization)
    def _alt(expr, _ctx):
        return _call("\\operatorname{Alt}", expr.arg)

    @_register(HodgeStar)
    def _hodge(expr, ctx):
        star = "\\star"
        if expr.metric is not None:
            star = f"\\star_{{{to_latex(expr.metric, 0)}}}"
        return _wrap(f"{star} {to_latex(expr.arg, _P_CALL)}", _P_CALL, ctx)

    @_register(IndexedSum)
    def _isum(expr, ctx):
        dummy = to_latex(expr.dummy, 0)
        rng = getattr(expr.range_, "name", None)
        sub = f"{dummy} \\in {latex_name(rng)}" if rng else dummy
        body = to_latex(expr.body, _P_SUM + 1)
        return _wrap(f"\\sum_{{{sub}}} {body}", _P_SUM, ctx)

    @_register(Pairing)
    def _pairing(expr, _ctx):
        return f"\\langle {to_latex(expr.alpha, 0)},\\, {to_latex(expr.X, 0)} \\rangle"

    @_register(MultiEval)
    def _multi_eval(expr, ctx):
        head = to_latex(expr.head, _P_CALL + 1)
        return _wrap(_call(head, *expr.args), _P_CALL, ctx)

    @_register(Wildcard)
    def _wild(expr, _ctx):
        return f"\\mathord{{?}}{latex_name(expr.name)}"

    @_register(SeqWildcard)
    def _seqwild(expr, _ctx):
        return f"\\mathord{{?\\ast}}{latex_name(expr.name)}"

    @_register(Derivation)
    def _deriv(expr, _ctx):
        return latex_name(expr.name)

    @_register(Act)
    def _act(expr, ctx):
        op = to_latex(expr.op, _P_CALL + 1)
        return _wrap(_call(op, expr.arg), _P_CALL, ctx)

    @_register(Commutator)
    def _comm(expr, _ctx):
        return f"\\left[{to_latex(expr.a, 0)},\\, {to_latex(expr.b, 0)}\\right]"

    @_register(LieBracketVF)
    def _lie_vf(expr, _ctx):
        return f"\\left[{to_latex(expr.X, 0)},\\, {to_latex(expr.Y, 0)}\\right]"

    @_register(BracketApply)
    def _bracket_apply(expr, _ctx):
        tag = latex_name(expr.bracket.name)
        return (
            f"\\left[{to_latex(expr.a, 0)},\\, {to_latex(expr.b, 0)}\\right]"
            f"_{{{tag}}}"
        )


def _load_central_handlers() -> None:
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
    from jacopy.central.objects.connection import CovariantOp
    from jacopy.central.objects.endomorphism import EndoForm, EndoVF
    from jacopy.central.objects.frame import (
        CoframeField,
        FrameField,
        KroneckerDelta,
    )
    from jacopy.central.objects.interior import Interior
    from jacopy.central.objects.metric import InverseMetric, Metric
    from jacopy.central.objects.multivector_interior import (
        MultivectorInterior,
    )
    from jacopy.central.objects.musical import Flat, Sharp
    from jacopy.central.objects.tilde_interior import TildeInterior
    from jacopy.central.tangent.anholonomy import AnholonomyCoefficient

    def _index(s) -> str:
        inner = getattr(s, "_repr_inner", None)
        return latex_name(inner() if callable(inner) else str(s))

    @_register(Interior)
    def _iota(expr, _ctx):
        return f"\\iota_{{{to_latex(expr.vector, 0)}}}"

    @_register(TildeInterior)
    def _tilde_iota(expr, _ctx):
        return f"\\tilde{{\\iota}}_{{{to_latex(expr.form, 0)}}}"

    @_register(MultivectorInterior)
    def _mv_iota(expr, _ctx):
        return f"\\iota_{{{to_latex(expr.multivector, 0)}}}"

    @_register(LieDerivative)
    def _lie(expr, _ctx):
        op_name = getattr(expr, "_op_name", "L")
        head = "\\tilde{\\mathcal{L}}" if _COMBINING_TILDE_CHAR in op_name else "\\mathcal{L}"
        if op_name not in ("L", "L" + _COMBINING_TILDE_CHAR, "ℒ", "ℒ" + _COMBINING_TILDE_CHAR):
            head = latex_name(op_name)
        return f"{head}_{{{to_latex(expr.vector, 0)}}}"

    @_register(ExteriorDerivative)
    def _d(expr, _ctx):
        return latex_name(expr.name)

    @_register(CovariantOp)
    def _nabla(expr, _ctx):
        head = latex_name(expr.connection_name) if expr.connection_name != "∇" else "\\nabla"
        return f"{head}_{{{to_latex(expr.vector, 0)}}}"

    @_register(EndoVF)
    def _endo_vf(expr, _ctx):
        head = latex_name(expr.endo_name) + ("^{-1}" if expr.inverted else "")
        return _call(head, expr.arg)

    @_register(EndoForm)
    def _endo_form(expr, _ctx):
        head = "\\tilde{" + latex_name(expr.endo_name) + "}" + ("^{-1}" if expr.inverted else "")
        return _call(head, expr.arg)

    @_register(FrameField)
    def _frame(expr, _ctx):
        return f"{latex_name(expr.base_name)}_{{{_index(expr.index)}}}"

    @_register(CoframeField)
    def _coframe(expr, _ctx):
        return f"{latex_name(expr.base_name)}^{{{_index(expr.index)}}}"

    @_register(KroneckerDelta)
    def _delta(expr, _ctx):
        return f"\\delta^{{{_index(expr.upper)}}}_{{{_index(expr.lower)}}}"

    @_register(AnholonomyCoefficient)
    def _anhol(expr, _ctx):
        a, b = expr.lower
        head = "\\gamma" if expr.frame_name == "e" else f"\\gamma^{{({latex_name(expr.frame_name)})}}"
        return f"{head}^{{{_index(expr.upper)}}}_{{{_index(a)}{_index(b)}}}"

    @_register(Flat)
    def _flat(expr, _ctx):
        return f"{to_latex(expr.metric, _P_CALL + 1)}^{{\\flat}}"

    @_register(Sharp)
    def _musical_sharp(expr, _ctx):
        return f"{to_latex(expr.bivector, _P_CALL + 1)}^{{\\sharp}}"

    @_register(Metric)
    def _metric(expr, _ctx):
        return latex_name(expr.name)

    @_register(InverseMetric)
    def _inv_metric(expr, _ctx):
        return f"{latex_name(expr.name)}^{{-1}}"

    @_register(AlgebroidBracket)
    def _alg_bracket(expr, _ctx):
        return (
            f"\\left[{to_latex(expr.u, 0)},\\, {to_latex(expr.v, 0)}\\right]"
            f"_{{{latex_name(expr.algebroid_name)}}}"
        )

    @_register(AnchoredVF)
    def _anchor(expr, _ctx):
        return _call("\\rho", expr.section)

    @_register(CoboundaryForm)
    def _cobound(expr, _ctx):
        return f"d_{{{latex_name(expr.algebroid_name)}}}{to_latex(expr.function, _P_CALL)}"

    @_register(EMetric)
    def _emetric(expr, _ctx):
        return _call(latex_name(expr.metric_name) + f"_{{{latex_name(expr.algebroid_name)}}}", expr.u, expr.v)

    @_register(MetricSharp)
    def _msharp(expr, _ctx):
        return _call(latex_name(expr.metric_name) + "^{-1}", expr.form)

    @_register(LocalityOperator)
    def _locality(expr, _ctx):
        return _call("L", expr.form, expr._u, expr._v)

    @_register(Jacobiator)
    def _jac(expr, _ctx):
        return _call(f"J_{{{latex_name(expr.algebroid_name)}}}", *expr.sections)

    @_register(Derivator)
    def _derivator(expr, _ctx):
        return _call(f"D_{{{latex_name(expr.map_name)}}}", *expr.sections)

    @_register(Predator)
    def _predator(expr, _ctx):
        return _call(f"P_{{{latex_name(expr.map_name)}}}", *expr.sections)

    @_register(MappedSection)
    def _mapped(expr, _ctx):
        return _call(latex_name(expr.map_name), expr.section)


def _load_package_handlers() -> None:
    """Register the domain-package handlers once (lazy: keeps
    ``import jacopy.display`` light and tolerates a package that fails
    to import — the generic name handler then applies)."""
    global _PACKAGE_HANDLERS_LOADED
    if _PACKAGE_HANDLERS_LOADED:
        return
    _PACKAGE_HANDLERS_LOADED = True
    _load_core_handlers()
    _load_central_handlers()
    try:
        _load_poisson_handlers()
    except ImportError:  # pragma: no cover
        pass
    try:
        _load_metric_affine_handlers()
    except ImportError:  # pragma: no cover
        pass
    try:
        _load_generalized_handlers()
    except ImportError:  # pragma: no cover
        pass


def _load_poisson_handlers() -> None:
    from jacopy.packages.poisson.core import (
        HamiltonianVF,
        PoissonBracketValue,
        SharpVF,
    )
    from jacopy.packages.poisson.koszul import KoszulBracket
    from jacopy.packages.poisson.nambu import NambuSharpVF
    from jacopy.packages.poisson.symplectic import SymplecticHamiltonianVF
    from jacopy.packages.poisson.tilde import TildeAnholonomyCoefficient

    @_register(SharpVF)
    def _sharp(expr, _ctx):
        return _call(f"{to_latex(expr.pi, _P_CALL + 1)}^{{\\sharp}}", expr.alpha)

    @_register(NambuSharpVF)
    def _nsharp(expr, _ctx):
        return _call(to_latex(expr.pi, _P_CALL + 1), expr.omega)

    @_register(HamiltonianVF)
    def _ham(expr, _ctx):
        return f"X_{{{to_latex(expr.f, 0)}}}"

    @_register(SymplecticHamiltonianVF)
    def _sham(expr, _ctx):
        return f"X_{{{to_latex(expr.f, 0)}}}"

    @_register(PoissonBracketValue)
    def _pbv(expr, _ctx):
        return f"\\{{{to_latex(expr.f, 0)},\\, {to_latex(expr.g, 0)}\\}}"

    @_register(KoszulBracket)
    def _koszul(expr, _ctx):
        return (
            f"\\left[{to_latex(expr.alpha, 0)},\\, {to_latex(expr.beta, 0)}\\right]"
            f"_{{{to_latex(expr.pi, 0)}}}"
        )

    @_register(TildeAnholonomyCoefficient)
    def _tanhol(expr, _ctx):
        up = "".join(latex_name(str(i)) for i in expr.upper)
        return f"\\tilde{{\\gamma}}_{{{latex_name(str(expr.lower))}}}^{{{up}}}"


def _load_metric_affine_handlers() -> None:
    from jacopy.packages.metric_affine.frame_components import (
        ConnectionCoefficient,
        ConnectionForm,
    )
    from jacopy.packages.metric_affine.metric import (
        InverseMetricComponent,
        MetricValue,
        NonMetricity,
    )
    from jacopy.packages.metric_affine.torsion_curvature import (
        Curvature,
        Torsion,
    )

    @_register(MetricValue)
    def _mval(expr, _ctx):
        return _call(latex_name(expr.metric_name), expr.X, expr.Y)

    @_register(InverseMetricComponent)
    def _invcomp(expr, _ctx):
        a, b = expr.upper
        return f"{latex_name(expr.metric_name)}^{{{latex_name(str(a))}{latex_name(str(b))}}}"

    @_register(NonMetricity)
    def _nonmet(expr, _ctx):
        return _call("Q", *expr.arguments)

    @_register(Torsion)
    def _torsion(expr, _ctx):
        return _call("T", *expr.arguments)

    @_register(Curvature)
    def _curv(expr, _ctx):
        X, Y, Z = expr.arguments
        return f"{_call('R', X, Y)}{to_latex(Z, _P_CALL)}"

    @_register(ConnectionCoefficient)
    def _gamma(expr, _ctx):
        a, b = expr.lower
        return (
            f"\\Gamma^{{{latex_name(str(expr.upper))}}}"
            f"_{{{latex_name(str(a))}{latex_name(str(b))}}}"
        )

    @_register(ConnectionForm)
    def _omega_ab(expr, _ctx):
        return (
            f"\\omega^{{{latex_name(str(expr.upper))}}}"
            f"_{{{latex_name(str(expr.lower))}}}"
        )


def _load_generalized_handlers() -> None:
    from jacopy.packages.generalized.courant_axioms import CourantD

    @_register(CourantD)
    def _courant_d(expr, _ctx):
        return f"\\mathcal{{D}}{to_latex(expr.function, _P_CALL)}"


# --------------------------------------------------------------------- #
# Proof transcript                                                      #
# --------------------------------------------------------------------- #


_COMBINING_TILDE_RE = re.compile(rf"(.)({_COMBINING_TILDE_CHAR})")
_COMBINING_DOT_RE = re.compile(rf"(.)({_COMBINING_DOT_CHAR})")
_COMBINING_HAT_RE = re.compile(rf"(.)({_COMBINING_HAT_CHAR})")


def _escape_text(text: str) -> str:
    """Escape LaTeX-special characters and lift Unicode math glyphs
    into ``\\ensuremath{…}`` so a ``\\text{…}`` argument built from a
    rule name or a theorem statement is pdfLaTeX-safe."""
    text = (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("^", r"\^{}")
        .replace("~", r"\textasciitilde{}")
        .replace("<", r"\textless{}")
        .replace(">", r"\textgreater{}")
    )
    text = _COMBINING_TILDE_RE.sub(
        lambda m: f"\\ensuremath{{\\tilde{{{m.group(1)}}}}}", text
    )
    text = _COMBINING_DOT_RE.sub(
        lambda m: f"\\ensuremath{{\\dot{{{m.group(1)}}}}}", text
    )
    text = _COMBINING_HAT_RE.sub(
        lambda m: f"\\ensuremath{{\\hat{{{m.group(1)}}}}}", text
    )
    text = _SUB_RUN.sub(
        lambda m: "\\ensuremath{_{"
        + "".join(_SUBSCRIPT_DIGITS[c] for c in m.group(0))
        + "}}",
        text,
    )
    text = _SUP_RUN.sub(
        lambda m: "\\ensuremath{^{"
        + "".join("-" if c == "⁻" else _SUPERSCRIPT_DIGITS[c] for c in m.group(0))
        + "}}",
        text,
    )
    for glyph, cmd in _UNICODE_TO_LATEX.items():
        if glyph in text:
            text = text.replace(glyph, f"\\ensuremath{{{cmd}}}")
    return text


def step_to_latex(step: ProofStep) -> str:
    r"""Render a single :class:`ProofStep` as ``before \to after
    \quad \text{[rule]}`` (justification appended unless it merely
    repeats the rule)."""
    if not isinstance(step, ProofStep):
        raise TypeError("step_to_latex: expected a ProofStep")
    before = to_latex(step.before)
    after = to_latex(step.after)
    rule = _escape_text(step.rule)
    tag = f"\\,({step.provenance_tag})" if step.provenance_tag else ""
    annotation = f"\\text{{[{rule}]{tag}}}"
    just = step.justification or ""
    just_strip = just.removeprefix("apply axiom: ").removeprefix("apply ")
    if just and just_strip != step.rule:
        annotation = annotation + f"\\;\\text{{--- {_escape_text(just)}}}"
    return f"{before} \\to {after} \\quad {annotation}"


def chain_to_latex(chain: ProofChain, *, max_steps: int | None = None) -> str:
    r"""Render a :class:`ProofChain` as a ``gather*`` block (one
    independent row per top-level step; nested sub-proofs are not
    expanded — see :func:`jacopy.display.jupyter.display_chain_collapsible`
    for the tree view). ``max_steps`` truncates long transcripts with
    an explicit ellipsis row."""
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_latex: expected a ProofChain")
    if len(chain) == 0:
        return "\\begin{gather*}\n\\text{(empty proof chain)}\n\\end{gather*}"
    steps = list(chain.steps)
    rows = [step_to_latex(s) for s in (steps if max_steps is None else steps[:max_steps])]
    if max_steps is not None and len(steps) > max_steps:
        rows.append(f"\\vdots \\quad \\text{{({len(steps) - max_steps} more steps)}}")
    body = " \\\\\n".join(rows)
    return (
        "{\\allowdisplaybreaks\\scriptsize\n"
        f"\\begin{{gather*}}\n{body}\n\\end{{gather*}}\n"
        "}"
    )


def theorem_to_latex(theorem: Theorem, *, with_proof: bool = True, max_steps: int | None = None) -> str:
    r"""Render a :class:`Theorem` as a ``theorem``-style paragraph:
    the statement (escaped text), the proven equation ``lhs = rhs`` in
    display math, the assumptions it depends on, and (optionally) the
    proof transcript."""
    if not isinstance(theorem, Theorem):
        raise TypeError("theorem_to_latex: expected a Theorem")
    lines = [
        f"\\paragraph{{Theorem ({_escape_text(theorem.name)}).}} "
        f"{_escape_text(theorem.statement)}",
        f"\\[ {to_latex(theorem.lhs)} = {to_latex(theorem.rhs)} \\]",
        f"\\noindent\\textit{{Generality:}} {_escape_text(theorem.generality)}.",
    ]
    if theorem.from_axioms:
        items = "".join(f"\\item {_escape_text(a)}\n" for a in theorem.from_axioms)
        lines.append("\\noindent\\textit{Depends on:}\n\\begin{itemize}\n" + items + "\\end{itemize}")
    if theorem.notes:
        lines.append(f"\\noindent\\textit{{Notes:}} {_escape_text(theorem.notes)}")
    if with_proof:
        lines.append("\\noindent\\textit{Proof.}")
        lines.append(chain_to_latex(theorem.proof, max_steps=max_steps))
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# Standalone document export                                             #
# --------------------------------------------------------------------- #


_DEFAULT_PREAMBLE = (
    r"\usepackage{amsmath}" "\n"
    r"\usepackage{amssymb}" "\n"
    r"\usepackage[utf8]{inputenc}" "\n"
)


def _document(body: str, *, title: str, author: str, preamble_extras: str) -> str:
    lines = [r"\documentclass{article}", _DEFAULT_PREAMBLE.rstrip()]
    if preamble_extras:
        lines.append(preamble_extras.rstrip())
    if title:
        lines.append(f"\\title{{{_escape_text(title)}}}")
    if author:
        lines.append(f"\\author{{{_escape_text(author)}}}")
    lines.append(r"\begin{document}")
    if title or author:
        lines.append(r"\maketitle")
    lines.append(body)
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def chain_to_latex_document(
    chain: ProofChain,
    *,
    title: str = "",
    author: str = "",
    preamble_extras: str = "",
    max_steps: int | None = None,
) -> str:
    r"""Wrap :func:`chain_to_latex` in a pdfLaTeX-ready ``article``."""
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_latex_document: expected a ProofChain")
    return _document(
        chain_to_latex(chain, max_steps=max_steps),
        title=title, author=author, preamble_extras=preamble_extras,
    )


def theorem_to_latex_document(
    theorem: Theorem,
    *,
    author: str = "",
    preamble_extras: str = "",
    with_proof: bool = True,
    max_steps: int | None = None,
) -> str:
    r"""Wrap :func:`theorem_to_latex` in a pdfLaTeX-ready ``article``
    titled by the theorem's name."""
    if not isinstance(theorem, Theorem):
        raise TypeError("theorem_to_latex_document: expected a Theorem")
    return _document(
        theorem_to_latex(theorem, with_proof=with_proof, max_steps=max_steps),
        title=theorem.name, author=author, preamble_extras=preamble_extras,
    )


# --------------------------------------------------------------------- #
# TikZ chain diagram                                                     #
# --------------------------------------------------------------------- #


def chain_to_tikz(chain: ProofChain, *, node_distance: str = "1.2cm") -> str:
    r"""Render a :class:`ProofChain` as a vertical TikZ diagram: boxed
    expression nodes joined by arrows labelled with the rule names."""
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_tikz: expected a ProofChain")
    if len(chain) == 0:
        return (
            f"\\begin{{tikzpicture}}[node distance={node_distance}]\n"
            "\\node {(empty proof chain)};\n"
            "\\end{tikzpicture}"
        )
    lines = [f"\\begin{{tikzpicture}}[node distance={node_distance}]"]
    lines.append(f"\\node[draw, rectangle] (e0) {{${to_latex(chain.steps[0].before)}$}};")
    for i, step in enumerate(chain.steps):
        lines.append(
            f"\\node[draw, rectangle, below=of e{i}] (e{i + 1}) {{${to_latex(step.after)}$}};"
        )
    for i, step in enumerate(chain.steps):
        tag = f" ({step.provenance_tag})" if step.provenance_tag else ""
        rule = _escape_text(f"{step.rule}{tag}")
        lines.append(f"\\draw[->] (e{i}) -- node[right] {{\\small {rule}}} (e{i + 1});")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def chain_to_tikz_document(
    chain: ProofChain,
    *,
    title: str = "",
    author: str = "",
    node_distance: str = "1.2cm",
) -> str:
    r"""Standalone ``article`` wrapper around :func:`chain_to_tikz`."""
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_tikz_document: expected a ProofChain")
    body = "\\begin{center}\n" + chain_to_tikz(chain, node_distance=node_distance) + "\n\\end{center}"
    return _document(
        body,
        title=title,
        author=author,
        preamble_extras="\\usepackage{tikz}\n\\usetikzlibrary{positioning}",
    )
