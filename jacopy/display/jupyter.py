r"""
Jupyter / IPython display adapters.

Wraps the LaTeX output of :mod:`jacopy.display.latex` in lightweight
displayables that Jupyter's rich-display machinery picks up through
the duck-typed ``_repr_latex_`` / ``_repr_html_`` / ``_repr_mimebundle_``
protocol. No runtime dependency on IPython.

* :func:`display_expr` — an :class:`Expr` as inline math;
* :func:`display_step` / :func:`display_chain` (alias
  :func:`display_proof`) — a step / a chain as a ``gather*`` block;
* :func:`display_theorem` — a :class:`Theorem` as statement + equation
  + assumptions (+ proof);
* :func:`display_step_collapsible` / :func:`display_chain_collapsible`
  — an HTML ``<details>`` tree so nested sub-proofs fold in-notebook.

The Expr classes are deliberately not monkey-patched with
``_repr_latex_``; explicit wrappers keep the opt-in boundary clean.
"""

from __future__ import annotations

import html as _html

from jacopy.core.expr import Expr
from jacopy.display.ascii import VERBOSITY_MODES
from jacopy.display.latex import (
    chain_to_latex,
    section_to_latex,
    step_to_latex,
    theorem_to_latex,
    to_latex,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.theorems import Theorem


def _check_verbosity(verbosity: str) -> None:
    if verbosity not in VERBOSITY_MODES:
        raise ValueError(
            f"verbosity must be one of {VERBOSITY_MODES}, got {verbosity!r}"
        )


class LatexDisplay:
    """Jupyter-friendly wrapper around a LaTeX string. ``environment``
    marks a self-contained block (``gather*`` …, displayed as-is)
    versus a snippet (wrapped in ``$…$``)."""

    __slots__ = ("_latex", "_environment")

    def __init__(self, latex: str, *, environment: bool = False) -> None:
        if not isinstance(latex, str):
            raise TypeError("LatexDisplay expects a str")
        self._latex = latex
        self._environment = bool(environment)

    @property
    def latex(self) -> str:
        return self._latex

    @property
    def environment(self) -> bool:
        return self._environment

    def _repr_latex_(self) -> str:
        return self._latex if self._environment else f"${self._latex}$"

    def _repr_html_(self) -> str:
        body = self._latex if self._environment else f"\\({self._latex}\\)"
        return f'<div class="jacopy-latex">{body}</div>'

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        bundle = {
            "text/latex": self._repr_latex_(),
            "text/html": self._repr_html_(),
            "text/plain": self._latex,
        }
        if include is not None:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude is not None:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle

    def __str__(self) -> str:
        return self._latex

    def __repr__(self) -> str:
        return self._latex

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LatexDisplay):
            return (
                self._latex == other._latex
                and self._environment == other._environment
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._latex, self._environment))


def display_expr(expr) -> LatexDisplay:
    """Wrap an :class:`Expr` (or a research-layer
    ``GeneralizedSection``) as inline Jupyter math."""
    if isinstance(expr, Expr):
        return LatexDisplay(to_latex(expr))
    from jacopy.research.sections import GeneralizedSection

    if isinstance(expr, GeneralizedSection):
        return LatexDisplay(section_to_latex(expr))
    raise TypeError("display_expr: expected an Expr or a GeneralizedSection")


def display_step(step: ProofStep) -> LatexDisplay:
    """Wrap a single :class:`ProofStep` in a ``gather*`` body."""
    if not isinstance(step, ProofStep):
        raise TypeError("display_step: expected a ProofStep")
    return LatexDisplay(
        f"\\begin{{gather*}}\n{step_to_latex(step)}\n\\end{{gather*}}",
        environment=True,
    )


def display_chain(chain: ProofChain, *, max_steps: int | None = None) -> LatexDisplay:
    """Wrap a :class:`ProofChain` as a ``gather*`` block."""
    if not isinstance(chain, ProofChain):
        raise TypeError("display_chain: expected a ProofChain")
    return LatexDisplay(chain_to_latex(chain, max_steps=max_steps), environment=True)


display_proof = display_chain


class HtmlDisplay:
    """Jupyter-friendly wrapper around a raw HTML payload (MathJax
    ``\\(…\\)`` delimiters inside are typeset by the notebook)."""

    __slots__ = ("_html",)

    def __init__(self, html: str) -> None:
        if not isinstance(html, str):
            raise TypeError("HtmlDisplay expects a str")
        self._html = html

    @property
    def html(self) -> str:
        return self._html

    def _repr_html_(self) -> str:
        return self._html

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict:
        bundle = {"text/html": self._html, "text/plain": self._html}
        if include is not None:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude is not None:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle

    def __str__(self) -> str:
        return self._html

    def __repr__(self) -> str:
        return self._html

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HtmlDisplay):
            return self._html == other._html
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._html)


HtmlProofDisplay = HtmlDisplay  # v2 name


def display_theorem(theorem: Theorem, *, with_proof: bool = True, max_steps: int | None = 40) -> HtmlDisplay:
    """A :class:`Theorem` as an HTML card: statement, the proven
    equation (MathJax), generality, assumptions, and the proof as a
    ``gather*`` block (truncated to ``max_steps`` rows)."""
    if not isinstance(theorem, Theorem):
        raise TypeError("display_theorem: expected a Theorem")
    parts = [
        '<div class="jacopy-theorem">',
        f'<div class="jacopy-theorem-head"><b>Theorem</b> ({_html.escape(theorem.name)}). '
        f"{_html.escape(theorem.statement)}</div>",
        f'<div class="jacopy-math">\\[ {to_latex(theorem.lhs)} = {to_latex(theorem.rhs)} \\]</div>',
        f'<div class="jacopy-meta"><i>Generality:</i> {_html.escape(theorem.generality)}</div>',
    ]
    if theorem.from_axioms:
        items = "".join(f"<li>{_html.escape(a)}</li>" for a in theorem.from_axioms)
        parts.append(f'<div class="jacopy-meta"><i>Depends on:</i><ul>{items}</ul></div>')
    if theorem.notes:
        parts.append(f'<div class="jacopy-meta"><i>Notes:</i> {_html.escape(theorem.notes)}</div>')
    if with_proof:
        parts.append(
            '<details class="jacopy-proof"><summary><i>Proof</i> '
            f"({len(theorem.proof)} steps)</summary>"
            f"{chain_to_latex(theorem.proof, max_steps=max_steps)}</details>"
        )
    parts.append("</div>")
    return HtmlDisplay("".join(parts))


def _step_summary_html(step: ProofStep, verbosity: str) -> str:
    rule = _html.escape(step.rule)
    pieces = [f'<span class="jacopy-rule">[{rule}]</span>']
    if step.provenance_tag:
        pieces.append(f'<span class="jacopy-tag">({_html.escape(step.provenance_tag)})</span>')
    if verbosity != "compact":
        pieces.append(
            f'<span class="jacopy-math">\\({to_latex(step.before)} \\to {to_latex(step.after)}\\)</span>'
        )
        if verbosity == "full" and step.justification:
            pieces.append(f'<span class="jacopy-just">, {_html.escape(step.justification)}</span>')
    return " ".join(pieces)


def _step_to_html(step: ProofStep, verbosity: str, max_depth: int) -> str:
    summary = _step_summary_html(step, verbosity)
    show_children = verbosity != "compact" and max_depth > 0 and bool(step.children)
    if not show_children:
        return f'<div class="jacopy-step">{summary}</div>'
    child_html = "".join(_step_to_html(ch, verbosity, max_depth - 1) for ch in step.children)
    return (
        '<details open class="jacopy-step">'
        f"<summary>{summary}</summary>"
        f'<div class="jacopy-children">{child_html}</div>'
        "</details>"
    )


def display_step_collapsible(
    step: ProofStep, *, max_depth: int = 64, verbosity: str = "full"
) -> HtmlDisplay:
    """A :class:`ProofStep` as a collapsible HTML tree."""
    if not isinstance(step, ProofStep):
        raise TypeError("display_step_collapsible: expected a ProofStep")
    _check_verbosity(verbosity)
    return HtmlDisplay(_step_to_html(step, verbosity, max_depth))


def display_chain_collapsible(
    chain: ProofChain, *, max_depth: int = 64, title: bool = True, verbosity: str = "full"
) -> HtmlDisplay:
    """A :class:`ProofChain` as a collapsible HTML tree (nested
    sub-proofs fold; ``verbosity`` as in :data:`VERBOSITY_MODES`)."""
    if not isinstance(chain, ProofChain):
        raise TypeError("display_chain_collapsible: expected a ProofChain")
    _check_verbosity(verbosity)
    if len(chain) == 0:
        return HtmlDisplay('<div class="jacopy-proof">(empty proof chain)</div>')
    steps_html = "".join(_step_to_html(s, verbosity, max_depth) for s in chain.steps)
    body = steps_html
    if title:
        body = f'<div class="jacopy-proof-header">Proof ({len(chain)} steps)</div>' + steps_html
    return HtmlDisplay(f'<div class="jacopy-proof">{body}</div>')
