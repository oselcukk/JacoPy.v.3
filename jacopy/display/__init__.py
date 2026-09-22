"""Rendering helpers for expressions, proof transcripts and theorems
(PDF item 1: "integrated with LaTeX"; the deferred-items ledger's K1,
ported from v2 on 2026-09-22).

* :mod:`jacopy.display.ascii` — plain text, precedence-aware;
* :mod:`jacopy.display.latex` — LaTeX snippets, ``gather*`` proof
  transcripts, theorem paragraphs, standalone documents, TikZ chains;
* :mod:`jacopy.display.jupyter` — ``_repr_latex_`` / ``_repr_html_``
  wrappers (inline math, proof blocks, collapsible proof trees,
  theorem cards);
* :mod:`jacopy.display.terminal` — coloured trees via the optional
  ``rich`` package, ASCII fallback otherwise.

The display layer reads expressions and chains as data; it never
changes them and the core never depends on it. User-defined nodes get
a LaTeX handler through :func:`jacopy.display.latex.register`.
"""

from jacopy.display.ascii import (
    VERBOSITY_MODES,
    chain_to_ascii,
    step_to_ascii,
    to_ascii,
)
from jacopy.display.jupyter import (
    HtmlDisplay,
    HtmlProofDisplay,
    LatexDisplay,
    display_chain,
    display_chain_collapsible,
    display_expr,
    display_proof,
    display_step,
    display_step_collapsible,
    display_theorem,
)
from jacopy.display.latex import (
    chain_to_latex,
    chain_to_latex_document,
    chain_to_tikz,
    chain_to_tikz_document,
    latex_name,
    register,
    step_to_latex,
    theorem_to_latex,
    theorem_to_latex_document,
    to_latex,
)
from jacopy.display.terminal import (
    HAS_RICH,
    print_chain,
    print_expr,
    print_step,
    render_chain,
    render_expr,
    render_step,
)

__all__ = [
    "HAS_RICH",
    "VERBOSITY_MODES",
    "HtmlDisplay",
    "HtmlProofDisplay",
    "LatexDisplay",
    "chain_to_ascii",
    "chain_to_latex",
    "chain_to_latex_document",
    "chain_to_tikz",
    "chain_to_tikz_document",
    "display_chain",
    "display_chain_collapsible",
    "display_expr",
    "display_proof",
    "display_step",
    "display_step_collapsible",
    "display_theorem",
    "latex_name",
    "print_chain",
    "print_expr",
    "print_step",
    "register",
    "render_chain",
    "render_expr",
    "render_step",
    "step_to_ascii",
    "step_to_latex",
    "theorem_to_latex",
    "theorem_to_latex_document",
    "to_ascii",
    "to_latex",
]
