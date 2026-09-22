"""
Rich-terminal renderer: coloured, tree-style output for proof
transcripts. Uses the optional ``rich`` package; without it every
function falls back to :mod:`jacopy.display.ascii` (the return type
is always ``str``). :data:`HAS_RICH` reports which path is active.
"""

from __future__ import annotations

import io
from typing import Optional

from jacopy.core.expr import Expr
from jacopy.display.ascii import (
    VERBOSITY_MODES,
    chain_to_ascii,
    step_to_ascii,
    to_ascii,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


def _check_verbosity(verbosity: str) -> None:
    if verbosity not in VERBOSITY_MODES:
        raise ValueError(
            f"verbosity must be one of {VERBOSITY_MODES}, got {verbosity!r}"
        )


try:  # pragma: no cover - depends on the environment
    from rich.console import Console
    from rich.text import Text
    from rich.tree import Tree

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False
    Console = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    Tree = None  # type: ignore[assignment]


_STYLE_RULE = "bold cyan"
_STYLE_TAG_AXIOM = "yellow"
_STYLE_TAG_THEOREM = "green"
_STYLE_ARROW = "dim"
_STYLE_JUSTIFY = "italic dim"
_STYLE_TITLE = "bold"


def _step_head(step: ProofStep, verbosity: str = "full"):  # pragma: no cover
    t = Text()
    t.append(f"[{step.rule}]", style=_STYLE_RULE)
    if step.provenance_tag == "axiom":
        t.append(f" ({step.provenance_tag})", style=_STYLE_TAG_AXIOM)
    elif step.provenance_tag == "theorem":
        t.append(f" ({step.provenance_tag})", style=_STYLE_TAG_THEOREM)
    if verbosity == "compact":
        return t
    t.append(" ")
    t.append(to_ascii(step.before))
    t.append(" → ", style=_STYLE_ARROW)
    t.append(to_ascii(step.after))
    if verbosity == "full" and step.justification:
        t.append(f" , {step.justification}", style=_STYLE_JUSTIFY)
    return t


def _build_step_tree(step: ProofStep, max_depth: int, verbosity: str = "full"):  # pragma: no cover
    tree = Tree(_step_head(step, verbosity))
    if max_depth > 0 and verbosity != "compact":
        for ch in step.children:
            tree.add(_build_step_tree(ch, max_depth - 1, verbosity))
    return tree


def _build_chain_tree(chain: ProofChain, max_depth: int, title: bool, verbosity: str = "full"):  # pragma: no cover
    heading = Text(f"Proof ({len(chain)} steps)", style=_STYLE_TITLE) if title else Text("")
    root = Tree(heading)
    for step in chain.steps:
        root.add(_build_step_tree(step, max_depth, verbosity))
    return root


def _render_to_text(renderable, *, styles: bool = True) -> str:  # pragma: no cover
    console = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=True,
        color_system="truecolor" if styles else None,
        width=120,
    )
    console.print(renderable)
    return console.export_text(styles=styles).rstrip("\n")


def render_expr(expr: Expr) -> str:
    """Render an :class:`Expr` for the terminal (same as :func:`to_ascii`)."""
    if not isinstance(expr, Expr):
        raise TypeError("render_expr: expected an Expr")
    return to_ascii(expr)


def render_step(step: ProofStep, *, max_depth: int = 64, verbosity: str = "full") -> str:
    """Render a :class:`ProofStep` as a coloured tree (or ASCII fallback)."""
    if not isinstance(step, ProofStep):
        raise TypeError("render_step: expected a ProofStep")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        return step_to_ascii(step, max_depth=max_depth, verbosity=verbosity)
    return _render_to_text(_build_step_tree(step, max_depth, verbosity))  # pragma: no cover


def render_chain(
    chain: ProofChain, *, max_depth: int = 64, title: bool = True, verbosity: str = "full"
) -> str:
    """Render a :class:`ProofChain` as a coloured tree (or ASCII fallback)."""
    if not isinstance(chain, ProofChain):
        raise TypeError("render_chain: expected a ProofChain")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        return chain_to_ascii(chain, max_depth=max_depth, verbosity=verbosity)
    if len(chain) == 0:  # pragma: no cover
        return "(empty proof chain)"
    return _render_to_text(_build_chain_tree(chain, max_depth, title, verbosity))  # pragma: no cover


def print_expr(expr: Expr, *, console: Optional["Console"] = None) -> None:
    """Pretty-print an :class:`Expr` to the terminal."""
    if not isinstance(expr, Expr):
        raise TypeError("print_expr: expected an Expr")
    if not HAS_RICH:
        print(to_ascii(expr))
        return
    (console or Console()).print(to_ascii(expr))  # pragma: no cover


def print_step(
    step: ProofStep, *, console: Optional["Console"] = None, max_depth: int = 64, verbosity: str = "full"
) -> None:
    """Pretty-print a :class:`ProofStep` to the terminal."""
    if not isinstance(step, ProofStep):
        raise TypeError("print_step: expected a ProofStep")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        print(step_to_ascii(step, max_depth=max_depth, verbosity=verbosity))
        return
    (console or Console()).print(_build_step_tree(step, max_depth, verbosity))  # pragma: no cover


def print_chain(
    chain: ProofChain,
    *,
    console: Optional["Console"] = None,
    max_depth: int = 64,
    title: bool = True,
    verbosity: str = "full",
) -> None:
    """Pretty-print a :class:`ProofChain` to the terminal."""
    if not isinstance(chain, ProofChain):
        raise TypeError("print_chain: expected a ProofChain")
    _check_verbosity(verbosity)
    if not HAS_RICH:
        print(chain_to_ascii(chain, max_depth=max_depth, verbosity=verbosity))
        return
    if len(chain) == 0:  # pragma: no cover
        print("(empty proof chain)")
        return
    (console or Console()).print(_build_chain_tree(chain, max_depth, title, verbosity))  # pragma: no cover
