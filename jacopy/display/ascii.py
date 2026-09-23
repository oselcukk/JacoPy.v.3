"""
Plain-text renderer for :class:`~jacopy.core.expr.Expr` trees and
proof transcripts.

Every v3 node already carries a mathematical ``_repr_inner``; this
module adds sign normalisation (``a + (-b)`` → ``a - b``) and
precedence-aware parentheses on the core arithmetic nodes, and the
verbosity-controlled step/chain layouts shared with the terminal and
Jupyter renderers. Unknown nodes fall back to ``_repr_inner``.
"""

from __future__ import annotations

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


_P_ATOM = 100
_P_CALL = 90
_P_POWER = 80
_P_PRODUCT = 60
_P_WEDGE = 55
_P_NEG = 50
_P_SUM = 40


Handler = Callable[[Expr, int], str]
_HANDLERS: Dict[Type[Expr], Handler] = {}


def _register(cls: Type[Expr]):
    def decorator(fn: Handler) -> Handler:
        _HANDLERS[cls] = fn
        return fn

    return decorator


def to_ascii(expr: Expr, ctx_precedence: int = 0) -> str:
    """Render ``expr`` as plain text."""
    if not isinstance(expr, Expr):
        raise TypeError("to_ascii: expected an Expr")
    for cls in type(expr).__mro__:
        h = _HANDLERS.get(cls)
        if h is not None:
            return h(expr, ctx_precedence)
    return expr._repr_inner()


def _wrap(text: str, own_prec: int, ctx_prec: int) -> str:
    if own_prec < ctx_prec:
        return f"({text})"
    return text


@_register(Symbol)
def _sym(expr: Symbol, _ctx: int) -> str:
    return expr.name


@_register(Integer)
def _int(expr: Integer, ctx: int) -> str:
    v = expr.value
    return _wrap(str(v), _P_NEG, ctx) if v < 0 else str(v)


@_register(Rational)
def _rat(expr: Rational, ctx: int) -> str:
    # a fraction is a quotient, not an atom: as a power base or
    # exponent it needs parentheses — ``3/2**2`` would read as 3/4
    # (2026-09-23 audit, F5)
    text = f"{expr.p}/{expr.q}"
    return _wrap(text, _P_PRODUCT if expr.p >= 0 else _P_NEG, ctx)


@_register(Neg)
def _neg(expr: Neg, ctx: int) -> str:
    return _wrap(f"-{to_ascii(expr.arg, _P_NEG + 1)}", _P_NEG, ctx)


@_register(Sum)
def _sum(expr: Sum, ctx: int) -> str:
    parts: list[str] = []
    for i, child in enumerate(expr.children):
        if isinstance(child, Neg):
            parts.append(("- " if i > 0 else "-") + to_ascii(child.arg, _P_NEG + 1))
        else:
            parts.append(("+ " if i > 0 else "") + to_ascii(child, _P_SUM + 1))
    text = " ".join(parts) if len(parts) > 1 else parts[0] if parts else "0"
    return _wrap(text, _P_SUM, ctx)


@_register(Product)
def _prod(expr: Product, ctx: int) -> str:
    parts = [to_ascii(c, _P_PRODUCT + 1) for c in expr.children]
    return _wrap(" * ".join(parts) if parts else "1", _P_PRODUCT, ctx)


@_register(Power)
def _pow(expr: Power, ctx: int) -> str:
    return _wrap(
        f"{to_ascii(expr.base, _P_POWER + 1)}**{to_ascii(expr.exp, _P_POWER + 1)}",
        _P_POWER,
        ctx,
    )


def _load_structural() -> None:
    from jacopy.core.wedge import Wedge

    @_register(Wedge)
    def _wedge(expr, ctx):
        return _wrap(" ∧ ".join(to_ascii(c, _P_WEDGE + 1) for c in expr.children), _P_WEDGE, ctx)


_load_structural()


#: Recognised verbosity levels for proof-transcript rendering:
#: ``"full"`` (rule + tag + before → after + justification + children),
#: ``"summary"`` (no justification), ``"compact"`` (rule + tag only).
VERBOSITY_MODES = ("full", "summary", "compact")


def _check_verbosity(verbosity: str) -> None:
    if verbosity not in VERBOSITY_MODES:
        raise ValueError(
            f"verbosity must be one of {VERBOSITY_MODES}, got {verbosity!r}"
        )


def step_to_ascii(
    step: ProofStep,
    indent: int = 0,
    max_depth: int = 64,
    *,
    verbosity: str = "full",
) -> str:
    """Render a single :class:`ProofStep` including nested children."""
    if not isinstance(step, ProofStep):
        raise TypeError("step_to_ascii: expected a ProofStep")
    _check_verbosity(verbosity)
    pad = "  " * indent
    tag = f" ({step.provenance_tag})" if step.provenance_tag else ""
    if verbosity == "compact":
        return f"{pad}[{step.rule}]{tag}"
    head = f"{pad}[{step.rule}]{tag} {to_ascii(step.before)} -> {to_ascii(step.after)}"
    if verbosity == "full" and step.justification:
        head = f"{head}  -- {step.justification}"
    lines = [head]
    if max_depth > 0 and step.children:
        for ch in step.children:
            lines.append(step_to_ascii(ch, indent + 1, max_depth - 1, verbosity=verbosity))
    return "\n".join(lines)


def chain_to_ascii(
    chain: ProofChain,
    max_depth: int = 64,
    *,
    verbosity: str = "full",
) -> str:
    """Render an entire :class:`ProofChain` as an ordered step list."""
    if not isinstance(chain, ProofChain):
        raise TypeError("chain_to_ascii: expected a ProofChain")
    _check_verbosity(verbosity)
    if len(chain) == 0:
        return "(empty proof chain)"
    return "\n".join(
        step_to_ascii(s, indent=0, max_depth=max_depth, verbosity=verbosity)
        for s in chain.steps
    )
