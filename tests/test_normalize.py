"""Faz 8 step 4b — the common normalization contract: one loop,
``NormalForm(expr, stopped_by ∈ {fixpoint, rounds, steps, target})``,
``converged`` = a fixpoint of the pipeline (not a unique canonical
form), a raising layer on top, and the old helpers as adapters that
keep their own defaults and failure modes."""

import pytest

from jacopy.central.algebroid import algebroid
from jacopy.central.algebroid.engine import algebroid_engine
from jacopy.central.algebroid.theorems import _normalize as _algebroid_normalize
from jacopy.central.objects import Bundle, forms, functions, vector_fields
from jacopy.central.tangent.engine import tangent_engine
from jacopy.central.tangent.lie_bracket import lie_bracket
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import (
    Definition,
    ExpansionEngine,
    NormalForm,
    NormalizationBudgetExceeded,
    ProofFailure,
    normalize,
    require_normal_form,
)
from jacopy.proof.normalize import PASSES, STOP_REASONS


class AtoB(Definition):
    name = "a → b"

    def matches(self, expr):
        return expr == Symbol("a")

    def rewrite(self, expr):
        return Symbol("b")


class BtoA(Definition):
    name = "b → a"

    def matches(self, expr):
        return expr == Symbol("b")

    def rewrite(self, expr):
        return Symbol("a")


def _b_to_a_pass(expr, registry):
    return Symbol("a") if expr == Symbol("b") else expr


def _grow(expr, registry):
    return Sum(expr, Integer(1))


def test_fixpoint_is_a_fixpoint_of_the_pipeline_not_a_canonical_form():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    X, Y = vector_fields("X Y")
    from jacopy.algebra.derivation import Act

    e = Act(lie_bracket(X, Y), f)
    nf = normalize(tangent_engine(registry=reg), e, reg)
    assert isinstance(nf, NormalForm) and nf.stopped_by == "fixpoint" and nf.converged and not nf.exhausted
    assert nf.rounds >= 1 and nf.steps >= 1
    # a fixpoint: running the pipeline again changes nothing and costs no step
    again = normalize(tangent_engine(registry=reg), nf.expr, reg)
    assert again.expr == nf.expr and again.steps == 0 and again.rounds == 1
    # …but a different pipeline (no passes) may stop elsewhere: no uniqueness claim
    bare = normalize(tangent_engine(registry=reg), e, reg, passes=())
    assert bare.converged
    assert "fixpoint of the pipeline" in str(nf)
    assert STOP_REASONS == ("fixpoint", "rounds", "steps", "target") and set(PASSES) == {"product_rule", "simplify"}


def test_round_budget_exhaustion_is_reported_not_raised():
    # a pass that undoes the engine's rewrite makes the ROUND idempotent:
    # that is a genuine fixpoint of the composite pipeline
    nf = normalize(ExpansionEngine([AtoB()]), Symbol("a"), passes=(_b_to_a_pass,), rounds=3)
    assert nf.stopped_by == "fixpoint" and nf.expr == Symbol("a")
    # a pass that keeps growing the expression never settles: the round budget runs out
    nf = normalize(ExpansionEngine([]), Symbol("a"), passes=(_grow,), rounds=3)
    assert nf.stopped_by == "rounds" and nf.exhausted and not nf.converged and nf.rounds == 3
    assert nf.expr == Sum(Sum(Sum(Symbol("a"), Integer(1)), Integer(1)), Integer(1))
    assert "still changing" in nf.detail and "round budget exhausted" in str(nf)
    with pytest.raises(NormalizationBudgetExceeded) as info:
        require_normal_form(ExpansionEngine([]), Symbol("a"), passes=(_grow,), rounds=3, what="growth")
    assert info.value.normal_form.stopped_by == "rounds" and "growth" in str(info.value)


def test_step_budget_exhaustion_keeps_the_partial_expression():
    eng = ExpansionEngine([AtoB(), BtoA()])                 # a true cycle inside one expansion pass
    nf = normalize(eng, Symbol("a"), max_steps=5, record=True)
    assert nf.stopped_by == "steps" and nf.steps == 5 and nf.expr in (Symbol("a"), Symbol("b"))
    assert len(nf.chain) == 5 and nf.chain[0].before == Symbol("a")
    with pytest.raises(NormalizationBudgetExceeded):
        require_normal_form(eng, Symbol("a"), max_steps=5)
    # the old helpers keep raising as engine.expand did
    from jacopy.packages.poisson.tilde import _normalized_by

    with pytest.raises(RuntimeError):
        _normalized_by(eng, Symbol("a"), None)


def test_target_stops_early():
    reg = PropertyRegistry()
    X, Y = vector_fields("X Y")
    e = Sum(X, Neg(X))
    nf = normalize(tangent_engine(registry=reg), e, reg, stop_when=lambda x: x == Integer(0))
    assert nf.stopped_by == "target" and nf.converged and nf.expr == Integer(0)


def test_nested_shape_records_the_strategy_steps():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    E = algebroid("E", Bundle("E"), declare=("lie",))
    u, v, w = E.sections("u v w")
    eng = algebroid_engine(E, registry=reg)
    expr = E.anchor(E.bracket(u, Product(f, v)))
    nf = normalize(eng, expr, reg, rounds=8, inner_rounds=64, record=True)
    assert nf.converged and nf.chain
    rules = [s.rule for s in nf.chain]
    assert any("right-Leibniz" in r for r in rules)
    # the adapter of the algebroid provers returns exactly the contract's answer
    out, steps = _algebroid_normalize(expr, eng, reg)
    assert out == nf.expr and [s.rule for s in steps] == rules
    # inner-loop exhaustion keeps the prover's historical ProofFailure
    with pytest.raises(ProofFailure):
        _algebroid_normalize(Symbol("a"), _PingPongEngine(), reg)


class _PingPongEngine:
    """An engine whose expand_once alternates a ↔ b forever, but one
    step at a time (each expansion pass makes exactly one step) — the
    inner loop then never settles."""

    def __init__(self):
        self._flip = False
        self.definitions = ()
        self.owners = {}
        self.version = 0

    def expand_once(self, expr):
        from jacopy.proof import ProofStep

        if expr not in (Symbol("a"), Symbol("b")):
            return expr, None
        if self._flip:
            self._flip = False
            return expr, None
        self._flip = True
        out = Symbol("b") if expr == Symbol("a") else Symbol("a")
        return out, ProofStep(expr, out, rule="flip")


def test_inner_loop_exhaustion_is_reported_with_its_detail():
    nf = normalize(_PingPongEngine(), Symbol("a"), rounds=2, inner_rounds=3, max_steps=10)
    assert nf.stopped_by == "rounds" and nf.detail.startswith("inner loop")


def test_adapters_keep_their_defaults_and_agree_with_the_contract():
    reg = PropertyRegistry()
    (f,) = functions("f", registry=reg)
    (om,) = forms("ω", degree=1)
    X, Y = vector_fields("X Y")
    from jacopy.algebra.derivation import Act
    from jacopy.central.tangent.exterior import d
    from jacopy.core.multi_eval import MultiEval
    from jacopy.packages.generalized.dialect_bridge import _normalize as _dialect
    from jacopy.packages.metric_affine.bianchi import _normalize as _bianchi
    from jacopy.packages.poisson.symplectic import _normalized_by as _symp
    from jacopy.packages.poisson.tilde import _normalized_by as _tilde

    e = MultiEval(d(om), X, Y, alternating=True, slot_kind="vector")
    eng = tangent_engine(registry=reg)
    want = normalize(eng, e, reg, rounds=10, max_steps=1024).expr
    assert _tilde(eng, e, reg) == want and _symp(eng, e, reg) == want
    assert _dialect(eng, e, reg) == normalize(eng, e, reg, rounds=12, max_steps=60000).expr
    assert _bianchi(e, eng, reg) == normalize(eng, e, reg, rounds=8, inner_rounds=64).expr


def test_invalid_arguments():
    with pytest.raises(TypeError):
        normalize(ExpansionEngine([]), "a")
    with pytest.raises(ValueError):
        normalize(ExpansionEngine([]), Symbol("a"), rounds=0)
    with pytest.raises(ValueError):
        normalize(ExpansionEngine([]), Symbol("a"), passes=("magic",))
    with pytest.raises(ValueError):
        NormalForm(Symbol("a"), "done", 1, 0)
