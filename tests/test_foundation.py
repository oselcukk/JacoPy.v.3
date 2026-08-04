"""Foundation smoke tests — do the core layers (core / engine / algebra /
brackets) work without calculus or any package?

These tests verify the soundness of v3's preserved core (the expr tree +
proof engine carried over from v2). As geometry packages (central code,
metric-affine, generalized, …) are added, they will bring their own test
files.
"""

from jacopy.core.expr import (
    Expr,
    Symbol,
    Integer,
    Sum,
    Product,
    Neg,
    Zero,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.algebra.derivation import Act, Derivation
from jacopy.algorithms.simplify import simplify
from jacopy.proof import (
    ProofChain,
    ProofStep,
    ExpansionEngine,
    ExpandAndSimplify,
    default_engine,
)
from jacopy.brackets import GradedBracket, BracketApply, DerivedBracket


# --------------------------------------------------------------------- #
# Import / package skeleton                                             #
# --------------------------------------------------------------------- #


def test_top_level_import():
    import jacopy
    assert jacopy.__version__ == "0.0.1"


def test_package_skeleton_importable():
    import jacopy.central
    import jacopy.central.objects
    import jacopy.central.tangent
    import jacopy.central.algebroid
    import jacopy.packages
    import jacopy.packages.metric_affine
    import jacopy.packages.poisson
    import jacopy.packages.algebroid_calculus
    import jacopy.packages.generalized
    import jacopy.packages.research


# --------------------------------------------------------------------- #
# Expr tree                                                             #
# --------------------------------------------------------------------- #


def test_expr_construction_and_equality():
    X, Y = Symbol("X"), Symbol("Y")
    a = Sum(Product(X, Y), Neg(Product(X, Y)))
    b = Sum(Product(X, Y), Neg(Product(X, Y)))
    assert a == b
    assert hash(a) == hash(b)


def test_simplify_cancels():
    X, Y = Symbol("X"), Symbol("Y")
    e = Sum(Product(X, Y), Neg(Product(X, Y)))
    assert simplify(e) == Zero


def test_symbolic_degree_polynomial():
    p = Degree.var("p")
    q = p + p + Degree.const(1)          # 2p + 1
    assert q == Degree.const(1) + Degree.var("p") + Degree.var("p")


# --------------------------------------------------------------------- #
# Engine (pure)                                                         #
# --------------------------------------------------------------------- #


def test_default_engine_is_pure():
    """The pure engine carries only the structural Act-over-Sum rule; it has
    no calculus (d, ι, L) dependency."""
    eng = default_engine()
    assert isinstance(eng, ExpansionEngine)


def test_act_over_sum_op_linearity():
    """(A + B)(x) = A(x) + B(x) — operator-slot linearity (pure engine)."""
    A = Derivation("A", degree=0)
    B = Derivation("B", degree=0)
    X = Symbol("X")
    expr = Act(Sum(A, B), X)
    out, _steps = default_engine().expand(expr)
    assert isinstance(out, Sum)
    assert len(out.children) == 2


# --------------------------------------------------------------------- #
# Proof data types                                                      #
# --------------------------------------------------------------------- #


def test_proof_chain_records_steps():
    X, Y = Symbol("X"), Symbol("Y")
    before = Product(X, Y)
    after = Product(Y, X)
    chain = ProofChain()
    chain.append(ProofStep(before, after, rule="test", justification="swap"))
    assert len(chain) == 1
    assert chain.steps[0].before == before
    assert chain.steps[0].after == after
