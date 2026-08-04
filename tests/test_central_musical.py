"""Central code — tilde-interior ι̃ + musical ♯/♭ (Phase 1.E.2)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.core.expr import Atom
from jacopy.core.symbolic_degree import Degree
from jacopy.central.objects import (
    forms,
    vector_fields,
    p_vectors,
    bivector,
    TildeInterior,
    tilde_interior,
    tilde_contract,
    Flat,
    Sharp,
    flat,
    sharp,
)


# --------------------------------------------------------------------- #
# Tilde-interior ι̃_ω                                                   #
# --------------------------------------------------------------------- #


class TestTildeInterior:
    def test_is_derivation(self):
        (w,) = forms("ω", degree=1)
        it = tilde_interior(w)
        assert isinstance(it, TildeInterior)
        assert isinstance(it, Derivation)

    def test_degree_minus_one(self):
        (w,) = forms("ω", degree=1)
        assert tilde_interior(w).degree == Degree.const(-1)

    def test_carries_form(self):
        (w,) = forms("ω", degree=1)
        assert tilde_interior(w).form is w

    def test_repr(self):
        (w,) = forms("ω", degree=1)
        assert tilde_interior(w)._repr_inner() == "ι̃_ω"

    def test_tilde_contract_is_act(self):
        (w,) = forms("ω", degree=1)
        (pi,) = p_vectors("π", degree=3)
        result = tilde_contract(w, pi)
        assert isinstance(result, Act)
        assert isinstance(result.op, TildeInterior)
        assert result.arg is pi

    def test_tilde_contract_lowers_degree(self):
        """ι̃_ω π: p-vector of degree |π| − 1."""
        (w,) = forms("ω", degree=1)
        (pi,) = p_vectors("π", degree=3)
        assert degree_of(tilde_contract(w, pi)) == Degree.const(2)

    def test_tilde_contract_bivector(self):
        """ι̃_ω π (π a bivector, degree 2) → 1-vector (degree 1)."""
        (w,) = forms("ω", degree=1)
        pi = bivector("π")
        assert degree_of(tilde_contract(w, pi)) == Degree.const(1)

    def test_symbolic_degree(self):
        (w,) = forms("ω", degree=1)
        p = Degree.var("p")
        (pi,) = p_vectors("π", degree=p)
        assert degree_of(tilde_contract(w, pi)) == p + Degree.const(-1)

    def test_distinct_forms_distinct(self):
        w, e = forms("ω η", degree=1)
        assert tilde_interior(w) != tilde_interior(e)


# --------------------------------------------------------------------- #
# Musical ♭ (flat): TM → T*M                                           #
# --------------------------------------------------------------------- #


class TestFlat:
    def test_is_atom_not_derivation(self):
        """Musicals are tensorial → Atom, NOT a Derivation (no Leibniz)."""
        (g,) = forms("g", degree=2)  # stand-in for the metric
        fb = flat(g)
        assert isinstance(fb, Atom)
        assert not isinstance(fb, Derivation)

    def test_flat_degree_plus_one(self):
        (g,) = forms("g", degree=2)
        assert flat(g).degree == Degree.const(1)

    def test_flat_raises_grading(self):
        """g^♭(X): vector (0) → 1-form (1)."""
        (g,) = forms("g", degree=2)
        (X,) = vector_fields("X")
        result = flat(g)(X)
        assert isinstance(result, Act)
        assert degree_of(result) == Degree.const(1)  # 1-form ✓

    def test_flat_repr(self):
        (g,) = forms("g", degree=2)
        assert flat(g)._repr_inner() == "g♭"

    def test_flat_rejects_non_expr_arg(self):
        (g,) = forms("g", degree=2)
        with pytest.raises(TypeError):
            flat(g)("nope")


# --------------------------------------------------------------------- #
# Musical ♯ (sharp): T*M → TM                                          #
# --------------------------------------------------------------------- #


class TestSharp:
    def test_is_atom(self):
        pi = bivector("π")
        assert isinstance(sharp(pi), Atom)
        assert not isinstance(sharp(pi), Derivation)

    def test_sharp_degree_minus_one(self):
        pi = bivector("π")
        assert sharp(pi).degree == Degree.const(-1)

    def test_sharp_lowers_grading(self):
        """π^♯(α): 1-form (1) → vector (0)."""
        pi = bivector("π")
        (alpha,) = forms("α", degree=1)
        result = sharp(pi)(alpha)
        assert isinstance(result, Act)
        assert degree_of(result) == Degree.const(0)  # vector ✓

    def test_sharp_repr(self):
        pi = bivector("π")
        assert sharp(pi)._repr_inner() == "π♯"

    def test_sharp_carries_bivector(self):
        pi = bivector("π")
        assert sharp(pi).bivector is pi

    def test_sharp_flat_distinct(self):
        (g,) = forms("g", degree=2)
        assert sharp(g) != flat(g)

    def test_sharp_rejects_non_expr(self):
        with pytest.raises(TypeError):
            Sharp("nope")


# --------------------------------------------------------------------- #
# item 8w: index raise/lower composition                                #
# --------------------------------------------------------------------- #


class TestMusicalComposition:
    def test_flat_then_sharp_grading_roundtrip(self):
        """π^♯(g^♭(X)): vector → 1-form → vector, degree returns to 0."""
        (g,) = forms("g", degree=2)
        pi = bivector("π")
        (X,) = vector_fields("X")
        lowered = flat(g)(X)        # 1-form, degree 1
        raised = sharp(pi)(lowered) # vector, degree 0
        assert degree_of(lowered) == Degree.const(1)
        assert degree_of(raised) == Degree.const(0)
