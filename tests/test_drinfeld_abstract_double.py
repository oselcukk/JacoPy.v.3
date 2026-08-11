"""Drinfeld package — Phase 6.I.4: the ABSTRACT main theorem of the
double as a finite declaration sweep on the Phase 3 layer. Each
property of the double closes under exactly the right declared
subset of A-properties, and honest-fails one declaration short."""

import pytest

from jacopy.core.registry import PropertyRegistry
from jacopy.central.objects import Bundle, forms, functions
from jacopy.central.algebroid import algebroid
from jacopy.proof.strategies import ProofFailure
import jacopy.packages.drinfeld.abstract_double as ad


def _setup(decls, tag):
    reg = PropertyRegistry()
    (f,) = functions("fa", registry=reg)
    E = algebroid(f"Ea{tag}", Bundle(f"Ea{tag}"), declare=decls)
    u, v, w, pr = E.sections("ua va wa sa")
    al, be, ga = forms("αa βa γa", degree=1, bundle=E.bundle)
    return reg, f, E, u, v, w, pr, al, be, ga


class TestRightLeibnizSweep:
    """The double's right-Leibniz defect IS the locality operator:
    under the 'local' level the form-component defect equals
    ⟨α, L(df,v,w)⟩ exactly (paper (3.17)/(5.8) in instance form)."""

    def test_local_closes_with_locality_defect(self):
        reg, f, E, u, v, w, pr, al, be, ga = _setup(("local",), "r1")
        cv, cf = ad.prove_abstract_double_right_leibniz(
            E, u, al, v, be, f, w, registry=reg
        )
        assert cv.steps and cf.steps

    def test_right_leibniz_alone_fails(self):
        """Without left-Leibniz the [f·v, w] legs stay open."""
        reg, f, E, u, v, w, pr, al, be, ga = _setup(
            ("almost-leibniz",), "r2"
        )
        with pytest.raises(ProofFailure):
            ad.prove_abstract_double_right_leibniz(
                E, u, al, v, be, f, w, registry=reg
            )

    def test_bare_fails(self):
        reg, f, E, u, v, w, pr, al, be, ga = _setup((), "r3")
        with pytest.raises(ProofFailure):
            ad.prove_abstract_double_right_leibniz(
                E, u, al, v, be, f, w, registry=reg
            )


class TestSymmetricPartSweep:
    """form = d_E⟨,⟩₊ is the 3.F magic theorem; the vector leg is
    exactly the antisymmetric declaration."""

    def test_antisymmetric_closes(self):
        reg, f, E, u, v, w, pr, al, be, ga = _setup(
            ("antisymmetric",), "s1"
        )
        cv, cf = ad.prove_abstract_double_symmetric_part(
            E, u, al, v, be, f, w, registry=reg
        )
        assert cv.steps and cf.steps

    def test_bare_fails(self):
        reg, f, E, u, v, w, pr, al, be, ga = _setup((), "s2")
        with pytest.raises(ProofFailure):
            ad.prove_abstract_double_symmetric_part(
                E, u, al, v, be, f, w, registry=reg
            )


class TestJacobiSweep:
    """The double's form Leibniz-Jacobi closes on a LIE algebroid
    (jacobi + antisymmetric + right-leibniz; anchor-morphism enters
    as the 3.D theorem, declared) with cited bracket-level Jacobi
    instances — and honest-fails when jacobi is withheld."""

    def test_lie_closes(self):
        reg, f, E, u, v, w, pr, al, be, ga = _setup(
            ("lie", "anchor-morphism"), "j1"
        )
        chain, used = ad.prove_abstract_double_jacobi_form(
            E, u, al, v, be, w, ga, pr, f, registry=reg
        )
        assert chain.steps and used

    def test_without_jacobi_fails(self):
        reg, f, E, u, v, w, pr, al, be, ga = _setup(
            ("local", "antisymmetric", "anchor-morphism"), "j2"
        )
        with pytest.raises(ProofFailure):
            ad.prove_abstract_double_jacobi_form(
                E, u, al, v, be, w, ga, pr, f, registry=reg
            )
