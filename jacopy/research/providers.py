"""
Structure providers (Faz 8 step 4a).

A geometric structure — an algebroid, a Poisson or Nambu-Poisson
structure — is the AUTHORITY on its own rules: ``engine_rules(registry,
phase=None)`` returns the rules it contributes to an engine, each
owned by the structure and keyed structurally (``Definition.key``),
and its ``declarations`` say which of them are ASSUMPTIONS. Engine
builders add the structure-free base (Cartan calculus, wedge and slot
laws) once and ask every structure for its own rules; a flag such as
the old ``declare_fi`` is never distributed across structures — one
structure may declare the fundamental identity while another does
not, and the engine's owners / a result's requirements say which.

The protocol is duck-typed: any object with ``name``,
``declarations``, ``declares(key)``, ``with_declarations(*keys)`` and
``engine_rules(registry, *, phase=None)`` is a provider.
"""

from __future__ import annotations

from typing import Iterable, List

from jacopy.proof.expansion import Definition

REQUIRED = ("name", "declarations", "declares", "with_declarations", "engine_rules")


def is_provider(structure) -> bool:
    """True when ``structure`` implements the provider protocol."""
    return all(hasattr(structure, attr) for attr in REQUIRED)


def require_provider(structure):
    if not is_provider(structure):
        missing = [a for a in REQUIRED if not hasattr(structure, a)]
        raise TypeError(f"{structure!r} is not a structure provider (missing {missing})")
    return structure


def structure_rules(structure, registry=None, *, phase=None) -> List[Definition]:
    """The rules ``structure`` contributes (its own declarations
    decide the assumption rules)."""
    return list(require_provider(structure).engine_rules(registry, phase=phase))


def declared_axioms(structures: Iterable) -> dict:
    """``{structure name: sorted declarations}`` — what each structure
    assumes, for reports."""
    return {s.name: sorted(s.declarations) for s in structures}


__all__ = ["REQUIRED", "is_provider", "require_provider", "structure_rules", "declared_axioms"]
