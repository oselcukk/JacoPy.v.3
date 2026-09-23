"""
Minimal context identity — OWNERSHIP of rules, theorems and proof
steps (Faz 8 step 2c).

The expression tree names a structure by a STRING: an algebroid
bracket is ``AlgebroidBracket("E", u, v)``, a sharp map carries the
bivector ``π``. Two structure objects with the same name therefore
produce IDENTICAL expressions — ``algebroid("E", declare=("lie",))``
and a bare ``algebroid("E")`` build the same ``[u, v]_E`` — while their
assumptions differ. A rule declared for one of them matches the
expressions of the other, and a theorem proved under the first one's
axioms is structurally citable in the second one's engine. The tree
cannot tell them apart, so ownership is tracked NEXT TO the tree, and
the tree itself does not change:

* every :class:`~jacopy.proof.expansion.Definition` that is built for
  a structure records it as ``owner`` (the declaration rules of an
  algebroid, its definitional expansions, a cited
  :class:`~jacopy.proof.theorems.TheoremDefinition`);
* every :class:`~jacopy.proof.theorems.Theorem` proved in a structure
  records it as ``owner`` (``None`` = unowned / legacy, see step 3b);
* every :class:`~jacopy.proof.step.ProofStep` fired by an owned rule
  records that owner — a closed proof's step list is its exact
  assumption record, now down to the structure OBJECT and not only
  its display name.

The policy (:func:`check_owners`, applied by every
:class:`~jacopy.proof.expansion.ExpansionEngine` on registration):
two owners that share the tree-visible name but are different
structures may not coexist — the mixture is AMBIGUOUS and is
rejected with :class:`AmbiguousOwnerError`, never resolved by a
guess. Equal structures (same name, bundle and declarations) are one
owner. :class:`OwnerScope` is the explicit context object: ``bind`` a
structure to its name once, ``check`` items against the binding, and
``transfer`` a theorem to another structure only when the tree stays
valid (same name) and the target assumes at least what the proof
assumed (its declarations contain the owner's).

Owners are compared with ``==``; a structure class decides its own
identity (an :class:`~jacopy.central.algebroid.context.Algebroid` by
name, bundle and declarations; a Poisson structure by its bivector).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, Optional


class AmbiguousOwnerError(ValueError):
    """Two different structures share one tree-visible name in a place
    where the tree cannot tell them apart (an engine, a scope, a
    citation)."""


def owner_name(owner) -> str:
    """The name under which ``owner`` appears in the expression tree:
    its ``name`` when it has a string one, else the display name of
    its ``pi`` (Poisson / Nambu structures), else its ``repr``."""
    name = getattr(owner, "name", None)
    if isinstance(name, str) and name:
        return name
    pi = getattr(owner, "pi", None)
    if pi is not None and hasattr(pi, "_repr_inner"):
        return pi._repr_inner()
    return repr(owner)


def owner_of(item):
    """The recorded owner of a rule / theorem / step (``None`` when it
    records none). An object without an ``owner`` attribute is taken
    to BE an owner (a structure passed directly)."""
    if hasattr(type(item), "owner") or hasattr(item, "owner"):
        return getattr(item, "owner", None)
    return item


def _describe(owner) -> str:
    decl = getattr(owner, "declarations", None)
    if decl is not None:
        return f"{owner!r} with declarations {sorted(decl)}"
    return repr(owner)


def _same_owner(a, b) -> bool:
    return a is b or a == b


def is_substructure(owner, of) -> bool:
    """True when ``owner`` is the SAME structure as ``of`` with at most
    the same declarations: same tree-visible name and bundle, and
    ``owner.declarations ⊆ of.declarations``. Rules and theorems of a
    sub-structure are valid in the super-structure (they assume less);
    the converse is not (Faz 8 step 3b refinement of 2c). Structures
    without a declaration record are compatible only when equal."""
    if _same_owner(owner, of):
        return True
    d1, d2 = getattr(owner, "declarations", None), getattr(of, "declarations", None)
    if d1 is None or d2 is None:
        return False
    return (
        owner_name(owner) == owner_name(of)
        and getattr(owner, "bundle", None) == getattr(of, "bundle", None)
        and set(d1) <= set(d2)
    )


def check_owners(items: Iterable, *, bound: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Group the owners of ``items`` by tree-visible name; raise
    :class:`AmbiguousOwnerError` when one name is claimed by two
    different structures, or (with ``bound``) by a structure other
    than the one bound to that name. Returns ``{name: owner}``."""
    seen: Dict[str, object] = {}
    for item in items:
        owner = owner_of(item)
        if owner is None:
            continue
        name = owner_name(owner)
        prev = seen.get(name)
        if prev is None:
            if bound is not None and name in bound and not is_substructure(owner, bound[name]):
                raise AmbiguousOwnerError(
                    f"the name {name!r} is bound to {_describe(bound[name])}, "
                    f"but this item is owned by {_describe(owner)}, which assumes "
                    "more or is a different structure; the tree cannot distinguish "
                    "them — bind one structure per name, or transfer the item explicitly"
                )
            seen[name] = bound[name] if (bound is not None and name in bound) else owner
        elif is_substructure(owner, prev):
            continue  # a sub-structure's item is valid under the structure already seen
        elif is_substructure(prev, owner):
            seen[name] = owner  # the new item's structure dominates: it becomes the name's owner
        else:
            raise AmbiguousOwnerError(
                f"two different structures share the tree-visible name {name!r}: "
                f"{_describe(prev)} vs {_describe(owner)}; their rules and theorems "
                "match the same expressions, so mixing them is ambiguous — use one "
                "structure object per name (or rename one of them)"
            )
    return seen


def chain_owners(chain) -> tuple:
    """The distinct owners recorded on the steps of ``chain`` (children
    included), in first-appearance order — the structures whose
    assumptions the proof used."""
    out: list = []

    def visit(steps):
        for s in steps:
            o = getattr(s, "owner", None)
            if o is not None and not any(_same_owner(o, x) for x in out):
                out.append(o)
            visit(getattr(s, "children", ()))

    visit(getattr(chain, "steps", chain))
    return tuple(out)


class OwnerScope:
    """The explicit context: one structure per tree-visible name.

    ``bind`` records a structure under its name (binding an equal
    structure again is a no-op; a DIFFERENT one under a bound name is
    rejected). ``check`` verifies rules / theorems / steps / structures
    against the binding. ``transfer`` re-owns a theorem explicitly.
    """

    __slots__ = ("_bound",)

    def __init__(self) -> None:
        self._bound: Dict[str, object] = {}

    @property
    def bound(self) -> Dict[str, object]:
        return dict(self._bound)

    def bind(self, structure, *, name: Optional[str] = None):
        """Bind ``structure`` to ``name`` (default: its tree-visible
        name). Returns the structure."""
        if structure is None:
            raise TypeError("cannot bind None")
        key = name if name is not None else owner_name(structure)
        prev = self._bound.get(key)
        if prev is not None:
            if _same_owner(prev, structure):
                return prev  # an equal structure is the same owner: keep the first
            raise AmbiguousOwnerError(
                f"the name {key!r} is already bound to {_describe(prev)}; "
                f"refusing to also bind {_describe(structure)} — the tree "
                "cannot distinguish them"
            )
        self._bound[key] = structure
        return structure

    def owner(self, name: str):
        """The structure bound to ``name`` (``KeyError`` when unbound)."""
        if name not in self._bound:
            raise KeyError(f"no structure bound to {name!r}; bound: {sorted(self._bound)}")
        return self._bound[name]

    def check(self, items: Iterable) -> Dict[str, object]:
        """:func:`check_owners` against this scope's binding. Items may
        be rules, theorems, steps or structures."""
        return check_owners(items, bound=self._bound)

    def check_engine(self, engine) -> Dict[str, object]:
        """Check an engine's rule owners against the binding."""
        return self.check(engine.definitions)

    def transfer(self, theorem, to, *, note: str = ""):
        """Re-own ``theorem`` to the structure ``to``, explicitly.

        Allowed only when the tree stays valid — ``to`` has the SAME
        tree-visible name as the recorded owner — and ``to`` assumes at
        least what the proof assumed: both structures expose a
        ``declarations`` set and ``to``'s contains the owner's. An
        unowned theorem, a renamed target, a target lacking a
        declaration the proof used, or structures without a
        declaration record are all refused (never guessed)."""
        from jacopy.proof.theorems import Theorem

        if not isinstance(theorem, Theorem):
            raise TypeError("transfer expects a Theorem")
        old = theorem.owner
        if old is None:
            raise AmbiguousOwnerError(
                f"theorem {theorem.name!r} records no owner; an unowned (legacy) "
                "theorem cannot be transferred — its assumptions are not recorded"
            )
        if owner_name(old) != owner_name(to):
            raise AmbiguousOwnerError(
                f"theorem {theorem.name!r} is stated in the tree under the name "
                f"{owner_name(old)!r}; it cannot be transferred to {owner_name(to)!r} "
                "without changing the expressions (the tree does not change)"
            )
        old_decl = getattr(old, "declarations", None)
        new_decl = getattr(to, "declarations", None)
        if old_decl is None or new_decl is None:
            if _same_owner(old, to):
                return theorem
            raise AmbiguousOwnerError(
                f"theorem {theorem.name!r}: {_describe(old)} and {_describe(to)} carry "
                "no declaration record to compare, so whether the target assumes "
                "what the proof used is not decidable here"
            )
        missing = set(old_decl) - set(new_decl)
        if missing:
            raise AmbiguousOwnerError(
                f"theorem {theorem.name!r} was proved under the declarations "
                f"{sorted(old_decl)} of {old!r}; the target {to!r} lacks "
                f"{sorted(missing)}"
            )
        bound = self._bound.get(owner_name(to))
        if bound is not None and not _same_owner(bound, to):
            raise AmbiguousOwnerError(
                f"the name {owner_name(to)!r} is bound to {_describe(bound)} in this "
                f"scope, not to {_describe(to)}"
            )
        stamp = f"transferred from {old!r} (declarations {sorted(old_decl)}) to {to!r}"
        if note:
            stamp += f": {note}"
        notes = (theorem.notes + "\n" if theorem.notes else "") + stamp
        return dataclasses.replace(theorem, owner=to, notes=notes)

    def __repr__(self) -> str:
        return f"OwnerScope({sorted(self._bound)})"


__all__ = [
    "AmbiguousOwnerError",
    "is_substructure",
    "OwnerScope",
    "chain_owners",
    "check_owners",
    "owner_name",
    "owner_of",
]
