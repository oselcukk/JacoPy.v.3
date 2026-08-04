"""
Central code — basic objects and operations valid on both TM and an
algebroid E (PDF items 8-10).

Design principle: every object is defined so that taking ``E = TM``,
``ρ_E = id_TM`` and ``[·,·]_E = [·,·]_Lie`` makes the algebroid
version yield the usual one.

Subpackages:

* :mod:`~jacopy.central.objects` — basic objects: functions, vector
  fields, p-forms/p-vectors, (q,r)-tensors, metric, connection,
  frame/coframe, interior/tilde-interior, wedge/tensor product,
  (anti)symmetrization (item 8).
* :mod:`~jacopy.central.tangent` — the TM special case: Lie bracket,
  anholonomy coefficients, d, L, Schouten-Nijenhuis, Cartan relations
  (item 9).
* :mod:`~jacopy.central.algebroid` — the algebroid special case:
  anchor, bracket, locality projector, Jacobiator, Derivator,
  Predator and the algebroid properties (item 10).
"""
