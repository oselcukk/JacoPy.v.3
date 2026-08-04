# jacopy v3 — Architecture

*Status as of Phase 2.F: ~13,500 lines, 388 tests. Layer contents grow
with the phases (see `ROADMAP.md`), but the layer* **boundaries** *and
dependency rules below are permanent.*

The one-sentence summary: **bottom-up layers, with dependency arrows
always pointing downward** — an upper layer knows the one below it,
the lower one never knows the upper. This keeps the proof engine
independent of geometry and the geometry independent of the packages
(the structural guarantee behind PDF requirement 5, "packages must be
compatible").

```
core  →  algorithms  →  algebra  →  proof
                                      │
                     central/objects  │
                     central/calculus │   (central knows proof;
                     central/tangent  │    proof never knows central)
                     central/algebroid┘
                            │
                        packages/     (empty skeletons until Phase 4+)
```

The few unavoidable upward references are `late imports` inside
function bodies, each marked with a comment explaining why.

---

## 1. `core/` — expression tree and algebraic primitives

The atoms of the system. Depends on nothing; contains *representation*
only, no mathematical computation.

| File | Contents |
|---|---|
| `expr.py` | The tree itself: `Expr` base + `Atom`, `Symbol`, `Integer`, `Rational`, `Neg`, `Sum`, `Product`, `Power`. Structural equality (`_key`), `walk()`, `_rebuild()`. Every node is **inert** — construction never computes. |
| `symbolic_degree.py` | `Degree` — a symbolic degree *polynomial* (`Degree.var("p") + 1`). Parity decisions for Koszul signs come from here; this is what makes arbitrary-p forms possible. |
| `registry.py`, `properties.py` | `PropertyRegistry` — attaching properties (`Scalar`, `Graded`) to nodes from the outside. Generic atoms like `Symbol` have context-dependent degrees; the registry keeps that state local. |
| `pairing.py`, `multi_eval.py` | Evaluation primitives: `⟨α, X⟩` (`Pairing`) and `ω(Y_1,…,Y_k)` (`MultiEval` — `alternating` flag, `vector/covector/mixed` slot kinds). |
| `wedge.py`, `tensor_product.py`, `symmetrize.py`, `hodge.py` | `∧`, `⊗`, `Sym`/`Alt`, `⋆` nodes — *algebraic* primitives, so they live in core; their geometric reading is added in `central`. |
| `indexed_sum.py`, `wildcards.py`, `equality.py` | Symbolic `Σᵢ` (reserved for the symbolic-p route), pattern matching, equality helpers. |

**Why this shape:** representation and computation are separated. The
tree being "dumb" is what allows every computational step to be
recorded as a *rule firing* — the precondition for step-by-step proof
display.

## 2. `algorithms/` — structural algebra passes

Transformations over core trees, ignorant of any geometry.

| File | Contents |
|---|---|
| `simplify.py` | Pipeline driver: flatten → canonicalize → `normalize_alternating` → distribute → sort_product → collect_terms fix-point, plus a final *collecting* phase (`factor_common`, `collect_pairings`) accepted only when strictly smaller. |
| `product_rule.py` | **Graded Leibniz**: `D(a·b) = D(a)·b + (−1)^{|D||a|} a·D(b)`, operator-composition unfolding, `D(0) = 0`, `D(numeric) = 0`, and the `is_constant` protocol (`D(δ^a_b) = 0`). |
| `normalize_alternating.py` | Canonical forms for symmetry: alternating `MultiEval` (sorted args + permutation-parity sign + repeated-arg zero) and `Wedge` of odd-degree factors (same, plus Neg/zero pull-out). Makes orientation-flipped terms cancel in `collect_terms` with **no engine rule**. |
| `factor_terms.py` | The collecting direction: common lead-factor grouping (`f·A + f·B → f·(A+B)`) and pairing/MultiEval slot collection (`Σ sᵢ⟨ω,Vᵢ⟩ → ⟨ω, Σ sᵢVᵢ⟩`) — what exposes bracket-identity residuals to cited theorems. |
| `canonicalize.py`, `sort_product.py`, `flatten.py`, `distribute.py`, `collect_terms.py`, `rewrite.py` | Neg folding, numeric-coefficient consolidation, Koszul-signed graded-commutative sorting (scalars to the front), associativity flattening, distribution, like-term cancellation. |

**Why separate from `proof/`:** these are not rules; they belong to
graded algebra itself and hold in every geometry. Rules (`Definition`s)
depend on declared definitions.

## 3. `algebra/` — operator algebra

| File | Contents |
|---|---|
| `derivation.py` | `Derivation` (a named operator atom carrying a degree), `Act` (the inert application node `D(x)`), **`degree_of`** (all degree laws: `|D(x)| = |D|+|x|`, wedge/⊗ sums with the `wedge_degree` lift, `Sym/Alt` preservation, `|⋆ω| = n−|ω|`, …), `compose`. |
| `commutator.py` | The graded `[A, B]` node with Koszul-signed expansion. |
| `lie_bracket_vf.py` | `LieBracketVF` — the opaque `[X, Y]` Derivation atom (a vector field in its own right). |

"Being an operator" (carrying a degree, participating in Leibniz) is an
algebra concept that precedes geometry: `VectorField`, `Interior`,
`LieDerivative`, `ExteriorDerivative` are all subclasses of
`Derivation`.

## 4. `proof/` — the proof engine

The mechanism is entirely geometry-independent (preserved from v2,
strengthened in the interim engine phase E).

| File | Contents |
|---|---|
| `expansion.py` | `Definition` ABC (`matches`/`rewrite` + **`anchor`** dispatch + axiom/theorem classification) and `ExpansionEngine` (leftmost-innermost fix-point, step recording, `efficient`/`foundational` modes). `default_engine` contains only `ActOverSumOp` — **the purification decision**: the engine ships no geometry rules; each layer registers its own. |
| `strategies.py` | `ExpandAndSimplify` (obstruction `lhs − rhs` → [expand ↔ product_rule] fix-point → simplify, in outer rounds so collected shapes re-enter the engine), `AgreementOnGenerators`, `UnrollToFoundations`, `PortfolioStrategy`, `ProofFailure` (carrying the machine-readable `residual`). |
| `theorems.py` | `Theorem` (the equation itself + a **generality tag**: operator-level / generic-function / symbolic-p / instance + the proof), `TheoremBook`, `TheoremDefinition` + `cite()` — "prove once, cite forever". Citation is the only sanctioned way a proven result enters a proof (no circular proofs). |
| `search.py` | `SearchStrategy`: bounded exhaustive BFS over the memoized expression *state graph* (different rule orders reconverging to the same expression collapse into one node) + greedy beam fallback; scorers (`shortest` / `readable` / `elementary`); the `prove(lhs, rhs, prefer=…)` API. An e-graph backend is a planned drop-in behind the same interface. |
| `chain.py`, `step.py` | `ProofChain`/`ProofStep` — before/after/rule/provenance-tag/sub-steps; the output format of every proof. |
| `diagnostics.py`, `diagnostic_rules.py` | Structural classification of a non-closing residual — the report attached to `ProofFailure`. |
| `recognizers.py`, `verifier.py` | Shape recognizers and higher-level proof helpers. |

## 5. `brackets/` — abstract bracket framework

`GradedBracket` (name, degree, axiom flags, abstract `expand`),
`BracketApply` (the inert `[a, b]` node), `DerivedBracket` /
`VanishingCondition` (Jacobi machinery). Concrete brackets subclass
this: `SchoutenBracket` today; Koszul, Courant, Dorfman in later
phases.

## 6. `central/` — the PDF's "central code" (the actual geometry)

Split into four sub-groups on purpose:

### `central/objects/` — PDF item 8: *objects, no computation*

`Bundle`/`TM` context; `VectorField` (a Derivation!); `Form`/`PVector`
(degree-carrying atoms); `Tensor` (+ `signature_of` (q,r)-type
tracking); `Metric`/`InverseMetric`; `Connection`/`CovariantOp`;
`Frame`/`Coframe` (+ `KroneckerDelta`, an `is_constant` atom);
`Interior`/`TildeInterior` (+ `contract_all`, the alternating case of
item 8w); musical `Flat`/`Sharp`. Common pattern: every object carries
a bundle reference and arity/type guards, but **no computation rules**
— it builds nodes, nothing else. The object/rule split is the file-level
shadow of the definition policy.

**Grading convention** (decided in the Phase 1 audit): vector fields
carry *operator* degree 0 (so `U(f)` is again degree 0); the
*multivector* grading lives on `PVector` and, via the `wedge_degree`
lift, on wedges of vector fields (`|X∧Y| = 2`).

### `central/calculus/` — the *generic* machine (definition policy §4)

`BracketCalculus(name, anchor, bracket)` — an `(ρ, [·,·])` pair as a
context object — and the operators derived from it:
`ExteriorDerivative` (`d`, degree +1) and `LieDerivative` (`L_X`,
degree 0), both carrying their calculus identity. The canonical
definitions live here as engine rules:

- `IntrinsicDDefinition` — the Palais formula (no combinatorial
  prefactor, matching the determinant wedge convention),
- `IntrinsicLDefinition` — `L_X f = ρ(X)f`, `L_X Y = [X,Y]`, and the
  evaluation formula on covariant slots,
- `interior_rules.py` — `ι_X` slot insertion (calculus-independent),
- `symmetrize_rules.py` — `Sym`/`Alt` permutation unfolding (`1/k!`),
- `SlotZero`/`SlotNeg` — multilinearity of the evaluation primitives,
- `scalars.py` — conservative scalar-function recognition.

**Nothing here knows TM.** The same files will serve the algebroid
(`(ρ_E, [·,·]_E)`, Phase 3) and the Poisson cotangent case
(`(π^♯, [·,·]_Kos)` giving `d̃`/`L̃`, Phase 5). Per PDF item 8a, the
usual operators *are* the `(id, Lie)` instantiation — not a separate
code path.

### `central/tangent/` — PDF item 9: the TM instantiation + its theorems

| File | Contents |
|---|---|
| `lie_bracket.py` | The commutator definition `[U,V] = U∘V − V∘U` as an engine rule; the 2.A property proofs (antisymmetry, Jacobi, Leibniz, ℝ-bilinearity); `BracketOrientationDefinition` (theorem-backed canonical orientation). |
| `definitions.py` | The Phase 1 deferrals: `∇_X f = X(f)` (with the scalar guard) and frame duality `⟨e^a, e_b⟩ = δ^a_b`. |
| `exterior.py` | `CARTAN_TM = BracketCalculus("Cartan-TM", id, Lie)`; `d(ω)`; `d² = 0` on functions + TheoremBook registration. |
| `cartan.py` | `L(X, T)`; the Cartan relations proved as theorems (magic `L_X = dι_X + ι_X d` above all — an *axiom* in v2, a *theorem* here); the **repair loop** `prove_with_bracket_identities` (proves residual bracket identities on a generic function, cites them, retries) closing `d² = 0` for p ≥ 1. |
| `anholonomy.py` | `γ^c_ab := ⟨e^c, [e_a, e_b]⟩` (coefficient extraction — no dimension sum, works at symbolic dimension); the frame writing of the bracket properties; holonomic-frame mode. |
| `schouten.py` | The Schouten-Nijenhuis bracket on `Wedge` multivectors (v2 sign convention: shifted grading, four base cases + wedge-Leibniz recursion); its theorems. |
| `engine.py` | `tangent_engine()` — the single assembly point of every definitional rule (plus the canonical-form normalizations, which are theorem-classified but always-on by documented exception). |

### `central/algebroid/` — empty placeholder

Reserved for PDF item 10 (Phase 3): anchor, algebroid bracket,
Jacobiator/Derivator/Predator, locality.

## 7. `packages/` — empty skeletons

`metric_affine/`, `poisson/`, `generalized/`, `algebroid_calculus/`,
`research/` — the PDF item 7 package list, to be filled in Phases 4-8.
Their early existence documents, at the directory level, that the
central code precedes and never depends on the packages.

## 8. `tests/` — one file per delivery slice

Naming follows the layers: `test_foundation` (core), `test_central_*`
(Phase 1 objects, one file per module family), `test_tangent_*`
(Phase 2 proofs), `test_engine_upgrades` (interim phase E). Every proof
family ships *sanity tests* — a deliberately false identity that the
prover must reject.

---

## Cross-cutting decisions (where to look them up)

| Decision | Recorded in |
|---|---|
| Definition policy (one canonical definition per operator; everything else a theorem; definitional vs derived rules; concrete-p unroll vs symbolic-p routes) | `ROADMAP.md`, Phase 2 header |
| Theorem generality tags | `proof/theorems.py` |
| Canonical-form family (alternating sort, wedge sort, γ index order, bracket orientation) — theorem-backed, always-on | rule docstrings + the policy-guard test in `test_engine_upgrades.py` |
| Grading convention (operator vs multivector; the Wedge lift) | `central/objects/multivector.py`, `vector_field.py` docstrings |
| Import-direction exceptions | inline comments at each late import |
