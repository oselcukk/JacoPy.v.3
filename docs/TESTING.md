# Testing

The protected test set is the union of the four groups below. A change
is green when all four are; the counts change as the suite grows and
are not the criterion — the commands are.

## 1. Fast suite (every push, every supported Python)

```sh
.venv/bin/python -m pytest tests -q
```

Pure Python, no optional dependency needed. In CI
(`.github/workflows/tests.yml`, job `wheel`) this runs on Python 3.10,
3.11, 3.12, 3.13 and 3.14 **from the installed wheel**: the wheel is
built with `python -m build`, installed into a clean interpreter, and
the tests are executed from a copy of `tests/` outside the repository
so the source tree cannot shadow the package. The same check locally:

```sh
python -m build --wheel --outdir dist
python -m venv /tmp/jacopy-wheel && /tmp/jacopy-wheel/bin/pip install dist/*.whl pytest
cp -R tests /tmp/jacopy-wheel/tests && cd /tmp/jacopy-wheel && bin/python -m pytest tests -q
```

## 2. Slow proof suites (weekly / on request)

Some closures take one to fifteen minutes. They are gated by
environment variables and skipped otherwise (the skip reason names the
variable and the expected time):

```sh
JACOPY_RUN_SLOW=1 .venv/bin/python -m pytest tests -q          # ~1–2 min closures
JACOPY_RUN_VERY_SLOW=1 .venv/bin/python -m pytest tests -q     # the ~14 min rotated metric invariance
```

CI runs the `JACOPY_RUN_SLOW=1` set in the `slow` job on the weekly
schedule and on a manual `workflow_dispatch` with `slow = true`.

## 3. Notebook smoke run

```sh
.venv/bin/python scripts/run_notebooks.py
```

Executes every code cell of every notebook under `examples/` in a fresh
namespace, in order. It is a code check, not a rendering check.

## 4. Independent audit tests (outside the repository)

The audit directory `../audit_jacopy_v3/` holds the reviewer's
counterexample tests. They are part of the protected set and must be
re-run after every fix round:

```sh
cd ../audit_jacopy_v3 && ../jacopy_v3/.venv/bin/python -m pytest -q
```

Files: `test_review_regressions.py`, `test_fix_recheck.py`,
`test_second_fix_recheck.py`, `test_compliance_gaps.py`,
`test_new_developments_audit.py`, `test_e6e9c1e_recheck.py`,
`test_research_notebook_audit.py`, `test_post_f685_audit.py`,
`test_f6b6791_recheck.py`, `test_library_api_audit.py` (the six
library-API contracts), `test_4dd888e_progress.py`,
`test_1b3c4b3_recheck.py`, `test_0058df0_k3_audit.py`,
`test_7a11162_recheck.py`, plus the coordinate oracles they import.
The audit files are not published with the package.

## Optional dependencies

`rich` (coloured terminal proofs), `sympy` (component workflows),
`pdflatex` (the display tests compile a real document when it is on
`PATH`; they are skipped otherwise). None is needed for the fast suite.
