# jacopy — development convenience targets.
#
# Quick start:
#   make setup      # one-shot: create .venv, install editable + all deps,
#                   # register Jupyter kernel.
#   make test       # run the full test suite inside .venv.
#   make notebooks  # smoke-execute every tutorial notebook.
#   make clean      # remove .venv and caches.

# Override with e.g. `make setup PYTHON=/opt/homebrew/bin/python3.12`.
PYTHON ?= python3
VENV   := .venv
VBIN   := $(VENV)/bin

.PHONY: setup test notebooks clean kernel deps test-wheel

# One-shot dev environment.
setup: $(VENV)/.installed kernel
	@echo "✓ Setup complete. Activate with:  source $(VBIN)/activate"
	@echo "  Or open a notebook and pick the 'Python (jacopy v3)' kernel."

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(VBIN)/pip install --upgrade pip

$(VENV)/.installed: $(VENV)/bin/python pyproject.toml
	$(VBIN)/pip install -e ".[dev,parallel]"
	@touch $@

deps: $(VENV)/.installed

kernel: $(VENV)/.installed
	$(VBIN)/python -m ipykernel install --user --name=jacopy-v3 \
		--display-name="Python (jacopy v3)"

test: $(VENV)/.installed
	$(VBIN)/python -m pytest tests/ -q

notebooks: $(VENV)/.installed
	$(VBIN)/python scripts/run_notebooks.py

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ .mypy_cache .ruff_cache

# Build the wheel and run the fast suite from the INSTALLED package in a
# throw-away venv (what CI does on every supported Python).
test-wheel: $(VENV)/.installed
	rm -rf build dist && $(VBIN)/python -m pip install -q build && $(VBIN)/python -m build --wheel --outdir dist
	rm -rf .wheel-test && $(PYTHON) -m venv .wheel-test && .wheel-test/bin/pip install -q dist/*.whl pytest
	rm -rf .wheel-test/tests && cp -R tests .wheel-test/tests && cd .wheel-test && bin/python -m pytest tests -q -p no:cacheprovider
