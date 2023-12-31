# Makefile
SHELL = /bin/zsh
# Styling
.PHONY: style
style:
	black .
	flake8
	isort .
.PHONY: logs
# Environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e ".[dev]" && \
	pre-commit install && \
	pre-commit autoupdate
# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage
# Test
.PHONY: test
test:
	cd tests && great_expectations checkpoint run traffic_data
	pytest
.PHONY: help
help:
	@echo "Commands"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
.PHONY: dvc
dvc:
	dvc add data/data.csv
	dvc push
