.PHONY: install lint typecheck test train train-baseline evaluate evaluate-baseline audit-classes benchmark data-synthetic clean help

# Default Python and config
PYTHON ?= python
CONFIG ?= configs/base.yaml

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package with all dependencies
	pip install -e ".[all]"

lint: ## Run ruff linter
	$(PYTHON) -m ruff check phasephyto/ tests/ scripts/

lint-fix: ## Auto-fix lint issues
	$(PYTHON) -m ruff check phasephyto/ tests/ scripts/ --fix

typecheck: ## Run mypy type checker
	$(PYTHON) -m mypy phasephyto/ --ignore-missing-imports

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-pc: ## Run phase congruency tests only
	$(PYTHON) -m pytest tests/test_phase_congruency.py -v

test-model: ## Run model forward/backward tests only
	$(PYTHON) -m pytest tests/test_model_forward.py -v

verify: lint typecheck test ## Run all checks (lint + types + tests)

data-synthetic: ## Generate synthetic test data
	$(PYTHON) scripts/download_data.py --dataset synthetic --output data/synthetic

data-plantvillage: ## Download PlantVillage (requires Kaggle API)
	$(PYTHON) scripts/download_data.py --dataset plantvillage --output data/plant_disease

data-plantdoc: ## Download PlantDoc from GitHub
	$(PYTHON) scripts/download_data.py --dataset plantdoc --output data/plant_disease

data-all: ## Download all real datasets
	$(PYTHON) scripts/download_data.py --dataset all --output data/plant_disease

train: ## Train PhasePhyto (CONFIG=configs/plant_disease.yaml)
	$(PYTHON) -m phasephyto.train --config $(CONFIG)

train-baseline: ## Train semantic-only baseline (CONFIG=...)
	$(PYTHON) -m phasephyto.train_baseline --config $(CONFIG)

evaluate: ## Evaluate checkpoint (CONFIG=... CKPT=...)
	$(PYTHON) -m phasephyto.evaluate --config $(CONFIG) --checkpoint $(CKPT)

evaluate-baseline: ## Evaluate baseline checkpoint (CONFIG=... CKPT=...)
	$(PYTHON) -m phasephyto.evaluate_baseline --config $(CONFIG) --checkpoint $(CKPT)

audit-classes: ## Audit source/target class overlap (SOURCE=... TARGET=...)
	$(PYTHON) scripts/audit_class_overlap.py --source $(SOURCE) --target $(TARGET)

benchmark: ## Run PhasePhyto-vs-baseline benchmark orchestration
	$(PYTHON) scripts/benchmark.py --config $(CONFIG)

inference: ## Run inference (CONFIG=... CKPT=... INPUT=...)
	$(PYTHON) -m phasephyto.inference --config $(CONFIG) --checkpoint $(CKPT) --input $(INPUT) --gradcam

clean: ## Remove generated files
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build
	rm -f *.pt eval_results.json
