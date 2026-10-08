# G-Research crypto forecasting - common commands
# Run `make` with no arguments to list every target.

PYTHON    ?= python3
DATA_DIR  ?= data
ARTIFACTS ?= artifacts
MODEL     ?= lgbm

.PHONY: help install data verify test train train-nn ablation replay pipeline clean clean-all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PYTHON) -m pip install -r requirements.txt

data:  ## Generate synthetic data
	$(PYTHON) scripts/generate_synthetic_data.py --output-dir $(DATA_DIR)

verify:  ## Check the target reimplementation against the provided column
	$(PYTHON) scripts/verify_target.py --data-dir $(DATA_DIR)

test:  ## Run the unit tests
	$(PYTHON) -m pytest tests/ -v

train:  ## Cross-validate and save one model. Override: make train MODEL=ridge
	$(PYTHON) -m cryptoforecast.train --model $(MODEL) --data-dir $(DATA_DIR) \
		--output-dir $(ARTIFACTS)

train-nn:  ## Train the cross-asset neural network
	$(PYTHON) -m cryptoforecast.train --config configs/crossasset.yaml \
		--data-dir $(DATA_DIR) --output-dir $(ARTIFACTS)

ablation:  ## The six-way comparison (slow: includes two neural networks)
	$(PYTHON) -m cryptoforecast.train --ablation --data-dir $(DATA_DIR) \
		--output-dir $(ARTIFACTS)

replay:  ## Minute-by-minute replay of the inference path
	$(PYTHON) scripts/run_replay.py --artifacts $(ARTIFACTS) --name $(MODEL) \
		--data-dir $(DATA_DIR)

pipeline: data verify test train replay  ## Everything except the slow ablation

clean:  ## Remove caches and artifacts, keep the data
	rm -rf $(ARTIFACTS) .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean  ## Remove the data as well
	rm -rf $(DATA_DIR)
