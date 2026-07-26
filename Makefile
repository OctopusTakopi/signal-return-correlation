PY := python3
LAKE := $(or $(shell command -v lake 2>/dev/null),$(HOME)/.elan/bin/lake)

.PHONY: all verify figures test lean calibrate clean help

all: verify figures test

help:
	@echo "make verify        symbolic checks, worked examples, Monte Carlo,"
	@echo "                   counterexamples, applications, regression"
	@echo "make figures       report figures from results/*.json"
	@echo "make test          unit tests on the closed forms"
	@echo "make lean          formally verify the deterministic algebra with Lean"
	@echo "make calibrate     re-measure the market parameters; needs"
	@echo "                   BINANCE_DATA_DIR to name the kline archive"
	@echo "make all           verify, figures, test"

verify:
	$(PY) scripts/check_algebra.py
	$(PY) scripts/check_examples.py
	$(PY) scripts/simulate_bound.py
	$(PY) scripts/counterexamples.py
	$(PY) scripts/crypto_market_making.py
	$(PY) scripts/vip0_fee_analysis.py
	$(PY) scripts/cross_sectional_ic.py
	$(PY) scripts/r2_regression.py
	$(PY) scripts/gbm_r2.py

# Separate from `verify` because it is the only step that needs an external
# data archive; everything else is self-contained.
calibrate:
	$(PY) scripts/calibrate_from_data.py

figures:
	$(PY) scripts/figures.py

test:
	$(PY) tests/test_engine.py

lean:
	$(LAKE) build

clean:
	rm -rf results figures/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
