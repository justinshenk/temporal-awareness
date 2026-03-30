.PHONY: install verify verify-quick verify-probes verify-steering figures eap-ig-workflow clean help

# Default target
help:
	@echo "Temporal Awareness - Available commands:"
	@echo ""
	@echo "  make install        Install dependencies"
	@echo "  make verify         Verify all claims (requires GPU)"
	@echo "  make verify-quick   Quick verification (cached results)"
	@echo "  make figures        Generate all figures"
	@echo "  make eap-ig-workflow   Run the EAP-IG workflow"
	@echo "  make clean          Remove generated files"
	@echo ""

# Install
install:
	pip install -e .

# Verification
verify:
	python scripts/verify_all_claims.py --gpu

verify-quick:
	python scripts/verify_all_claims.py --quick

verify-probes:
	python scripts/probes/train_temporal_probes_caa.py --eval-only

verify-steering:
	python scripts/probes/validate_dataset_split.py

# Figures
figures:
	jupyter nbconvert --execute --to notebook notebooks/01_reproduce_main_results.ipynb
	@echo "Figures saved to results/figures/"

eap-ig-workflow:
	python scripts/experiments/eap_ig/run_eap_ig_workflow.py --top-n 500

# Clean
clean:
	rm -rf results/figures/*.png
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
