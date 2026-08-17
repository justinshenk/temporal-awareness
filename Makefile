.PHONY: install test paper-a paper-b clean help

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make install   Install dependencies"
	@echo "  make test      Run the CPU test suite (no GPU, no network)"
	@echo "  make paper-a   Build the register-vs-procedure abstract PDF"
	@echo "  make paper-b   Build the context-fatigue paper PDF"
	@echo "  make clean     Remove caches"
	@echo ""

install:
	uv sync

test:
	uv run pytest tests/ -q

paper-a:
	cd papers/register_vs_procedure/abstract && tectonic register_vs_procedure_abstract.tex

paper-b:
	cd context_fatigue_paper && tectonic context_fatigue.tex

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.egg-info .pytest_cache
