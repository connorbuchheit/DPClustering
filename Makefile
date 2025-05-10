.PHONY: all test run-main run-insurance-plots

all: test
run-main:
	PYTHONWARNINGS=ignore PYTHONPATH=. python main.py
run-insurance-plots:
	PYTHONWARNINGS=ignore PYTHONPATH=. python tests/insurance.py
