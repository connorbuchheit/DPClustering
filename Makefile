.PHONY: all test run-main run-insurance-plots run-prediction-example

# Default target
all: test

# Run general tests and experiments
test:
	PYTHONPATH=. python tests/_.py
	PYTHONPATH=. python tests/experiments.py

# Run the main synthetic data example
run-main:
	PYTHONPATH=. python main.py

# Run insurance-specific experiments and generate plots (like in the report)
run-insurance-plots:
	PYTHONPATH=. python tests/insurance.py

# Run the insurance prediction example
run-prediction-example:
	PYTHONPATH=. python tests/predict_insurance.py