.PHONY: all test run-main run-insurance-plots run-prediction-example

all: test
test:
	PYTHONWARNINGS=ignore PYTHONPATH=. python tests/_.py
	PYTHONWARNINGS=ignore PYTHONPATH=. python tests/experiments.py
run-main:
	PYTHONWARNINGS=ignore PYTHONPATH=. python main.py
run-insurance-plots:
	PYTHONWARNINGS=ignore PYTHONPATH=. python tests/insurance.py
run-prediction-example:
	PYTHONWARNINGS=ignore PYTHONPATH=. python tests/predict_insurance.py