#!/bin/bash

# run experiments of StackedELM
python run_experiment.py --model StackedELM

# run experiments of PLN
python run_experiment.py --model PLN

# run experments of PMLP (both original and proposal)
python run_experiment.py --model PMLP
