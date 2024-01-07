#!/bin/bash

python explainer-image.py --exp mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

python explainer-image.py --exp mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

python explainer-image.py --exp mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

python explainer-image.py --exp mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000