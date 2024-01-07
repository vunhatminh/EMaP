#!/bin/bash

# python explainer-image.py --exp mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --lime True --no_samples 1000

# python explainer-image.py --exp fashion_mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --lime True --no_samples 1000



# python explainer-image-lime.py --exp mnist  --shuffle True  --no_samples 1000 --sigma 1.0

# python explainer-image-emap.py --exp mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --no_samples 1000

# python explainer-image-emap.py --exp mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --no_samples 1000

# python explainer-image-emap.py --exp mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --no_samples 1000

# python explainer-image-emap.py --exp mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --no_samples 1000

# python explainer-image-lime.py --exp fashion_mnist  --shuffle True  --no_samples 1000 --sigma 1.0

python explainer-image-emap.py --exp fashion_mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --no_samples 1000

python explainer-image-emap.py --exp fashion_mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 2 --no_perturbations 10 --shuffle True --no_samples 1000

python explainer-image-emap.py --exp fashion_mnist --radius 0.0001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --no_samples 1000

python explainer-image-emap.py --exp fashion_mnist --radius 0.00001 --sigma 1.0 --multiplier 100 --dim 3 --no_perturbations 10 --shuffle True --no_samples 1000

# python explainer-image-others.py --exp mnist  --shuffle True  --no_samples 1000

# python explainer-image-others.py --exp fashion_mnist  --shuffle True  --no_samples 1000

# python explainer-image-lime.py --exp mnist  --shuffle True  --no_samples 1000 --sigma 1.0 --fidnoise 1.0

# python explainer-image-lime.py --exp fashion_mnist  --shuffle True  --no_samples 1000 --sigma 1.0 --fidnoise 1.0
