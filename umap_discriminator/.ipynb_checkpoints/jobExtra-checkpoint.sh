#!/bin/bash

python manifold_discriminator.py --exp fashion_mnist --std 0.01 --dim 3
python manifold_discriminator.py --exp fashion_mnist --std 0.1 --dim 3
python manifold_discriminator.py --exp fashion_mnist --std 1.0 --dim 3

python manifold_discriminator.py --exp fashion_mnist --std 0.2 --dim 2
python manifold_discriminator.py --exp fashion_mnist --std 0.2 --dim 3
python manifold_discriminator.py --exp fashion_mnist --std 0.2 --dim 4


