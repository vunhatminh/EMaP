#!/bin/bash

# python umap_discriminator.py --exp fashion_mnist --std 0.01 --n 100 --runs 100 --dim 4
# python umap_discriminator.py --exp fashion_mnist --std 0.1 --n 100 --runs 100 --dim 4
# python umap_discriminator.py --exp fashion_mnist --std 1.0 --n 100 --runs 100 --dim 4
# python umap_discriminator.py --exp fashion_mnist --std 5.0 --n 100 --runs 100 --dim 4

# python umap_discriminator.py --exp mnist --std 0.01 --n 100 --runs 100 --dim 4
# python umap_discriminator.py --exp mnist --std 0.1 --n 100 --runs 100 --dim 4
# python umap_discriminator.py --exp mnist --std 1.0 --n 100 --runs 100 --dim 4
# python umap_discriminator.py --exp mnist --std 5.0 --n 100 --runs 100 --dim 4

python compass_discriminator.py --exp compass --std 0.00001 --dim 2
python compass_discriminator.py --exp compass --std 0.000001 --dim 2
python compass_discriminator.py --exp compass --std 0.0000001 --dim 2

python compass_discriminator.py --exp german --std 0.00001 --dim 2
python compass_discriminator.py --exp german --std 0.000001 --dim 2
python compass_discriminator.py --exp german --std 0.0000001 --dim 2

# python compass_discriminator.py --exp german --std 0.01 --dim 3
# python compass_discriminator.py --exp german --std 0.1 --dim 3
# python compass_discriminator.py --exp german --std 1.0 --dim 3

python compass_discriminator.py --exp cc --std 0.00001 --dim 2
python compass_discriminator.py --exp cc --std 0.000001 --dim 2
python compass_discriminator.py --exp cc --std 0.0000001 --dim 2

# python compass_discriminator.py --exp cc --std 0.01 --dim 4
# python compass_discriminator.py --exp cc --std 0.1 --dim 4
# python compass_discriminator.py --exp cc --std 1.0 --dim 4

# python compass_discriminator.py --exp compass --std 0.01 --dim 2
# python compass_discriminator.py --exp compass --std 0.1 --dim 2
# python compass_discriminator.py --exp compass --std 1.0 --dim 2

python manifold_discriminator.py --exp fashion_mnist --std 0.0001 --dim 2
python manifold_discriminator.py --exp fashion_mnist --std 0.00001 --dim 2
python manifold_discriminator.py --exp fashion_mnist --std 0.000001 --dim 2

# python manifold_discriminator.py --exp fashion_mnist --std 0.01 --dim 3
# python manifold_discriminator.py --exp fashion_mnist --std 0.1 --dim 3
# python manifold_discriminator.py --exp fashion_mnist --std 0.2 --dim 3
# python manifold_discriminator.py --exp fashion_mnist --std 1.0 --dim 3

# python manifold_discriminator.py --exp fashion_mnist --std 0.01 --dim 4
# python manifold_discriminator.py --exp fashion_mnist --std 0.1 --dim 4
# python manifold_discriminator.py --exp fashion_mnist --std 0.2 --dim 4
# python manifold_discriminator.py --exp fashion_mnist --std 1.0 --dim 4

python manifold_discriminator.py --exp mnist --std 0.0001 --dim 2
python manifold_discriminator.py --exp mnist --std 0.00001 --dim 2
python manifold_discriminator.py --exp mnist --std 0.000001 --dim 2

# python manifold_discriminator.py --exp mnist --std 0.01 --dim 4
# python manifold_discriminator.py --exp mnist --std 0.1 --dim 4
# python manifold_discriminator.py --exp mnist --std 1.0 --dim 4



