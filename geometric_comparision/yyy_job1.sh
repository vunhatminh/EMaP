#!/bin/bash

# for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
# do
#     for radius in '0.001' '0.002' '0.005' '0.01'
#     do
#         python point_clouds_persistence_diagram.py --exp mnist --radius $radius --multiplier 1000 --dim 2 --runs 10 --target $label --shuffle True
#     done
# done

# python point_clouds_persistence_diagram.py --exp fashion_mnist --radius 0.01 --multiplier 500 --dim 2 --runs 10 --target 0 --shuffle True

# for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
# do
#     for radius in '0.001' '0.01' '0.1' '1.0'
#     do
#         python point_clouds_persistence_diagram.py --exp fashion_mnist --radius $radius --multiplier 500 --dim 2 --runs 10 --target $label --shuffle True
#     done
# done

for label in '0' '1'
do
    for radius in '0.0001' '0.00001' '0.000001'
    do
        python point_clouds_persistence_diagram-tabular.py --exp compass --radius $radius --multiplier 100 --dim 2 --runs 100 --target $label --shuffle True
    done
done
    