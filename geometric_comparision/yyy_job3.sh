#!/bin/bash

# for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
# do
#     for radius in '0.001' '0.01' '0.1' '1.0'
#     do
#         python point_clouds_persistence_diagram.py --exp mnist --radius $radius --multiplier 500 --dim 3 --runs 10 --target $label --shuffle True
#     done
# done

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    for radius in '0.001' '0.01' '0.1' '1.0'
    do
        python point_clouds_persistence_diagram.py --exp fashion_mnist --radius $radius --multiplier 500 --dim 3 --runs 10 --target $label --shuffle True
    done
done




    