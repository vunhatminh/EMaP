#!/bin/bash


for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    for radius in '0.1' '1.0'
    do
        python point_clouds_persistence_diagram-image.py --exp mnist --radius $radius --multiplier 1000 --dim 2 --runs 10 --target $label --shuffle True
    done
done

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    for radius in '0.002' '0.005'
    do
        python point_clouds_persistence_diagram-image.py --exp mnist --radius $radius --multiplier 1000 --dim 3 --runs 10 --target $label --shuffle True
    done
done
