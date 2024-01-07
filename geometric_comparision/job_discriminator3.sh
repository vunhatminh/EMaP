#!/bin/bash

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    for radius in '0.0001' '0.00001'
    do
        python discriminator-image.py --exp mnist --radius $radius --base 0 --multiplier 100 --dim 2 --runs 100 --target $label --shuffle True
    done
done

for label in '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
do
    for radius in '0.0001' '0.00001'
    do
        python discriminator-image.py --exp fashion_mnist --radius $radius --base 0 --multiplier 100 --dim 3 --runs 100 --target $label --shuffle True
    done
done

    

    