#!/bin/bash

for label in '0' '1'
do
    for radius in  '0.0001' 
    do
        python discriminator-tabular-more.py --exp german --radius $radius --alpha 0.0 --multiplier 100 --runs 100 --target $label --shuffle True
        python discriminator-tabular-more.py --exp cc --radius $radius --alpha 0.0 --multiplier 100 --runs 100 --target $label --shuffle True
    done
done

    