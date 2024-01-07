#!/bin/bash

# for label in '0' '1'
# do
#     for radius in  '0.000001' 
#     do
#         python discriminator-tabular.py --exp german --radius $radius --alpha 0.0 --multiplier 100 --dim 3 --runs 100 --target $label --shuffle True
#     done
# done

for label in '0' '1'
do
    for radius in  '0.00001'
    do
        python discriminator-tabular.py --exp cc --radius $radius --alpha 0.0 --multiplier 100 --dim 3 --runs 100 --target $label --shuffle True
    done
done

for label in '0' '1'
do
    for radius in  '0.00001'
    do
        python discriminator-tabular.py --exp cc --radius $radius --alpha 0.0 --multiplier 100 --dim 2 --runs 100 --target $label --shuffle True
    done
done

    

    