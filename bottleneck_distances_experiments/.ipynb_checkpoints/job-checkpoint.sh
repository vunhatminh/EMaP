#!/bin/bash

for label in '0' '1'
do
    for radius in '1.0'
    do
        python point_clouds_persistence_diagram-tabular.py --exp compass --radius $radius --multiplier 100 --dim 2 --runs 100 --target $label --shuffle True
        python point_clouds_persistence_diagram-tabular.py --exp german --radius $radius --multiplier 100 --dim 2 --runs 100 --target $label --shuffle True
        python point_clouds_persistence_diagram-tabular.py --exp cc --radius $radius --multiplier 100 --dim 2 --runs 100 --target $label --shuffle True
    done
done


    