#!/bin/bash

# python main.py --experiment synthetic --syn-sub circle --no-points 400 --data-noise 0.1 --sampler-noise 5.0 --no-runs 100
# python main.py --experiment synthetic --syn-sub spiral --no-points 1000 --data-noise 0.02 --sampler-noise 4.0 --no-runs 100
# python main.py --experiment synthetic --syn-sub line --no-points 100 --data-noise 0.1 --sampler-noise 2.0 --no-runs 100
# python main.py --experiment synthetic --syn-sub circle2 --no-points 200 --data-noise 0.01 --sampler-noise 2.5 --no-runs 100
# python main.py --experiment synthetic --syn-sub circle3 --no-points 200 --data-noise 0.01 --sampler-noise 2.5 --no-runs 100
# python main.py --experiment synthetic --syn-sub spiral2 --no-points 1000 --data-noise 0.02 --sampler-noise 4.0 --no-runs 100

# python main.py --experiment synthetic --syn-sub spiral3 --no-points 1000 --data-noise 0.02 --sampler-noise 4.0

python main.py --experiment synthetic --syn-sub circle --no-points 400 --data-noise 0.01 --sampler-noise 0.1 --no-runs 100
python main.py --experiment synthetic --syn-sub spiral --no-points 1000 --data-noise 0.002 --sampler-noise 0.1 --no-runs 100
python main.py --experiment synthetic --syn-sub line --no-points 100 --data-noise 0.01 --sampler-noise 0.1 --no-runs 100
python main.py --experiment synthetic --syn-sub circle2 --no-points 200 --data-noise 0.001 --sampler-noise 0.1 --no-runs 100
python main.py --experiment synthetic --syn-sub circle3 --no-points 200 --data-noise 0.001 --sampler-noise 0.1 --no-runs 100
python main.py --experiment synthetic --syn-sub spiral2 --no-points 1000 --data-noise 0.002 --sampler-noise 0.1 --no-runs 100
    