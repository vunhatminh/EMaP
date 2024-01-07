#!/bin/bash

# python emap_experiments.py --dataset 'multi_polarity_books' --algorithm 'l1logreg' --explainer 'greedy' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_books' --algorithm 'l1logreg' --explainer 'random' --feat_only True
python emap_experiments.py --dataset 'multi_polarity_books' --algorithm 'l1logreg' --explainer 'parzen' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_books' --algorithm 'l1logreg' --explainer 'lime' --feat_only True

# python emap_experiments.py --dataset 'multi_polarity_dvd' --algorithm 'l1logreg' --explainer 'greedy' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_dvd' --algorithm 'l1logreg' --explainer 'random' --feat_only True
python emap_experiments.py --dataset 'multi_polarity_dvd' --algorithm 'l1logreg' --explainer 'parzen' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_dvd' --algorithm 'l1logreg' --explainer 'lime' --feat_only True

# python emap_experiments.py --dataset 'multi_polarity_electronics' --algorithm 'l1logreg' --explainer 'greedy' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_electronics' --algorithm 'l1logreg' --explainer 'random' --feat_only True
python emap_experiments.py --dataset 'multi_polarity_electronics' --algorithm 'l1logreg' --explainer 'parzen' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_electronics' --algorithm 'l1logreg' --explainer 'lime' --feat_only True

# python emap_experiments.py --dataset 'multi_polarity_kitchen' --algorithm 'l1logreg' --explainer 'greedy' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_kitchen' --algorithm 'l1logreg' --explainer 'random' --feat_only True
python emap_experiments.py --dataset 'multi_polarity_kitchen' --algorithm 'l1logreg' --explainer 'parzen' --feat_only True
# python emap_experiments.py --dataset 'multi_polarity_kitchen' --algorithm 'l1logreg' --explainer 'lime' --feat_only True


