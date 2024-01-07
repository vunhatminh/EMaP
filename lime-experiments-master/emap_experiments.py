import sys
import copy
sys.path.append('..')
import argparse
import umap
import explainers
import evaluate_explanations as Evaluator
import parzen_windows
import numpy as np
import pickle
import sklearn
from load_datasets import *
from sklearn.metrics import accuracy_score


def main():
  parser = argparse.ArgumentParser(description='Evaluate some explanations')
  parser.add_argument('--dataset', '-d', type=str, required=True,help='dataset name')
  parser.add_argument('--algorithm', '-a', type=str, required=True, help='algorithm name')
  parser.add_argument('--explainer', '-e', type=str, required=True, help='explainer name')
  parser.add_argument('--feat_only', '-f', type=bool, default=True, help='explaination features in inputs only')  
  parser.add_argument('--low_dim', '-l', type=bool, default=False, help='EMaP use low dimensional distances')  
  args = parser.parse_args()

  dataset = args.dataset
  algorithm = args.algorithm
  feat_only = args.feat_only
  low_dim = args.low_dim  

  evaluator = Evaluator.ExplanationEvaluator(classifier_names=[algorithm])
  evaluator.load_datasets([dataset])
  evaluator.vectorize_and_train()
  explain_fn = None
  
  print("Test method: ", args.explainer)
  if args.explainer == 'emap':
     print("Low dim? ", low_dim)

  if args.explainer == 'lime':
    rho = 25
    kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
    explainer = explainers.GeneralizedLocalExplainer(kernel, explainers.data_labels_distances_mapping_text, num_samples=15000, return_mean=False, verbose=False, return_mapped=True)
    explain_fn = explainer.explain_instance
  elif args.explainer == 'parzen':
    sigmas = {'multi_polarity_electronics': {'tree': 0.5,
    'l1logreg': 1},
    'multi_polarity_kitchen': {'tree': 0.75, 'l1logreg': 2.0},
    'multi_polarity_dvd': {'tree': 8.0, 'l1logreg': 1},
    'multi_polarity_books': {'tree': 2.0, 'l1logreg': 2.0}}

    explainer = parzen_windows.ParzenWindowClassifier()
    cv_preds = sklearn.model_selection.cross_val_predict(evaluator.classifiers[dataset][algorithm], evaluator.train_vectors[dataset], evaluator.train_labels[dataset])
    explainer.fit(evaluator.train_vectors[dataset], cv_preds)
    explainer.sigma = sigmas[dataset][algorithm]
    explain_fn = explainer.explain_instance
  elif args.explainer == 'greedy':
    explain_fn = explainers.explain_greedy
  elif args.explainer == 'random':
    explainer = explainers.RandomExplainer()
    explain_fn = explainer.explain_instance
  elif args.explainer == 'emap':
    rho = 25
    kernel = lambda d: np.sqrt(np.exp(-(d**2) / rho ** 2))
    explainer = explainers.EMaPLocalExplainer(kernel, 
                                        base_data = evaluator.train_vectors[dataset].todense(),
                                        clf_fn = evaluator.classifiers[dataset][algorithm],
                                        low_distance = low_dim,           
                                        radius = 1,
                                        return_mean=False, 
                                        verbose=False, 
                                        return_mapped=False)
    explain_fn = explainer.explain_instance
    
  recalls = []
  false_positives = []
  for budget in range(1,21):
    print("Number of return feats: ", budget)
    test_recall, test_fp = evaluator.measure_explanation(explain_fn, method = args.explainer, budget = budget, feat_only = feat_only)
    recalls.append(test_recall)
    false_positives.append(test_fp)
    print('Recall: ', np.mean(test_recall[dataset][algorithm]))
    print('FP: ', np.mean(test_fp[dataset][algorithm]))
    
  filename = 'results/' + dataset + "_" + args.explainer + "_"
  if args.explainer == 'emap':
        if args.low_dim == True:
            filename = filename + "low_"
        else:
            filename = filename + "base_"
  with open(filename + "recalls", 'wb') as outfile:
    pickle.dump(recalls, outfile)
  with open(filename + "fps", 'wb') as outfile:
    pickle.dump(false_positives, outfile)  
  
if __name__ == "__main__":
    main()
