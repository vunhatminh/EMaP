import warnings
warnings.filterwarnings('ignore') 

from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import pickle
import numpy as np
import pandas as pd
import argparse
import time

from copy import deepcopy

import manifold_sampling_tabular as mst

def arg_parse():
    parser = argparse.ArgumentParser(description="UMAP discriminator")
    parser.add_argument(
            "--exp", dest="exp", help="Experiments"
        )
    parser.add_argument(
            "--std", dest="std", type=float, help="Perturbation std"
        )
    parser.add_argument(
        "--multiplier", dest="multiplier", type=int, help="Number of times an image is perturbed"
    )
    parser.add_argument(
        "--perturbations", dest="num_perturbations", type=int, help="Number of perturbations"
    )
    parser.add_argument(
        "--dim", dest="dim", type=int, help="Number of low dim"
    )
    parser.add_argument(
        "--pivots", dest="pivots", type=int, help="Number of pivots"
    )
    parser.add_argument(
        "--shuffle", dest="shuffle", type=bool, help="Shuffle the pivots"
    )
    parser.add_argument(
            "--train_ratio", dest="train_ratio", type=float, help="ratio of training for the rf"
        )
    
        
    parser.set_defaults(
        exp = 'compass',
        std = 0.1,
        num_perturbations = 100,
        multiplier = 100,
        dim = 2,
        pivots = 20,
        shuffle = True,
        train_ratio = 0.9
    )
    return parser.parse_args()

prog_args = arg_parse()

EXPERIMENT = prog_args.exp
PERTURBATION_STD = prog_args.std
NUM_PERTURBATIONS = prog_args.num_perturbations
MULTIPLIER = prog_args.multiplier
DIM = prog_args.dim
PIVOTS = prog_args.pivots
TRAIN_RATIO = prog_args.train_ratio

print("EXPERIMENT: ", EXPERIMENT)
print("MULTIPLIER: ", MULTIPLIER)
print("PERTURBATION_STD: ", PERTURBATION_STD)
print("DIM: ", DIM)
print("PIVOTS: ", PIVOTS)

# Get the data set and do some preprocessing
params = Params("data/experiment_params.json")
np.random.seed(params.seed)
if EXPERIMENT == 'compass':
    X, y, cols = get_and_preprocess_compas_data(params)
elif EXPERIMENT == 'german':
    X, y, cols = get_and_preprocess_german(params)
elif EXPERIMENT == 'cc':
    X, y, cols = get_and_preprocess_cc(params)


print("Data shape: ", X.shape)

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size = 0.1)
ss = StandardScaler().fit(X)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)
xall = ss.transform(X)

start_time = time.time()
mani_sampler = mst.Manifold_Tabular_Sampler(xall, dim = DIM, labels = y)
duration = time.time() - start_time
print("Initialize duration: ", duration)

start_time = time.time()
mani_sampler.train_multiplier = MULTIPLIER
mani_sampler.std_train = PERTURBATION_STD
mani_sampler.train_pivot(no_pivots_per_label = PIVOTS, shuffle = False)
duration = time.time() - start_time
print("Train duration: ", duration)

def get_discriminator(X,y,n_estimators = 100, train_ratio = 0.5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1 - train_ratio)
    the_rf = RandomForestClassifier(n_estimators=n_estimators).fit(X_train, y_train)
    y_pred = the_rf.predict(X_test)
    the_rf_result = (y_pred == y_test).sum()
    return the_rf, the_rf_result/y_test.shape[0], X_train.shape[0]

def get_discriminator_performance(X,y,rf):
    y_pred = rf.predict(X)
    the_rf_result = (y_pred == y).sum()
    return the_rf_result/y.shape[0], y.shape[0]

X_in = mani_sampler.pivots
X_rawper = np.vstack([perturbs[0] for perturbs in mani_sampler.raw_perturbs])
X_per = np.vstack([perturbs[0] for perturbs in mani_sampler.perturbs])
X_plane = np.zeros_like(X_in)
X_ortho = np.zeros_like(X_in)
for i in range(X_plane.shape[0]):
    X_plane[i] = mani_sampler.pivots[i] + mani_sampler.plane_noise[i][0]
    X_ortho[i] = mani_sampler.pivots[i] + mani_sampler.ortho_noise[i][0]
    
X_discriminator_rawper = np.vstack((X_in, X_rawper))
X_discriminator_per = np.vstack((X_in, X_per))
X_discriminator_plane = np.vstack((X_in, X_plane))
X_discriminator_ortho = np.vstack((X_in, X_ortho))
y_discriminator = np.concatenate((np.zeros(X_in.shape[0]), np.ones(X_per.shape[0])))

the_rf_rawper, test_acc_rawper, no_trains = get_discriminator(X_discriminator_rawper,y_discriminator, n_estimators = 100, train_ratio = TRAIN_RATIO)
print(test_acc_rawper, no_trains)
the_rf_per, test_acc_per, no_trains = get_discriminator(X_discriminator_per,y_discriminator, n_estimators = 100, train_ratio = TRAIN_RATIO)
print(test_acc_per, no_trains)
the_rf_plane, test_acc_plane, no_trains = get_discriminator(X_discriminator_plane,y_discriminator, n_estimators = 100, train_ratio = TRAIN_RATIO)
print(test_acc_plane, no_trains)
the_rf_ortho, test_acc_ortho, no_trains = get_discriminator(X_discriminator_ortho,y_discriminator, n_estimators = 100, train_ratio = TRAIN_RATIO)
print(test_acc_ortho, no_trains)

print("Create Testing environment")

start_time = time.time()
explanation_sampler = mst.Manifold_Tabular_Sampler(xall, dim = DIM, labels = y)
duration = time.time() - start_time
print("Initialize duration: ", duration)

start_time = time.time()
explanation_sampler.train_multiplier = MULTIPLIER
explanation_sampler.std_train = PERTURBATION_STD
explanation_sampler.train_pivot(no_pivots_per_label = PIVOTS, shuffle = True)
duration = time.time() - start_time
print("Train duration: ", duration)

Z_in = explanation_sampler.pivots
acc_rawper = 0
acc_per = 0
acc_plane = 0
acc_ortho = 0
var_rawper = 0
var_per = 0 
var_plane = 0
var_ortho = 0
for p in range(NUM_PERTURBATIONS):
    Z_rawper = np.vstack([perturbs[0] for perturbs in explanation_sampler.raw_perturbs])
    Z_per = np.vstack([perturbs[p] for perturbs in explanation_sampler.perturbs])
    Z_plane = np.zeros_like(Z_in)
    Z_ortho = np.zeros_like(Z_in)
    for i in range(Z_plane.shape[0]):
        Z_plane[i] = explanation_sampler.pivots[i] + explanation_sampler.plane_noise[i][p]
        Z_ortho[i] = explanation_sampler.pivots[i] + explanation_sampler.ortho_noise[i][p]
    
    Z_discriminator_rawper = np.vstack((Z_in, Z_rawper))
    Z_discriminator_per = np.vstack((Z_in, Z_per))
    Z_discriminator_plane = np.vstack((Z_in, Z_plane))
    Z_discriminator_ortho = np.vstack((Z_in, Z_ortho))
    y_discriminator = np.concatenate((np.zeros(Z_in.shape[0]), np.ones(Z_per.shape[0])))
    
    t_acc_rawper, no_test = get_discriminator_performance(Z_discriminator_rawper, y_discriminator, the_rf_rawper)
    t_acc_per, no_test = get_discriminator_performance(Z_discriminator_per, y_discriminator, the_rf_per)
    t_acc_plane, no_test = get_discriminator_performance(Z_discriminator_plane, y_discriminator, the_rf_plane)
    t_acc_ortho, no_test = get_discriminator_performance(Z_discriminator_ortho, y_discriminator, the_rf_ortho)
    
    acc_rawper = acc_rawper + t_acc_rawper
    acc_per = acc_per + t_acc_per
    acc_plane = acc_plane + t_acc_plane
    acc_ortho = acc_ortho + t_acc_ortho
    var_rawper = var_rawper + np.var(Z_rawper-Z_in)
    var_per = var_per + np.var(Z_per-Z_in)
    var_plane = var_plane + np.var(Z_plane-Z_in)
    var_ortho = var_ortho + np.var(Z_ortho-Z_in)

acc_rawper = acc_rawper/NUM_PERTURBATIONS
acc_per = acc_per/NUM_PERTURBATIONS
acc_plane = acc_plane/NUM_PERTURBATIONS
acc_ortho = acc_ortho/NUM_PERTURBATIONS
var_rawper = var_rawper/NUM_PERTURBATIONS
var_per = var_per/NUM_PERTURBATIONS
var_plane = var_plane/NUM_PERTURBATIONS
var_ortho = var_ortho/NUM_PERTURBATIONS

df = pd.DataFrame({'rawper': [acc_rawper, test_acc_rawper, var_rawper],
                   'per': [acc_per, test_acc_per, var_per],
                   'plane': [acc_plane, test_acc_plane, var_plane],
                   'ortho': [acc_ortho, test_acc_ortho, var_ortho]})

discriminator_file = 'results/discriminator/' + EXPERIMENT + '_dim_' + str(DIM) + '_std_' + str(PERTURBATION_STD) +'_.pickle'
print("Save file to ", discriminator_file)
with open(discriminator_file, 'wb') as output:
    pickle.dump(df, output)
