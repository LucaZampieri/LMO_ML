# File run.py that should recreate the best score of Kaggle
# Ridge regression with the optimal parameters, after a suitable data pre-processing

# Authors: Ondine Chanon, Matteo Ciprian, Luca Zampieri
# license: MIT

# Import external libraries
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import datetime
import operator

# Import personal libraries
from helpers import *
from preprocessing_functions import *
from optimal_cross_validation import *
from optimal_ridge_regression import *


################## Loading the training and the testing sets ##################

DATA_FOLDER = '../data/' # Name of the input data folder

# Loading
print('Loading data...')
y_train, tx_train, ids_train = load_csv_data(DATA_FOLDER+'train.csv',sub_sample=False)
y_test, tx_test, ids_test = load_csv_data(DATA_FOLDER+'test.csv',sub_sample=False)


################## Cross validation ##################

# Array with the best degree for each spitted dataset (determined by the value of PRI_jet_num)
best_degree = np.array([10, 10, 11, 10]) 
# Array with the best lambda for each spitted dataset (determined by the value of PRI_jet_num)
best_lambda = np.array([1e-4, 1e-3, 1e-3, 1e-3])

# Uncomment this paragraph to have the best_degrees and best_lambdas directly from the cross validation
'''
# Set of degrees that we want to test for the cross validation
degrees = range(6,13) 
# Set of lambdas that we want to test for the cross validation
lambdas = np.logspace(-8,1,10)
# Number of k-fold for the cross validation
k_fold = 5
# Seed to generate reproducible random values
seed = 1

# Cross validation function for the ridge regression
best_degree, best_lambda, best_mean_accuracy_train, best_mean_accuracy_test = \
    cross_validation_all_jets_ridge_regression(degrees, lambdas, k_fold, seed, y_train, tx_train, tx_test)

# Print the best accuracies obtained as the mean of the accuracies on the training set, and on the testing 
# set (indipendently), during the cross validation.
print(best_mean_accuracy_train)
print(best_mean_accuracy_test)
'''

# Note: the values of the best parameters have been slightly modified with respect to the results of the
# cross validation in order to avoid instability of the results in the testing set.
print('Chosen degrees for each jet:', best_degree)
print('Chosen lambdas for each jet:', best_lambda)


################## Compute predictions ##################

print('Computing predictions...')
y_pred_train, y_pred_test, full_accuracy_train, = \
    ridge_regression_all_jets_pred(tx_train, tx_test, y_train, best_degree, best_lambda)

print('Accuracy in the training dataset: ', full_accuracy_train)

# Set the name of the output folder
OUTPUT_FOLDER = 'output/'
# Set the name of the prediction file
name = 'Prediction.csv'
# Generate the prediction file
create_csv_submission(ids_test, y_pred_test,OUTPUT_FOLDER + name)


################## END ##################
