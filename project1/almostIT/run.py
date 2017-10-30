# Authors:
# ondine.chanon@epfl.ch, matteo.ciprian@epfl.ch, luca.zampieri@epfl.ch
# MIT license, if not provided with this file, please visit 

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from implementations import * #our implementations of the functions done by us
from helpers import *   #the helper provided for the project and other helper functions
import datetime
import operator
from preprocessing_functions import * 
from ridge_regression_optimized import *
from cross_validation_optimized import *
####  function definition


###############################   data loading    ###############################


DATA_FOLDER = 'data/'

y_train, tx_train, ids_train = load_csv_data(DATA_FOLDER+'train.csv',sub_sample=False)

y_test, tx_test, ids_test = load_csv_data(DATA_FOLDER+'test.csv',sub_sample=False)



########################    cross validation  #################################


# Best parameters found with cross validation (not to run cross validation each time)
best_degree_per_jet = [9, 11, 12, 12]
best_lambda_per_jet = [  1.00000000e-08,   1.00000000e-03,   1.00000000e-02,   1.00000000e-02]


# Uncomment next paragraph to find the above optimal values
'''
degrees = range(6,16)
lambdas = np.logspace(-8,1,10)
k_fold = 10
seed = 1

best_degree_per_jet, best_lambda_per_jet, best_mean_accuracy_train, best_mean_accuracy_test = cross_validation_all_jets_ridge_regression(degrees, lambdas, k_fold, seed, y_train, tx_train, tx_test)
'''
print(best_degree_per_jet)
print(best_lambda_per_jet)


######################### Compute preditions ###################################


# Compute the prediction for the test dataset and the prediction of the accuracy
y_pred_train, y_pred_test, full_accuracy_train, = ridge_regression_all_jets_pred(tx_train, tx_test, y_train, best_degree_per_jet, best_lambda_per_jet)


# remove those lines?
print (full_accuracy_train)
print('Shapes are (for verification): ')
print(y_pred_test.shape)

print(y_pred_test[y_pred_test==-1].shape)
print(y_pred_test[y_pred_test==1].shape)

print(y_pred_test[0:200])


# Set name of the submission file
name = 'I_hope_a_good_score.csv'

# Create the submission file and put it in folder output
create_csv_submission(ids_test, y_pred_test, 'output/' + name)


# END


    


    
    
