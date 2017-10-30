#%matplotlib inline 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from implementations import * #our implementations of the functions done by us
from helpers import * #  #the helper provided for the project and other helper functions
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

#degrees = range(6,16)
#lambdas = np.logspace(-8,1,10)
#k_fold = 10
#seed = 1

#best_degree, best_lambda, best_mean_accuracy_train, best_mean_accuracy_test = cross_validation_all_jets_ridge_regression(degrees, lambdas, k_fold, seed, y_train, tx_train, tx_test)





best_degree_per_jet = [9, 11, 12, 12]
best_lambda_per_jet = [  1.00000000e-08,   1.00000000e-03,   1.00000000e-02,   1.00000000e-02]



print(best_degree_per_jet)
print(best_lambda_per_jet)



y_pred_train, y_pred_test, full_accuracy_train, = ridge_regression_all_jets_pred(tx_train, tx_test, y_train, best_degree_per_jet, best_lambda_per_jet)
    
print (full_accuracy_train)


print('Shapes are (for verification): ')
print(y_pred_test.shape)

print(y_pred_test[y_pred_test==-1].shape)
print(y_pred_test[y_pred_test==1].shape)



print(y_pred_test[0:200])



name = 'output/ridge_regression_ondine_splitjet_test.csv'
create_csv_submission(ids_test, y_pred_test, name)


    


    
    
