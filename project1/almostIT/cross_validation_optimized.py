# -*- coding: utf-8 -*-

import numpy as np 
from helpers import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation_single_jet_single_param_ridge_regression(single_jet_y, single_jet_tx, k_fold, seed, \
                                                              single_degree, single_lambda):
    # get k'th subgroup in test, others in train
    k_indices = build_k_indices(single_jet_y, k_fold, seed)
    accuracy_train = np.zeros(k_fold)
    accuracy_test = np.zeros(k_fold)
    
    for k in range(k_fold):
        #print('----- FOLD', k, '-----')
        k_index = k_indices[k]
        test_y = single_jet_y[k_index]
        test_tx = single_jet_tx[k_index,:]
        
        mask = np.ones(len(single_jet_y), dtype=bool) # set all elements to True
        mask[k_index] = False              # set test elements to False
        train_tx = single_jet_tx[mask,:]              # select only True elements (ie train elements)
        train_y = single_jet_y[mask]
        
        accuracy_train[k], accuracy_test[k] = \
            cleaned_ridge_regression_pred(single_degree, single_lambda, train_tx, test_tx, train_y, test_y, \
                                          predictions=False)

return np.mean(accuracy_train), np.mean(accuracy_test), np.var(accuracy_train), np.var(accuracy_test), \
    np.min(accuracy_train), np.min(accuracy_test), np.max(accuracy_train), np.max(accuracy_test)






def cross_validation_single_jet_ridge_regression(degrees, lambdas, y_single_jet_train, tx_single_jet_train, \
                                                 k_fold, seed, returnAll=False):
    mean_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    mean_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    var_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    var_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    min_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    min_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    max_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    max_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    
    for i, single_degree in enumerate(degrees):
        print('!!!! DEGREE', single_degree, '!!!!')
        for j, single_lambda in enumerate(lambdas):
            print('--- LAMBDA', single_lambda, '---')
            mean_acc_cv_train[i,j], mean_acc_cv_test[i,j], var_acc_cv_train[i,j], var_acc_cv_test[i,j], \
                min_acc_cv_train[i,j], min_acc_cv_test[i,j], max_acc_cv_train[i,j], max_acc_cv_test[i,j] = \
                cross_validation_single_jet_single_param_ridge_regression(y_single_jet_train, tx_single_jet_train, \
                                                                          k_fold, seed, single_degree, single_lambda)

max_id_deg, max_id_lambda = np.unravel_index(mean_acc_cv_test.argmax(), mean_acc_cv_test.shape)
print('Best mean accuracy: ', mean_acc_cv_test[max_id_deg, max_id_lambda])
print('attained with degree =', degrees[max_id_deg], 'and lambda =', lambdas[max_id_lambda])
plot_accuracy_evolution(degrees, lambdas, mean_acc_cv_train, mean_acc_cv_test, var_acc_cv_train, var_acc_cv_test)

if returnAll == True:
    return degrees[max_id_deg], lambdas[max_id_lambda], mean_acc_cv_train, mean_acc_cv_test, var_acc_cv_train, \
        var_acc_cv_test, min_acc_cv_train, min_acc_cv_test, max_acc_cv_train, max_acc_cv_test
    else:
        return degrees[max_id_deg], lambdas[max_id_lambda], mean_acc_cv_train[max_id_deg, max_id_lambda], \
            mean_acc_cv_test[max_id_deg, max_id_lambda]





