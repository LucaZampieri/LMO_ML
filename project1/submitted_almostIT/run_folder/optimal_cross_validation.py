# Authors: Ondine Chanon, Matteo Ciprian, Luca Zampieri
# license: MIT

# Implementation of the functions useful for the cross validation of the parameters of the ridge regression

# Import libraries
import numpy as np 
from helpers import *
from preprocessing_functions import *


def plot_accuracy_evolution(degrees, lambdas, mean_acc_cv_train, mean_acc_cv_test, var_acc_cv_train, \
                            var_acc_cv_test):
    """
	Plots the evolution of the accuracies of the training and testing sets during the cross validation
	for the different chosen values of degrees and lambdas. 
	
	Input: 
	     - degrees: array of maximal polynomial degrees on which the cross-validation is tested
	     - lambdas: array of lambdas on which the cross validation is tested
	     - mean_acc_cv_train: array of mean accuracies on the training sets
	     - mean_acc_cv_test: array of mean accuracies on the testing sets
	     - var_acc_cv_train: array of variances of the accuracies on the training
	     - var_acc_cv_test: array of variances of the accuracies on the testing
    """
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

    # ---- First graph with respect to lambda ----
    ax = axes[0]
    nb_deg = len(degrees)
    colors = iter(cm.rainbow(np.linspace(0, 1, nb_deg)))
    
    for idx in range(nb_deg):
        c = next(colors)
        ax.semilogx(lambdas, mean_acc_cv_train[idx,:], linestyle="--", color=c, \
                         label='train, deg='+str(degrees[idx]))
        ax.semilogx(lambdas, mean_acc_cv_test[idx,:], linestyle="-", color=c, \
                         label='test, deg='+str(degrees[idx]))
        
    ax.legend()
    ax.set_xlabel("lambda")
    ax.set_ylabel("accuracy")
    ax.grid(True)
    
    # ---- Second graph with respect to degree ----
    ax = axes[1]
    nb_lambda = len(lambdas)
    colors2 = iter(cm.rainbow(np.linspace(0, 1, nb_lambda)))
    
    for idx in range(nb_lambda):
        c = next(colors2)
        ax.plot(degrees, mean_acc_cv_train[:,idx], linestyle="--", color=c, \
                         label='train, lambda='+str(lambdas[idx]))
        ax.plot(degrees, mean_acc_cv_test[:,idx], linestyle="-", color=c, \
                         label='test, lambda='+str(lambdas[idx]))
    
    ax.set_xlabel("degree")
    ax.set_ylabel("accuracy")
    ax.grid(True)
    ax.legend()
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    plt.show()



def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold.
       Input:
	    - y: vector of output values
	    - k_fold: int, number of k_folds of the cross validation
	    - seed: int, random seed used for the reproducibility of the data
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)




def cross_validation_single_jet_single_param_ridge_regression(jet_id, single_jet_y, single_jet_tx, k_fold, seed, \
                                                              single_degree, single_lambda):
    """
    Splits the dataset into folds and compute the accuracy of the ridge regression on each fold.
    Input: 
	 - jet_id: int, considered jet number (influences the data pre-processing)
	 - single_jet_y: array of training outputs on a single jet
	 - single_jet_tx: array or matrix of input data on a single jet
	 - k_fold: int, number of folds for the cross validation
	 - seed: int, random seed used for the reproducibility of the data
	 - single_degree: int, maximum polynomial degree
	 - single_lambda: int, stabilizing parameter of the ridge regression
    Ouput: 
	 - mean accuracy on the training set
	 - mean accuracy on the testing set
	 - variance on the training set
	 - variance on the testing set
	 - minimum accuracy on the training set
	 - minimum accuracy on the testing set
	 - maximum accuracy on the training set
	 - maximum accuracy on the testing set
    """
    # get k'th subgroup in test, others in train
    k_indices = build_k_indices(single_jet_y, k_fold, seed)
    accuracy_train = np.zeros(k_fold)
    accuracy_test = np.zeros(k_fold)
    
    # Loop over the different folds for the cross validation
    for k in range(k_fold):
        print('----- FOLD', k, '-----')
	# Select the testing dataset
        k_index = k_indices[k]
        test_y = single_jet_y[k_index]
        test_tx = single_jet_tx[k_index,:]
	
	# Select the training dataset
        mask = np.ones(len(single_jet_y), dtype=bool) # set all elements to True
        mask[k_index] = False              	      # set test elements to False
        train_tx = single_jet_tx[mask,:]              # select only True elements (ie train elements)
        train_y = single_jet_y[mask]

	# Compute the ridge regression weights, predictions, and ouput the accuracy on the training and testing sets
	# corresponding to the actual fold.
        accuracy_train[k], accuracy_test[k] = \
            cleaned_ridge_regression_pred(jet_id, single_degree, single_lambda, train_tx, test_tx, train_y, test_y, \
                                          predictions=False)
            
    return np.mean(accuracy_train), np.mean(accuracy_test), np.var(accuracy_train), np.var(accuracy_test), \
           np.min(accuracy_train), np.min(accuracy_test), np.max(accuracy_train), np.max(accuracy_test)



def cross_validation_single_jet_ridge_regression(jet_id, degrees, lambdas, y_single_jet_train, tx_single_jet_train, \
                                                 k_fold, seed, returnAll=False):
    """
    On a single jet, compute the accuracy of the ridge regression for each chosen degree and lambda and returns the
    best values of degree and lambda for each jet number, and the corresponding training and testing accuracies.

    Input: 
	 - jet_id: int, considered jet number (influences the data pre-processing)
	 - degrees: array of maximal polynomial degrees on which the cross-validation is tested
	 - lambdas: array of lambdas on which the cross validation is tested
	 - y_single_jet_train: array of training ouput values on a single jet
	 - tx_single_jet_train: array or matrix of training input data on a single jet
	 - k_fold: int, number of folds used in the cross validation
	 - seed: int, random seed used for the reproducibility of the data
	 - returnAll: bool, if we want to return all statistical values on the accuracies.
    Ouput: 
	 - best degree
	 - best lambda
	 - (best) mean accuracy on the testing sets from the cross validation
	 - (best) mean accuracy on the training sets from the cross validation
	 If returnAll==True:
	 - variances on the training set
	 - variances on the testing set
	 - minima accuracy on the training set
	 - minima accuracy on the testing set
	 - maxima accuracy on the training set
	 - maxima accuracy on the testing set
    """
    # Initialize output variables
    mean_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    mean_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    var_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    var_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    min_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    min_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    max_acc_cv_train = np.zeros([len(degrees), len(lambdas)])
    max_acc_cv_test = np.zeros([len(degrees), len(lambdas)])
    
    # Loop over the chosen degrees
    for i, single_degree in enumerate(degrees):
        print('!!!! DEGREE', single_degree, '!!!!')
	# Loop over the chosen lambda values
        for j, single_lambda in enumerate(lambdas):
            print('--- LAMBDA', single_lambda, '---')
	    # Cross validation set on a specific degree and lambda for a single jet
            mean_acc_cv_train[i,j], mean_acc_cv_test[i,j], var_acc_cv_train[i,j], var_acc_cv_test[i,j], \
                min_acc_cv_train[i,j], min_acc_cv_test[i,j], max_acc_cv_train[i,j], max_acc_cv_test[i,j] = \
                cross_validation_single_jet_single_param_ridge_regression(jet_id, y_single_jet_train, tx_single_jet_train, \
                                                                          k_fold, seed, single_degree, single_lambda)

    # Compute the indices of the maximum testing accuracy and find the corresponding degree and lambda
    max_id_deg, max_id_lambda = np.unravel_index(mean_acc_cv_test.argmax(), mean_acc_cv_test.shape)
    print('Best mean accuracy: ', mean_acc_cv_test[max_id_deg, max_id_lambda])
    print('attained with degree =', degrees[max_id_deg], 'and lambda =', lambdas[max_id_lambda])
    # Plot the evolution of the accuracy for the different degrees and lambdas
    plot_accuracy_evolution(degrees, lambdas, mean_acc_cv_train, mean_acc_cv_test, var_acc_cv_train, var_acc_cv_test)

    if returnAll == True:
        return degrees[max_id_deg], lambdas[max_id_lambda], mean_acc_cv_train, mean_acc_cv_test, var_acc_cv_train, \
               var_acc_cv_test, min_acc_cv_train, min_acc_cv_test, max_acc_cv_train, max_acc_cv_test
    else:
        return degrees[max_id_deg], lambdas[max_id_lambda], mean_acc_cv_train[max_id_deg, max_id_lambda], \
               mean_acc_cv_test[max_id_deg, max_id_lambda]



def cross_validation_all_jets_ridge_regression(degrees, lambdas, k_fold, seed, full_y_train, full_tx_train, \
                                               full_tx_test):
    """
    Separate the full dataset into jets and perform the cross validation on the parameters of the ridge regression 
    on each jet. 

    Input: 
         - degrees: array of maximal polynomial degrees on which the cross-validation is tested
	 - lambdas: array of lambdas on which the cross validation is tested
	 - k_fold: int, number of folds used in the cross validation
	 - seed: int, random seed used for the reproducibility of the data
	 - full_y_train: array of training ouput values
	 - full_tx_train: array or matrix of training input data
	 - full_tx_test: array or matrix of testing input data
    Ouput: 
	 - best_degree: array of int, best maximum polynomial degree for each jet
	 - best_lambdas: array, best lambda values for each jet for the ridge regression
	 - best_full_accuracy_train: best accuracy on the whole training dataset (ie for all jets)
	 - best_full_accuracy_test: best accuracy on the whole testing dataset (ie for all jets)
    """
    # Create the masks to find the sub-datasets corresponding to the jet id.
    mask_jets_train = split_jets_mask(full_tx_train)
    mask_jets_test = split_jets_mask(full_tx_test)
    
    # Initialize ouput and useful variables
    len_mask = len(mask_jets_train)
    best_degree = np.zeros(len_mask)
    best_lambda = np.zeros(len_mask)
    best_acc_train = np.zeros(len_mask)
    best_acc_test = np.zeros(len_mask)
    len_jets_train = np.zeros(len_mask)
    len_jets_test = np.zeros(len_mask)

    # Loop over the jet numbers
    for mask_jet_id in range(len_mask):
        print('********** Jet ', mask_jet_id, '***********')
	# Select the right training and testing datasets (corresponding to the jet mask_jet_id)
        tx_single_jet_train = full_tx_train[mask_jets_train[mask_jet_id]]
        tx_single_jet_test = full_tx_test[mask_jets_test[mask_jet_id]]
        y_single_jet_train = full_y_train[mask_jets_train[mask_jet_id]]
        len_jets_train[mask_jet_id] = len(y_single_jet_train)
        len_jets_test[mask_jet_id] = tx_single_jet_test.shape[0]
        
	# Perform the cross validation on the chosen dataset corresponding to the jet mask_jet_id
        best_degree[mask_jet_id], best_lambda[mask_jet_id], best_acc_train[mask_jet_id], best_acc_test[mask_jet_id] = \
            cross_validation_single_jet_ridge_regression(mask_jet_id, degrees, lambdas, y_single_jet_train, tx_single_jet_train, \
                                                         k_fold, seed, returnAll=False)
    # Compute the accuracy of the training and testing full datasets.
    best_degree = best_degree.astype(int)
    best_full_accuracy_train = \
        np.sum([best_acc_train[id]*len_jets_train[id] for id in range(len_mask)])/len(full_y_train)
    best_full_accuracy_test = \
        np.sum([best_acc_test[id]*len_jets_test[id] for id in range(len_mask)])/full_tx_test.shape[0]
        
    return best_degree, best_lambda, best_full_accuracy_train, best_full_accuracy_test




