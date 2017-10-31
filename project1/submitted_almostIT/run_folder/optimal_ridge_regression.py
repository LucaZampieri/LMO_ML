# File optimal_ridge_regression.py computing the predictions using a ridge regression and returning
# the accuracy on the training set (i.e. the percentage of correctly predicted values on the training
# set. 

# Authors: Ondine Chanon, Matteo Ciprian, Luca Zampieri
# license: MIT

# Import external libraries
import numpy as np 
from numpy import matlib

# Import personal libraries
from helpers import *
from preprocessing_functions import *
import costs


def ridge_regression(y, tx, lambda_, fct='none'):
    """Computes the best weights and the result of the loss function obtained by ridge regression.
       
       Input: 
	    - y: array of output values.
	    - tx: array or matrix of data.
	    - lambda_: scalar, stabilizing parameter.
	    - fct: (optional parameter) string specifying the type of loss function, if the user wants
		   to compute it. It can be either 'mse' or 'rmse'. By default='none', i.e. no loss is
		   explicitely computed. 
       Ouput: 
	    - ridge: scalar, value of the loss if fct is specified.
	    - wstar: array of weights. 
    """
    wstar = np.linalg.solve(tx.T.dot(tx)+2*len(y)*lambda_*np.matlib.identity(tx.shape[1]),(tx.T).dot(y))
    if (fct=='mse' or fct=='rmse'):
        ridge = costs.compute_ridge_loss(y,tx,wstar,lambda_,fct)
        return ridge, wstar
    else: #'none'
        return wstar
    

def cleaned_ridge_regression_pred(jet_id, single_degree, single_lambda, single_train_tx, single_test_tx, \
                                  single_train_y, single_test_y=[], predictions=True):
    """ Computes the predictions of a single jet number with the ridge regression, given a degree (maximum 
	polynomial degree) and a lambda (stabilizing parameter).

	Input: 
	     - jet_id: int, considered jet number (influences the data pre-processing)
	     - single_degree: int, maximum polynomial degree
	     - single_lambda: int, stabilizing parameter of the ridge regression
	     - single_train_tx: matrix of training set (non preprocessed)
	     - single_test_tx: matrix of testing set (non preprocessed)
	     - single_train_y: array of ouput values for the training set
	     - single_test_y: (optional) array of ouput values for the testing set
	     - predictions: boolean, True if we want to output the predictions
	Ouput: 
	     - y_pred_train: array of predicted values in the training set, if predictions=True
	     - y_pred_test: array of predicted values in the testing set, if predictions=True
	     - accuracy_train: scalar, accuracy on the training set
	     - accuracy_test: scalar, accuracy on the testing set (if single_test_y is specified)
    """
    # Clean and prepare data 
    single_train_tx, single_test_tx = prepare_data(single_train_tx, single_test_tx, single_degree, jet_id)

    # Compute the weights with ridge regression
    weights = ridge_regression(single_train_y, single_train_tx, single_lambda)

    # Compute the predictions
    y_pred_train = predict_labels(weights, single_train_tx)
    y_pred_test = predict_labels(weights, single_test_tx)
    
    # Compute accuracy of the predictions
    accuracy_train = np.sum(y_pred_train == single_train_y)/len(single_train_y)
    if len(single_test_y) != 0:
        accuracy_test = np.sum(y_pred_test == single_test_y)/len(single_test_y)
        if predictions==True:
            return y_pred_train, y_pred_test, accuracy_train, accuracy_test
        else:
            return accuracy_train, accuracy_test
    else:
        if predictions==True:
            return y_pred_train, y_pred_test, accuracy_train
        else:
            return accuracy_train
        

def ridge_regression_all_jets_pred(full_tx_train, full_tx_test, full_y_train, degrees, lambdas):
    """ Splits the dataset by jets and computes the ridge regression predictions on each newly created
	dataset. 
	
	Input: 
	     - full_tx_train: matrix of training data
	     - full_tx_test: matrix of testing data
	     - full_y_train: array of training outputs
	     - degrees: array of maximal polynomial degrees for each jet
	     - lambdas: array of stabilizing parameters used for the ridgee regression corresponding to each jet
	Ouput: 
	     - y_pred_train: array of predicted values in the training set, if predictions=True
	     - y_pred_test: array of predicted values in the testing set, if predictions=True
	     - full_accuracy_train: scalar, accuracy on the full training set
    """
    # Generate the masks to select the jets
    mask_jets_train = split_jets_mask(full_tx_train)
    mask_jets_test = split_jets_mask(full_tx_test)
    
    # Initialize ouput variables
    len_mask = len(mask_jets_train)
    y_pred_train = np.zeros(len(full_y_train))
    y_pred_test = np.zeros(full_tx_test.shape[0])
    accuracy_train = np.zeros(len_mask)
    len_jets_train = np.zeros(len_mask)
    
    # Loop over the jet numbers
    for mask_jet_id in range(len_mask):
        print('********** Jet ', mask_jet_id, '***********')
	# Create the dataset corresponding to the mask_jet_id (i.e. to a certain jet number)
        tx_single_jet_train = full_tx_train[mask_jets_train[mask_jet_id]]
        tx_single_jet_test = full_tx_test[mask_jets_test[mask_jet_id]]
        y_single_jet_train = full_y_train[mask_jets_train[mask_jet_id]]
        len_jets_train[mask_jet_id] = len(y_single_jet_train)
        
	# Compute the predictions on a single dataset (corresponding to jet mask_jet_id)
        y_pred_train[mask_jets_train[mask_jet_id]], y_pred_test[mask_jets_test[mask_jet_id]], \
            accuracy_train[mask_jet_id] = cleaned_ridge_regression_pred(mask_jet_id, degrees[mask_jet_id], lambdas[mask_jet_id], \
                                                                        tx_single_jet_train, tx_single_jet_test, \
                                                                        y_single_jet_train, [], predictions=True)
    
    # Compute the accuracy on the full training set
    full_accuracy_train = \
        np.sum([accuracy_train[id]*len_jets_train[id] for id in range(len_mask)])/len(full_y_train)
        
    return y_pred_train, y_pred_test, full_accuracy_train



