# Authors: Ondine Chanon, Matteo Ciprian, Luca Zampieri
# license: MIT

# Implementation of the functions useful for the preprocessing


# Import libraries
import numpy as np 
from helpers import *


def split_jets_mask(tx):
    """
	Creates a mask to be able to separate the full dataset tx into the different jets (feature number 22)
    """
    idx_cat = 22
    return {
        0: tx[:,idx_cat] == 0,
        1: tx[:,idx_cat] == 1,
        2: tx[:,idx_cat] == 2,
        3: tx[:,idx_cat] == 3,
    }


def clean_missing_values(tx):
    """ 
    Handling missing values: replace -999 values of the dataset tx by the median of the column (i.e. feature)
    and create a corresponding matrix containing 1s where tx has a -999 value, and 0 otherwise.

    Input: 
    	 - tx: array, original dataset
    Ouput: 
    	 - tx: array, final dataset with -999 replaced
	 - nan_values: array of dummy variables corresponding to the -999 values
    """
    nan_values = (tx==-999)*1
    for col in range(tx.shape[1]):
        column = tx[:,col][tx[:,col]!=-999]
        if len(column) == 0:
            median = 0
        else:
            median = np.median(column)
        tx[:,col][tx[:,col]==-999] = median
    return tx, nan_values


def keep_unique_cols(tx):
    """
	Modify the dataset array tx by keeping only unique columns, and columns whose entries are
	not all equal. 
    """
    # If two (or more) columns of tx are equal, keep only one of them
    unique_cols_ids = [0]
    for i in range(1,tx.shape[1]):
        id_loop = unique_cols_ids
        erase = False
        for j in id_loop:
            if np.sum(tx[:,i]-tx[:,j])==0:
                erase = True
                break
        if erase == False:
            unique_cols_ids.append(i)
            
    index = np.argwhere(unique_cols_ids==22)
    unique_cols_ids = np.delete(unique_cols_ids, index)
    
    return unique_cols_ids



def add_cross_prod(tx, i, j):
    """ Add cross products between features i and j in matrix tx. """
    return np.concatenate((tx, np.array([tx[:,i]*tx[:,j]]).T), axis=1)


def add_all_cross_prod(tx):
    """ Add all cross products between two distinct columns of tx and returns the larger dataset."""
    sh = tx.shape[1]
    for i in range(sh):
        #print(i)
        for j in range(i+1, sh):
            if i != j:
                tx = add_cross_prod(tx, i, j)
    return tx


def build_poly(x, degree):
    """ Returns the polynomial basis functions for input data x, for j=2 up to j=degree."""
    return np.array([x**p for p in range(2,degree+1)]).T 
    # not range from 0 because we have already added a column of ones, 
    # not range from 1 because we already have the linear features.


def add_powers(tx, degree, first_data_id, len_init_data, features='x'):
    """ Add powers of each specified column. 
	Input: 
	     - tx: matrix or array, original dataset
	     - degree: int, maximum polynomial order for the polynomial basis of each feature
	     - first_data_id: int, first column id to consider
	     - len_init_data: int, length of the column ids to consider (ie we will consider the columns
	       from id first_data_id to id first_data_id+len_init_data
	     - features (optional): string, 'x' if we want the powers of the columns, 'cp' if we want the 
	       powers of the cross products.
	Ouput: 
	     - tx: array or matrix of enlarged data with the powers of the features added.
    """

    if features == 'x': # square roots of initial (kept) features
        range_c = range(first_data_id, first_data_id+len_init_data)
    elif features == 'cp': # square roots of cross products
        range_c = range(first_data_id, first_data_id+(len_init_data*(len_init_data-1))/2)
    else:
        raise NameError('Need to specity x (features) of cp (cross products)')
    for col in range_c: 
        tx = np.concatenate((tx, build_poly(tx[:,col], degree)), axis=1)
    return tx


def add_ones(tx):
    """
	Add column of ones to the dataset tx
    """
    return np.concatenate((tx, np.ones([tx.shape[0],1])), axis=1)


def add_square_roots(tx, first_data_id, len_init_data):
    """
	Add square roots of the features to the dataset tx with id of the features ranging
	from first_data_id to first_data_id+len_init_data.
    """
    range_c = range(first_data_id, first_data_id+len_init_data)
    sqrt_array = np.array([np.sqrt(np.abs(tx[:,c])) for c in range_c]).T
    return np.concatenate((tx, sqrt_array), axis=1)


def add_log_abs(tx, first_data_id, len_init_data):
    """
	Add log(1+|x|) for each feature x of the dataset tx, with id of the features ranging
	from first_data_id to first_data_id+len_init_data.
    """
    range_c = range(first_data_id, first_data_id+len_init_data)
    new_array = np.array([np.log(1+np.abs(tx[:,c])) for c in range_c]).T
    return np.concatenate((tx, new_array), axis=1)
    

def add_log_square(tx, first_data_id, len_init_data):
    """
	Add log(1+x^2) for each feature x of the dataset tx with id of the features ranging
	from first_data_id to first_data_id+len_init_data.
    """
    range_c = range(first_data_id, first_data_id+len_init_data)
    new_array = np.array([np.log(1+tx[:,c]**2) for c in range_c]).T
    return np.concatenate((tx, new_array), axis=1)


def add_gaussian(tx, first_data_id, len_init_data):
    """
	Add exp(-x^2/2) for each feature x of the dataset tx with id of the features ranging
	from first_data_id to first_data_id+len_init_data.
    """
    range_c = range(first_data_id, first_data_id+len_init_data)
    new_array = np.array([np.exp(-tx[:,c]**2/2.0) for c in range_c]).T
    return np.concatenate((tx, new_array), axis=1)


def standardize_prev(x):
	"""Standardize the columns of the original dataset.
	   Returns the modified dataset, the original means and standard deviations of each column.
	"""
	mean_x = np.mean(x, axis=0)
	x = x - mean_x
	std_x = np.std(x, axis=0)
	for idx in range(len(std_x)):
		if std_x[idx] > 1e-15:
			x[idx] = x[idx] / std_x[idx]
	return x, mean_x, std_x


def prepare_data(train_tx, test_tx, deg, jet_id):
    """
    Main function used to prepare the training and testing datasets.
    
    Input: 
	- train_tx: matrix of data with the original training data and features.
	- test_tx: matrix of data with the original testing data and features. 
	- deg: int, maximum polynomial degree on which are constructed the powers of the features.
	- jet_id: jet number considered (influences which features are added). 
    Ouput: 
	- train_tx: training data matrix with cleaned and added features and datapoints
	- test_tx: testing data matrix with cleaned and added features and datapoints
    """

    #print('Cleaning features')
    train_tx = clean_missing_values(train_tx)[0]
    test_tx = clean_missing_values(test_tx)[0]
    
    #print('Keeping unique cols')
    unique_cols = keep_unique_cols(train_tx)
    train_tx = train_tx[:,unique_cols]
    test_tx = test_tx[:,unique_cols]
    len_kept_data = len(unique_cols)
    
    #print('Standardizing')
    train_tx = standardize_prev(train_tx)[0]
    test_tx = standardize_prev(test_tx)[0]
    
    #print('Cross products')
    train_tx = add_all_cross_prod(train_tx)
    test_tx = add_all_cross_prod(test_tx)
    
    #print('Adding powers')
    train_tx = add_powers(train_tx, deg, 0, len_kept_data, 'x')
    test_tx = add_powers(test_tx, deg, 0, len_kept_data, 'x')
    
    #print('Adding square roots')
    train_tx = add_square_roots(train_tx, 0, len_kept_data)
    test_tx = add_square_roots(test_tx, 0, len_kept_data)
    
    if jet_id == 1:
     #   print('Adding log(1+|x|)')
        train_tx = add_log_abs(train_tx, 0, len_kept_data)
        test_tx = add_log_abs(test_tx, 0, len_kept_data)
    else:
      #  print('Adding log(1+x^2)')
        train_tx = add_log_square(train_tx, 0, len_kept_data)
        test_tx = add_log_square(test_tx, 0, len_kept_data)
    
    if jet_id > 1:
       # print('Adding gaussian')
        train_tx = add_gaussian(train_tx, 0, len_kept_data)
        test_tx = add_gaussian(test_tx, 0, len_kept_data)
    
    #print('Adding ones')
    train_tx = add_ones(train_tx)
    test_tx = add_ones(test_tx)
    
    return train_tx, test_tx


