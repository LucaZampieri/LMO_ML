# -*- coding: utf-8 -*-

import numpy as np 
from helpers import *



# splitting dataset into jets

def split_jets_mask(tx):
    idx_cat = 22
    return {
        0: tx[:,idx_cat] == 0,
        1: tx[:,idx_cat] == 1,
        2: tx[:,idx_cat] == 2,
        3: tx[:,idx_cat] == 3,
}




# cleaning -999 and replacing them with the median of the column

def clean_missing_values(tx):
    nan_values = (tx==-999)*1
    for col in range(tx.shape[1]):
        column = tx[:,col][tx[:,col]!=-999]
        if len(column) == 0:
            median = 0
        else:
            median = np.median(column)
        tx[:,col][tx[:,col]==-999] = median
    return tx, nan_values



# the unique columns are mantained, if there is two equal columns the second is removed

def keep_unique_cols(tx):
    # If two (or more) columns of tx are equal, keep only one of them
    unique_cols_ids = [0]
    for i in range(1,tx.shape[1]):
        id_loop = unique_cols_ids
        erase = False
        equal_to = []
        for j in id_loop:
            if np.sum(tx[:,i]-tx[:,j])==0:
                erase = True
                equal_to.append(j)
                break
        if erase == False:
            unique_cols_ids.append(i)
    #else:
#    print('column', i, 'deleted because equal to column(s) ', equal_to)

index = np.argwhere(unique_cols_ids==22)
unique_cols_ids = np.delete(unique_cols_ids, index)

    return unique_cols_ids


# add cross product to one column

def add_cross_prod(tx, i, j):
    return np.concatenate((tx, np.array([tx[:,i]*tx[:,j]]).T), axis=1)


#add cross product to the whole matrix

def add_all_cross_prod(tx):
    sh = tx.shape[1]
    for i in range(sh):
        #print(i)
        for j in range(i+1, sh):
            if i != j:
                tx = add_cross_prod(tx, i, j)
    return tx



def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    return np.array([x**p for p in range(2,degree+1)]).T
# not range from 0 because we have already added a column of ones,
# not range from 1 because we already have the linear features.




def add_powers(tx, degree, first_data_id, len_init_data, features='x'):
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
    return np.concatenate((tx, np.ones([tx.shape[0],1])), axis=1)



def prepare_data(train_tx, test_tx, deg):
    print('Cleaning features')
    train_tx = clean_missing_values(train_tx)[0]
    test_tx = clean_missing_values(test_tx)[0]
    
    print('Keeping unique cols')
    unique_cols = keep_unique_cols(train_tx)
    train_tx = train_tx[:,unique_cols]
    test_tx = test_tx[:,unique_cols]
    len_kept_data = len(unique_cols)
    
    print('Standardizing')
    train_tx = standardize(train_tx)[0]
    test_tx = standardize(test_tx)[0]
    
    print('Cross products')
    train_tx = add_all_cross_prod(train_tx)
    test_tx = add_all_cross_prod(test_tx)
    
    print('Adding powers')
    train_tx = add_powers(train_tx, deg, 0, len_kept_data, 'x')
    test_tx = add_powers(test_tx, deg, 0, len_kept_data, 'x')
    
    print('Adding ones')
    train_tx = add_ones(train_tx)
    test_tx = add_ones(test_tx)
    
    return train_tx, test_tx
