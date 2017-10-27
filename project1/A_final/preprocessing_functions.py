# -*- coding: utf-8 -*-

import numpy as np 

def split_jets_mask(tx):
    idx_cat = 22
    return {
        0: tx[:,idx_cat] == 0,
        1: tx[:,idx_cat] == 1,
        2: tx[:,idx_cat] == 2,
        3: tx[:,idx_cat] == 3,
    }
    
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
    
def keep_unique_cols(tx):
    # If two (or more) columns of tx are equal, keep only one of them
    unique_cols_ids = [0]
    for i in range(1,tx.shape[1]):
        id_loop = unique_cols_ids
        erase = False
        equal_to = []
        
        erase = len( tx[:,i] ) == len(tx[tx[:,i]==tx[0,i],i])
        if erase == False:
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
    
def add_cross_prod(tx, i, j):
    return np.concatenate((tx, np.array([tx[:,i]*tx[:,j]]).T), axis=1)

def add_all_cross_prod(tx):
    sh = tx.shape[1]
    for i in range(sh):
        for j in range(i+1, sh):
            if i != j:
                tx = add_cross_prod(tx, i, j)
    return tx
    
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    return np.array([x**p for p in range(2,degree+1)]).T 
    # not range from 0 because we have already added a column of ones, 
    # not range from 1 because we already have the linear features."""

def build_poly_mat(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # ***************************************************
    phi_tilde=x;
    for i in range(1,degree+1):
        phi_tilde=np.c_[phi_tilde,np.power(x,i)]
    return phi_tilde
    
    
    

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

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    for idx in range(len(std_x)):
        if std_x[idx] > 1e-15:
            x[:,idx] = x[:,idx] / std_x[idx]
    return x, mean_x, std_x

    
# ---------------- MAIN FUNCTION TO CLEAN DATA ----------------- #

def prepare_data(train_tx, test_tx, deg):
    print('Cleaning features')
    train_tx = clean_missing_values(train_tx)[0]
    test_tx = clean_missing_values(test_tx)[0]
    
    
    print('Keeping unique cols')
    unique_cols = keep_unique_cols(train_tx)
    train_tx = train_tx[:,unique_cols]
    test_tx = test_tx[:,unique_cols]
    len_kept_data = len(unique_cols)
 
    
    print('Cross products')
    train_tx = add_all_cross_prod(train_tx)
    test_tx = add_all_cross_prod(test_tx)
   
    
    print('Adding powers')
    train_tx=add_powers(train_tx, deg, 0, len_kept_data, features='x');
    test_tx=add_powers(test_tx, deg,  0, len_kept_data, features='x');
    
   
   
    print('Standardizing')
    train_tx = standardize(train_tx)[0]
    test_tx = standardize(test_tx)[0]
    
    print('Adding ones')
    train_tx = add_ones(train_tx)
    test_tx = add_ones(test_tx)
    
    return train_tx, test_tx
