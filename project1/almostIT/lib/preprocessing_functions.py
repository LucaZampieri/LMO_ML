# -*- coding: utf-8 -*-

import numpy as np 
from lib.helpers import *



def split_jets_mask(tx):
    """Creates a mask from array tx corresponding to feature 22 (number of jets)."""
    idx_cat = 22
    return {
        0: tx[:,idx_cat] == 0,
        1: tx[:,idx_cat] == 1,
        2: tx[:,idx_cat] == 2,
        3: tx[:,idx_cat] == 3,
    }

def clean_missing_values(tx, mean=False):
    """Replace missing values (-999) of tx by the mean (if mean==True) or by the median (if mean==False) of the 
       column in which the missing value belongs.
       Returns the modified matrix and a matrix similar to tx but with 1 instead of -999 and 0 otherwise."""
    
    nan_values = (tx==-999)*1
    for col in range(tx.shape[1]):
        # Select entries of col that have values -999
        column = tx[:,col][tx[:,col]!=-999]
        
        # If the column col contains missing values, replace them by the median or the mean
        if len(column) != 0:
            if mean==False:
                replace_value = np.median(column)
            else:
                replace_value = np.mean(column)
            tx[:,col][tx[:,col]==-999] = replace_value
            
    return tx, nan_values
    
def keep_unique_cols(tx):
    """Input: matrix tx. If two (or more) columns of tx are equal, keep only one of them.
       If a column has all of its entries identical, do not keep it (it will be replaced by the column of 1s).
       Returns the indices of the kept columns."""

    unique_cols_ids = [0]
    for i in range(1,tx.shape[1]):
        # Check if all entries of column i are identical
        erase = np.sum((tx[:,i]==tx[0,i])*1)==len(tx[:,i]) 
        
        # If all entries are not identical, check if the column is equal to another one amongst the 
        # already chosen columns
        if erase == False: 
            id_loop = unique_cols_ids
            for j in id_loop:
                if np.sum(tx[:,i]-tx[:,j])==0:
                    erase = True
                    break
                    
        # If the column is not equal to another one, nor it has all identical entries, keep it
        if erase == False: 
            unique_cols_ids.append(i)
    
    return unique_cols_ids
 
def add_cross_prod(tx, i, j):
    """Add a column to tx correponding to the product of the entries of the i-th and j-th columns."""
    return np.concatenate((tx, np.array([tx[:,i]*tx[:,j]]).T), axis=1)

def add_all_cross_prod(tx, selected_cols=[]):
    """
    Add columns with the products between 2 columns (two by two)
    input: 
    """
    if selected_cols==[]:
        selected_cols=range(tx.shape[1])
    for idx, i in enumerate(selected_cols):
        #print(idx)
        for j in selected_cols[idx+1:]:
            tx = add_cross_prod(tx, i, j)
    return tx

def add_ones(tx):
    """
    Add a column of 1's to matrix tx.
    input:  dataset
    output: input dataset with a column of one's added 
    """
    return np.concatenate((tx, np.ones([tx.shape[0],1])), axis=1)
    
def preprocess_data(tx, unique_cols=[], stdize="before"):
    """
    input:  tx --> dataset to pre-process
            unique_cols --> list of indices of columns to keep if already known,
                            default value empty list
            stdize --> can take two values: 'before' & 'after'. Default = 'before'
                       'before' if we want it standardized before adding the cross-products
                       'after' if we want the dataset to be standardized after the cross-products
    output: the processed dataset
    """
    tx, nan_values = clean_missing_values(tx)
    np.append(tx, nan_values[:,0]) # Add dummy variable to keep the information saying when the mass is -999 or not
    
    if unique_cols==[]:
        unique_cols = keep_unique_cols(tx)
    tx = tx[:,unique_cols]
    len_kept_data = len(unique_cols)
    
    if stdize=='before':
        tx = standardize(tx)[0]
    
    tx = add_all_cross_prod(tx)
    
    if stdize=='after':
        tx = standardize(tx)[0]
    
    tx = add_ones(tx)

    return tx, len_kept_data, unique_cols
    
def build_poly(x, range_degrees):
    """Polynomial basis functions for input array data x."""
    return np.array([x**p for p in range_degrees]).T 
    
def add_powers(tx, range_degrees, range_col_idx):
    """
    Adds columns with the power of the function up to degree range_degrees
    input: tx --> dataset
           range_degrees --> degree up to which we want to add 
           range_col_idx -->
    output: the dataset tx with the added columns
    """
    if len(range_degrees)>0:
        for col_id in range_col_idx:
            tx = np.concatenate((tx, build_poly(tx[:,col_id], range_degrees)), axis=1)
    return tx
    

