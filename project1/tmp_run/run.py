#%matplotlib inline 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from proj1_helpers import * #the helper provided for the project

from implementations import * #our implementations of the functions done by us
from helpers import * # other helpers
import datetime
import operator

####  function definition




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
    print('Cleaning missing values')
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


def cleaned_ridge_regression_pred(single_degree, single_lambda, single_train_tx, single_test_tx, \
                                  single_train_y, single_test_y=[], predictions=True):
    # Clean and prepare data !!!!!!!!!TODO!!!!!!!!!! NOT HERE WOULD BE MORE EFFICIENT.......
    single_train_tx, single_test_tx = prepare_data(single_train_tx, single_test_tx, single_degree)

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
    mask_jets_train = split_jets_mask(full_tx_train)
    mask_jets_test = split_jets_mask(full_tx_test)
    
    len_mask = len(mask_jets_train)
    y_pred_train = np.zeros(len(full_y_train))
    y_pred_test = np.zeros(full_tx_test.shape[0])
    accuracy_train = np.zeros(len_mask)
    len_jets_train = np.zeros(len_mask)
    
    for mask_jet_id in range(len_mask):
        print('********** Jet ', mask_jet_id, '***********')
        tx_single_jet_train = full_tx_train[mask_jets_train[mask_jet_id]]
        tx_single_jet_test = full_tx_test[mask_jets_test[mask_jet_id]]
        y_single_jet_train = full_y_train[mask_jets_train[mask_jet_id]]
        len_jets_train[mask_jet_id] = len(y_single_jet_train)
        
        y_pred_train[mask_jets_train[mask_jet_id]], y_pred_test[mask_jets_test[mask_jet_id]], \
            accuracy_train[mask_jet_id] = cleaned_ridge_regression_pred(degrees[mask_jet_id], lambdas[mask_jet_id], \
                                                                        tx_single_jet_train, tx_single_jet_test, \
                                                                        y_single_jet_train, [], predictions=True)
    
    full_accuracy_train = \
        np.sum([accuracy_train[id]*len_jets_train[id] for id in range(len_mask)])/len(full_y_train)
        
    return y_pred_train, y_pred_test, full_accuracy_train







#data loading


DATA_FOLDER = 'data/'

y_train, tx_train, ids_train = load_csv_data(DATA_FOLDER+'train.csv',sub_sample=False)

y_test, tx_test, ids_test = load_csv_data(DATA_FOLDER+'test.csv',sub_sample=False)


#############declaring best lambda and degree#############
best_degree_per_jet = [9, 11, 12, 12]
best_lambda_per_jet = [  1.00000000e-08,   1.00000000e-03,   1.00000000e-02,   1.00000000e-02]

##########################################################
print(best_degree_per_jet)
print(best_lambda_per_jet)



y_pred_train, y_pred_test, full_accuracy_train, = \
    ridge_regression_all_jets_pred(tx_train, tx_test, y_train, best_degree_per_jet, best_lambda_per_jet)
    
print (full_accuracy_train)


print('Shapes are (for verification): ')
print(y_pred_test.shape)

print(y_pred_test[y_pred_test==-1].shape)
print(y_pred_test[y_pred_test==1].shape)



print(y_pred_test[0:200])



name = 'output/ridge_regression_ondine_splitjet_2.csv'
create_csv_submission(ids_test, y_pred_test, name)



    


    
    