# -*- coding: utf-8 -*-


def cleaned_ridge_regression_pred(single_degree, single_lambda, single_train_tx, single_test_tx, \
                                  single_train_y, single_test_y=[], predictions=True):
    # Clean and prepare data
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
