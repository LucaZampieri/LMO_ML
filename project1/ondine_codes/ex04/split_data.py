# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    
    # split data with the given ratio
    data_size = len(y)
    training_size = int(np.round(data_size*ratio))

    shuffle_indices = np.random.permutation(np.arange(data_size))
    
    shuffled_y = y[shuffle_indices]
    shuffled_tx = x[shuffle_indices]
    y_train = shuffled_y[0:training_size]
    x_train = shuffled_tx[0:training_size]
    y_test = shuffled_y[training_size:]
    x_test = shuffled_tx[training_size:]
        
    return y_train, x_train, y_test, x_test
