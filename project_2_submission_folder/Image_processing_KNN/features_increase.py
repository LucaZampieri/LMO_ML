
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import pandas as pd
from skimage import data, io, filters

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


def add_ones(tx):
    """
	Add column of ones to the dataset tx
    """
    return np.concatenate((tx, np.ones([tx.shape[0],1])), axis=1)



def build_poly(x, degree):
    """ Returns the polynomial basis functions for input data x, for j=2 up to j=degree."""
    new_cols=np.array([x**p for p in range(2,degree+1)]).T;
    return new_cols

def add_powers(tx, degree):
    for col in range(0,tx.shape[1]): 
            tx = np.concatenate((tx, build_poly(tx[:,col], degree)), axis=1)
    return tx