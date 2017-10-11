# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # ***************************************************
    phi_tilde=np.ones((len(x),1));
    for i in range(1,degree+1):
        phi_tilde=np.c_[phi_tilde,np.power(x,i)]
    return phi_tilde
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
