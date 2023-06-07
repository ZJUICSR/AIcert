"""
These functions are used for reachability analysis in the Sigmoid function

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

TODO: Add more linearization methods
"""

import numpy as np
import copy as cp

def layer_linearize(s):
    """
    Process 's' with respect to a relu neuron using Sigmoid linearization

        Parameters:
            s (Vzono): An input reachable set

        Returns:
            s (Vzono): An output reachable set
    """

    vals = np.sum(np.abs(s.base_vectors), axis=1, keepdims=True)
    ubs = np.max(s.base_vertices,axis=1, keepdims=True) + vals
    lbs = np.min(s.base_vertices, axis=1, keepdims=True) - vals

    A, lyb, uyb = constraints(lbs, ubs)

    epsilons = (uyb-lyb)/2
    M = np.diag(A[:,0])
    b = (uyb+lyb)/2
    base_vectors_relax = np.diag(epsilons[:,0])

    new_base_vertices = np.dot(M,s.base_vertices) + b
    new_base_vectors = np.concatenate((np.dot(M, s.base_vectors), base_vectors_relax), axis=1)
    s.base_vertices = new_base_vertices
    s.base_vectors = new_base_vectors

    return s


def constraints(lbs, ubs):
    """
    Approximate the lower and upper linear constraints for the linearization
    """

    lp = np.zeros((lbs.shape[0],2)) # point locateing on the lower linear constraint
    up = np.zeros((lbs.shape[0],2)) # point locating on the upper linear constraint

    A = (sigmoid(ubs)-sigmoid(lbs))/(ubs-lbs)

    # lbs>=0 and ubs>=0
    dims_pos = np.logical_and(lbs > 0, ubs > 0)  # dimensions in the positive input range of Sigmoid

    dims1 = np.logical_and(dims_pos, ubs<=2)  # ubs <= 2
    lp[dims1[:,0],0], lp[dims1[:,0],1] = lbs[dims1], sigmoid(lbs[dims1])
    up[dims1[:,0],0], up[dims1[:,0],1] = ubs[dims1], piece_wise(ubs[dims1])

    dims2 = np.logical_and(dims_pos, lbs>=2) # lbs >= 2
    lp[dims2[:,0], 0], lp[dims2[:,0], 1] = lbs[dims2], sigmoid(lbs[dims2])
    up[dims2[:,0], 0], up[dims2[:,0], 1] = lbs[dims2], piece_wise(lbs[dims2])

    dims3 = np.logical_and(np.logical_and(dims_pos, ubs>2), lbs<2) # lbs<1 and ubs>1
    lp[dims3[:,0], 0], lp[dims3[:,0], 1] = lbs[dims3], sigmoid(lbs[dims3])
    up[dims3[:,0], 0], up[dims3[:,0], 1] = 2, 1

    # lbs<0 and ubs>0
    dims_neg_pos = np.logical_and(lbs <= 0, ubs >= 0)  # dimensions that span both negative and positive input range of Sigmoid

    dims1 = np.logical_and(dims_neg_pos, lbs>=-2) # lbs >= -2
    lp[dims1[:,0], 0], lp[dims1[:,0], 1] = lbs[dims1], piece_wise(lbs[dims1])

    dims1 = np.logical_and(dims_neg_pos, lbs<-2)  # lbs < -2
    lp[dims1[:,0], 0], lp[dims1[:,0], 1] = -2, 0

    dims2 = np.logical_and(dims_neg_pos, ubs<=2) # ubs <= 2
    up[dims2[:,0], 0], up[dims2[:,0], 1] = ubs[dims2], piece_wise(ubs[dims2])

    dims2 = np.logical_and(dims_neg_pos, ubs>2)  # ubs > 2
    up[dims2[:,0], 0], up[dims2[:,0], 1] = 2, 1

    # lbs < 0 and ubs < 0
    dims_neg = np.logical_and(lbs < 0, ubs < 0)  # dimensions in the negative input range of Sigmoid
    dims1 = np.logical_and(dims_neg, lbs>=-2)  # bs >= -2
    lp[dims1[:,0], 0], lp[dims1[:,0], 1] = lbs[dims1], piece_wise(lbs[dims1])
    up[dims1[:,0], 0], up[dims1[:,0], 1] = lbs[dims1], sigmoid(lbs[dims1])

    dims2 = np.logical_and(dims_neg, ubs<=-2)  # ubs <= -2
    lp[dims2[:,0], 0], lp[dims2[:,0], 1] = ubs[dims2], piece_wise(ubs[dims2])
    up[dims2[:,0], 0], up[dims2[:,0], 1] = lbs[dims2], sigmoid(lbs[dims2])

    dims3 = np.logical_and(np.logical_and(dims_neg, ubs>-2), lbs<-2)  # lbs<-2 and ubs>-2
    lp[dims3[:,0], 0], lp[dims3[:,0], 1] = -2, 0
    up[dims3[:,0], 0], up[dims3[:,0], 1] = lbs[dims3], sigmoid(lbs[dims3])

    lyb = -np.dot(np.diag(A[:,0]), lp[:,[0]]) + lp[:,[1]] # point locating the y-axis and the lower linear constraint
    uyb = -np.dot(np.diag(A[:,0]), up[:,[0]]) + up[:,[1]] # # point locating the y-axis and the lower linear constraint

    return [A, lyb, uyb]


def sigmoid(x):
    """
    Sigmoid function

    Parameters:
        x (np.ndarray): Input to this function

    Returns:
        y (np.ndarray): Output of this function
    """

    y = 1/(1+np.exp(-x))
    return y


def piece_wise(x):
    """
    A piecewise linear function to approximate the lower and upper constraints in Sigmoid linearization

    Parameters:
        x (np.ndarray): Input to this function

    Returns:
        y (np.ndarray): Output of this function
    """
    y = cp.deepcopy(x)
    y[x>=2] = 1
    y[x<=-2] = 0
    y[np.logical_and(x>-2, x<2)] = y[np.logical_and(x>-2, x<2)]/4 + 0.5

    return y
