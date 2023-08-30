"""
These functions are used to compute over-approximated reachable sets represented Vzono

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

TODO: add functions for CNNs
"""

import numpy as np
import itertools

class VzonoFFNN:
    """
    A class for Vzono set representation in over-approximation reachability analysis of FFNN

        Attributes:
            base_vertices (np.ndarray): Component of the set representation. Details can be found in
            Equation 13 of the work "Neural Network Repair with Reachability Analysis"
            base_vectors (np.ndarray): Component of the set representation. Details can be found in
            Equation 13 of the work "Neural Network Repair with Reachability Analysis"
    """
    def __init__(self, base_vertices=None, base_vectors=None):
        self.base_vertices = base_vertices
        self.base_vectors = base_vectors


    def create_from_bounds(self, lbs, ubs):
        """
        Construct the set from lower bounds and upper bounds

        Parameters:
            lbs (list): Lower bounds
            ubs (list): Upper bounds

        """
        self.base_vertices = (np.array([lbs])+np.array([ubs])).T/2
        self.base_vectors = np.diag((np.array(ubs)-np.array(lbs))/2)

    def get_sound_vertices(self):
        """
        Over approximation when there are too many vertices

        Returns:
            vertices (np.ndarray): Vertices of the Vzono set
        """

        vals = np.sum(np.abs(self.base_vectors), axis=1, keepdims=True)
        V = [[v,-v] for v in vals]
        combs = list(itertools.product(*V))

        vertices = []
        for cb in combs:
            vertices.append(self.base_vertices+np.array(cb))
        vertices = np.concatenate(vertices, axis=1)
        return vertices


    def get_vertices(self):
        """
        Compute exact vertices from the vzono object

        Returns:
            vertices (np.ndarray): Vertices of the Vzono set
        """
        vertices = []
        V = [[-self.base_vectors[:,n], self.base_vectors[:,n]] for n in range(self.base_vectors.shape[1])]
        combs = list(itertools.product(*V))
        for cb in combs:
            cb = np.sum(np.array(cb).T, axis=1, keepdims=True)
            vertices.append(self.base_vertices + cb)

        vertices = np.concatenate(vertices,axis=1)
        return vertices


    def affine_map(self, W, b):
        """
        Affine mapping of a vzono reachable set

        Parameters:
            W (np.ndarray): Weights matrix
            b (np.ndarray): Vector

        """
        self.base_vertices = np.dot(W, self.base_vertices) + b
        self.base_vectors = np.dot(W, self.base_vectors)


    def get_valid_neurons_for_over_app(self):
        """
        Compute neurons whose negative and positive input ranges are spanned by the set

        Returns:
            valid_neurons_neg_pos (np.ndarray): ReLU neurons whose both negative and positive input ranges are spanned by the input set
            valid_neurons_neg (np.ndarray): Neurons whose only negative input range is spanned by the input set
        """
        vals = np.sum(np.abs(self.base_vectors), axis=1, keepdims=True)
        temp_neg = np.all((self.base_vertices+vals) <= 0, 1)
        valid_neurons_neg = np.asarray(np.nonzero(temp_neg)).T[:, 0]
        temp_pos = np.all((self.base_vertices-vals) >= 0, 1)
        neurons_sum = temp_neg + temp_pos
        valid_neurons_neg_pos = np.asarray(np.nonzero(neurons_sum == False)).T[:, 0]

        return valid_neurons_neg_pos, valid_neurons_neg




