"""
These functions are used to compute reachable sets represented FVIM

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause


"""

import sys
import copy as cp
import numpy as np

class FVIM:
    """
    A class for the set representation of a polytope based on Facet-vertex Incidence Matrix

    Attributes:
        fmatrix (np.ndarray): Matrix encoding the containment relation between facets and vertices of the polytope
        vertices (np.ndarray): Vertices of the polytope
        dim (int): Dimension of the polytope
        M (np.ndarray): Matrix for the affine mapping relation
        b (np.ndarray): Vector for the affine mapping relation

    Methods:
        affine_map(M, b):
            Affine map the polytope through M and b
        affine_map_negative(n):
            Set nth dimension to zero
        relu_split(neuron_pos_neg):
            Split the polytope in a relu neuron
        relu_split_hyperplane(A, d):
            Split the polytope by a hyperplane Ax + d = 0

    """

    def __init__(self, fmatrix, vertices, dim, M, b):
        """
        Constructs all the necessary attributes for the polytope object

            Parameters:
                fmatrix (np.ndarray): Matrix encoding the containment relation between facets and vertices of the polytope
                vertices (np.ndarray): Vertices values of the polytope
                dim (int): Dimension of the polytope
                M (np.ndarray): Matrix for the affine mapping relation between current polytope and the initial one
                b (np.ndarray): Vector for the affine mapping relation

        """
        self.fmatrix = fmatrix
        self.vertices = vertices
        self.dim = dim
        self.M = M
        self.b = b


    def affine_map(self, W, b):
        """
        Affine map the polytope through M and b

            Parameters:
                W (np.ndarray): Matrix
                b (np.ndarray): Vector
        """
        self.M = np.dot(W, self.M)
        self.b = np.dot(W, self.b) + b


    def affine_map_negative(self, n):
        """
        Set nth dimension to zero

            Parameters:
                n (int): Dimension index of the matrix and vector
        """
        self.M[n, :] = 0
        self.b[n, :] = 0


    def relu_split(self, neuron_pos_neg):
        """
        Split the polytope in a relu neuron

            Parameters:
                neuron_pos_neg (int): index of the target neuron in the layer

            Returns:
                subset0 (FVIM): an output set
                subset1 (FVIM): an output set
        """

        elements = np.matmul(self.vertices, self.M[neuron_pos_neg,:].T)+self.b[neuron_pos_neg,:].T
        if np.any(elements==0.0):
            sys.exit('Hyperplane intersect with vertices!')

        # indices of vertices that locates in the positive input range of ReLU
        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        # indices of vertices that locates in the negative input range of ReLU
        negative_bool = np.invert(positive_bool)
        negative_id = np.asarray(negative_bool.nonzero()).T

        if len(positive_id)>=len(negative_id):
            less_bool = negative_bool
            more_bool = positive_bool
            flg = 1
        else:
            less_bool = positive_bool
            more_bool = negative_bool
            flg = -1

        vs_facets0 = self.fmatrix[less_bool]
        vs_facets1 = self.fmatrix[more_bool]
        vertices0 = self.vertices[less_bool]
        vertices1 = self.vertices[more_bool]
        elements0 = elements[less_bool]
        elements1 = elements[more_bool]

        # identify the vertices that are connected by edges
        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        if len(edges_indx[0])+len(edges_indx[1]) == 0:
            sys.exit('Intersected edges are empty!')
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]
        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))

        # compute the new vertices from the intersection between the edges and the hyperplane x_i = 0 in ReLU_i
        new_vs = p0s + ((p1s - p0s).T * alpha).T
        # compute the containment relation between facets and new vertices
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])

        # form FVIM for one subset
        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:,np.any(vs_facets0,0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = FVIM(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == 1:
            # map negative dims to zeros in terms of the relu function
            subset0.affine_map_negative(neuron_pos_neg)

        # form FVIM for the other subset
        new_vs_facets1 = np.concatenate((vs_facets1, new_vs_facets))
        sub_vs_facets1 = new_vs_facets1[:, np.any(vs_facets1, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets1), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets1 = np.concatenate((sub_vs_facets1, vs_facets_hp), axis=1)
        new_vertices1 = np.concatenate((vertices1, new_vs))
        subset1 = FVIM(sub_vs_facets1, new_vertices1, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == -1:
            # map negative dims to zeros in terms of the relu function
            subset1.affine_map_negative(neuron_pos_neg)
        
        return subset0, subset1


    def relu_split_hyperplane(self, A, d):
        """
        Split the polytope by a hyperplane Ax + d = 0

            Parameters:
                A (np.ndarray): Vector
                d (float): Value

            Returns:
                subset0 (FVIM): Subset locating in the halfspace, Ax + d <= 0
        """
        A_new = np.dot(A,self.M)
        d_new = np.dot(A, self.b) +d
        elements = np.dot(A_new, self.vertices.T) + d_new
        elements = elements[0]
        # the polytope locates in the positive halfspace, Ax + d >= 0
        if np.all(elements >= 0):
            return None
        # the polytope locates in the negative halfspace, Ax + d <= 0
        elif np.all(elements <= 0):
            return self
        # the polytope locates in the negative halfspace, Ax + d == 0
        elif np.any(elements == 0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements > 0)
        negative_bool = np.invert(positive_bool)

        vs_facets0 = self.fmatrix[negative_bool]
        vs_facets1 = self.fmatrix[positive_bool]
        vertices0 = self.vertices[negative_bool]
        vertices1 = self.vertices[positive_bool]
        elements0 = elements[negative_bool]
        elements1 = elements[positive_bool]

        # identify the vertices that are connected by edges
        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        if len(edges_indx[0])+len(edges_indx[1]) == 0:
            sys.exit('Intersected edges are empty!')
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]

        # compute the new vertices from the intersection between the edges and the hyperplane
        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))
        new_vs = p0s + ((p1s - p0s).T * alpha).T
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])

        # form the FVIM for the subset
        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:, np.any(vs_facets0, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):, 0] = True  # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = FVIM(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))

        # new_vs_facets1 = np.concatenate((vs_facets1, new_vs_facets))
        # sub_vs_facets1 = new_vs_facets1[:, np.any(vs_facets1, 0)]
        # vs_facets_hp = np.zeros((len(sub_vs_facets1), 1), dtype=bool)
        # vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        # sub_vs_facets1 = np.concatenate((sub_vs_facets1, vs_facets_hp), axis=1)
        # new_vertices1 = np.concatenate((vertices1, new_vs))
        # subset1 = FVIM(sub_vs_facets1, new_vertices1, self.dim, cp.copy(self.M), cp.copy(self.b))

        return subset0





