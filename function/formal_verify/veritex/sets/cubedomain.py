"""
These functions are used construct hypercubes in FVIM

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

import itertools
import numpy as np
from veritex.sets.facetvertex import FVIM


class CubeDomain:
    """
    A class for the cube set in FVIM

        Attributes:
            dims_active (np.ndarray): Dimensions that have intervals
            dims_static (np.ndarray): Dimensions that are constant values
            dim (int): Dimensionality of the cube
            lbs_active (np.ndarray): Lower bounds of the active dimensions
            ubs_active (np.ndarray): Upper bounds of the active dimensions
            lbs_static (np.ndarray): Lower bounds of the static dimensions
            ubs_static (np.ndarray): Upper bounds of the static dimensions
            M (np.ndarray): Matrix for an affine mapping
            b (np.ndarray): Vector for an affine mapping
            vertices (np.ndarray): Vertices of the cube

        Methods:
            to_FVIM():
                Convert CubeDomain object to FVIM
            compute_fmatrix():
                Compute the matrix encoding the containment relation between facets and vertices of the cube
            compute_vertices():
                Compute the vertices of the cube

    """

    def __init__(self, lbs, ubs):
        """
        Constructs all the necessary attributes for the CubeDomain object

        Parameters:
            lbs (list): Lower bounds of the interval
            ubs (list): upper bounds of the interval
        """

        assert len(lbs) == len(ubs)
        lbs, ubs = np.array(lbs), np.array(ubs)
        self.dims_active = np.nonzero(lbs != ubs)[0]
        self.dims_static = np.nonzero(lbs == ubs)[0]
        self.dim = len(self.dims_active)

        self.lbs_active = lbs[self.dims_active]
        self.ubs_active = ubs[self.dims_active]

        self.lbs_static = lbs[self.dims_static]
        self.ubs_static = ubs[self.dims_static]

        self.M = np.eye(len(lbs))
        self.b = np.zeros((len(lbs),1))
        self.vertices = self.compute_vertices()


    def to_FVIM(self):
        """
        Convert CubeDomain object to FVIM

        Returns:
            An FVIM object
        """
        fmatrix = self.compute_fmatrix()
        return FVIM(fmatrix, self.vertices, self.dim, self.M, self.b)


    def compute_fmatrix(self):
        """
        Compute the matrix encoding the containment relation between facets and vertices of the cube

        Returns:
             fmatrix (np.ndarray): Matrix
        """
        facets_vertex = []
        combs = np.array([self.lbs_active, self.ubs_active]).T
        for n, vals in enumerate(combs):
            indx = self.dims_active[n]
            for val in vals:
                vs_facet = self.vertices[:,indx]==val
                facets_vertex.append(vs_facet)

        fmatrix = np.array(facets_vertex).transpose() # fmatrix: vertices, facets
        return fmatrix


    def compute_vertices(self):
        """
        Compute the vertices of the cube

        Returns:
             vertices (np.ndarray): Vertices of the cube
        """
        dim_vertices = len(self.dims_active) + len(self.dims_static)
        vertices = np.zeros((2**self.dim, dim_vertices))
        V = []
        for i in range(self.dim):
            V.append([self.lbs_active[i], self.ubs_active[i]])

        vertices[:, self.dims_active] = np.array(list(itertools.product(*V)))
        vertices[:, self.dims_static] = np.repeat([self.lbs_static], 2**self.dim, axis=0)

        return vertices

