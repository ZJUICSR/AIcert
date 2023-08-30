"""
These functions are used construct hypercubes in Flattice

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause

"""

import sys
import itertools
import numpy as np
import operator as op
from functools import reduce
import veritex.sets.facelattice as fl
import collections as cln

class CubeLattice:
    """
    A class for the set representation of a cube based on Face Lattice

        Attributes:
            dim (int): Dimensionality of the cube
            bs (np.ndarray): Intervals of dimensions
            lb (list): Lower bound of the cube
            ub (list): Upper bound of the cube
            poss (list): Target dimensions for reachability analysis
            vertices (np.ndarray): Vertices of the cube
            lattice (list): Containment relation bewteen (k)-dim faces and (k-1)-dim faces
            id_vals (dict): Mapping from faces to their references
            vertices_ref: Mapping from vertices to their references
            ref_vertex: Mapping from references to vertex
    """

    def __init__(self, lbs, ubs, poss=None):
        self.dim = len(lbs)
        self.bs = np.array([lbs, ubs]).T
        self.lbs = lbs
        self.ubs = ubs
        self.M = np.eye(len(lbs))
        self.b = np.zeros((len(lbs),1))
        self.poss = poss
        self.vertices = self.compute_vertex(self.lbs, self.ubs)
        self.lattice, self.id_vals, self.vertices_ref, self.ref_vertex = self.initial_lattice()
        for m in range(1,self.dim):
            self.single_dim_face(m)


    def to_FlatticeFFNN(self): #shape height x weight
        return fl.FlatticeFFNN(self.lattice, self.vertices, self.dim, self.M, self.b)


    def to_FlatticeCNN(self): #shape height x weight
        assert self.poss is not None
        self.vertices = self.vertices.reshape((self.vertices.shape[0], 3, self.poss[1][0]-self.poss[0][0]+1, self.poss[1][1]-self.poss[0][1]+1))
        return fl.FlatticeCNN(self.lattice, self.vertices, self.vertices, self.dim)


    def initial_lattice(self):
        lattice = []
        id_vals = []
        vertex_ref = cln.OrderedDict()
        ref_vertex = cln.OrderedDict()
        n = self.dim
        for m in range(self.dim):
            num = 2**(n-m)*self.ncr(n,m)
            d = cln.OrderedDict()
            val = cln.OrderedDict()
            for i in range(num):
                id = reference(i)
                d.update({id:[set(),set()]})
                val.update({id: [[],[]]})
                if m == 0:
                    vertex_ref.update({tuple(self.vertices[i]):id})
                    ref_vertex.update({id: self.vertices[i]})
            lattice.append(d)
            id_vals.append([])

        # self.dim level
        id = reference(-1)
        for key in lattice[-1].keys():
            lattice[-1][key][1].add(id)
        lattice.append(cln.OrderedDict({id:[set(list(lattice[-1].keys())), set()]}))

        return lattice, id_vals, vertex_ref, ref_vertex


    def compute_vertex(self, lb, ub):
        # compute vertex
        V = []
        for i in range(len(ub)):
            V.append([lb[i], ub[i]])

        return np.array(list(itertools.product(*V)))


    # update lattice of m_face
    def single_dim_face(self, m):
        num = 2 ** (self.dim - m) * self.ncr(self.dim, m)
        Varray = self.vertices
        ref_m = list(self.lattice[m].keys())
        ref_m_1 = list(self.lattice[m-1].keys())

        id_vals_temp = cln.OrderedDict()

        nlist = list(range(len(self.lbs)))
        element_id_sets = list(itertools.combinations(nlist, self.dim-m))
        c = 0
        for element_id in element_id_sets:
            # start_time = time.time()
            elem_id_m = np.array(element_id)
            vals = [list(self.bs[e,:]) for e in elem_id_m]
            faces = np.array(list(itertools.product(*vals)))

            diff_elem = np.setdiff1d(np.array(range(self.dim)), elem_id_m)

            for f in faces:
                f_m = np.ones((self.dim))*100
                for i in range(len(elem_id_m)):
                    f_m[elem_id_m[i]] = f[i]
                k_m = tuple(np.concatenate((elem_id_m, f_m)))
                id_m = ref_m[c]
                id_vals_temp.update({k_m: id_m})

                for i in diff_elem:
                    elem_id_m_1 = np.copy(elem_id_m)
                    elem_id_m_1 = np.sort(np.append(elem_id_m_1, i))
                    f_m_1 = np.copy(f_m)
                    # upper bound
                    f_m_1[i] = self.ubs[i]
                    k_m_1 = tuple(np.concatenate((elem_id_m_1, f_m_1)))
                    if m!=1:
                        id_m_1 = self.id_vals[m - 1][k_m_1]
                    else:
                        id_m_1 = self.vertices_ref[tuple(f_m_1)]

                    self.lattice[m][ref_m[c]][0].add(id_m_1)
                    self.lattice[m - 1][id_m_1][1].add(ref_m[c])

                    # lower bound
                    f_m_1[i] = self.lbs[i]
                    k_m_1 = tuple(np.concatenate((elem_id_m_1, f_m_1)))
                    if m != 1:
                        id_m_1 = self.id_vals[m - 1][k_m_1]
                    else:
                        id_m_1 = self.vertices_ref[tuple(f_m_1)]

                    self.lattice[m][ref_m[c]][0].add(id_m_1)
                    self.lattice[m - 1][id_m_1][1].add(ref_m[c])

                c = c+1

        self.id_vals[m] = id_vals_temp

        if c!=num:
            print('Computation is wrong')
            sys.exit(1)


    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return int(numer / denom)


class reference:
    """
    A class to create an unique id
    """
    def __init__(self, val):
        self._value = val  # just refers to val, no copy



def partition_input(input_range, pnum, poss):
    """
    Partition one input range into mulitple sub inputs in Flattice

    Parameters:
        input_range (list): Lower bounds and upper bounds
        pnum (int): Number of each sub ranges in each dimension
        poss (list): Positions of the pixel block

    Returns:
        all_inputs (list): All sub inputs in Flattice
    """
    mean, std = [0, 0, 0], [1, 1, 1],
    input_range[0] = ((input_range[0].transpose((1,2,0))-mean)/std).transpose((2,0,1))
    input_range[1] = ((input_range[1].transpose((1,2,0))-mean)/std).transpose((2,0,1))

    ranges = divide_range(input_range, pnum)
    range_comb = list(itertools.product(*ranges))
    all_inputs = []
    for acomb in range_comb:
        lbs = [e[0] for e in acomb]
        ubs = [e[1] for e in acomb]
        aset = CubeLattice(lbs, ubs, poss=poss).to_FlatticeCNN()
        all_inputs.append(aset)

    return all_inputs


def divide_range(input_range, pnum):
    """
    Split input ranges into sub ranges

    Parameters:
        input_range (list): Lower bounds and upper bounds
        pnum (int): Number of each sub ranges in each dimension

    Returns:
        subs (list): Sub ranges
    """
    minv = input_range[0].flatten()
    maxv = input_range[1].flatten()

    subs = []
    for i in range(len(minv)):
        alist = []
        min = minv[i]
        max = maxv[i]
        ave = (max - min) / pnum
        for n in range(pnum):
            if n == 0:
                temp = [min, min + ave]
            if n == pnum - 1:
                temp = [min + (pnum - 1) * ave, max]
            else:
                temp = [min + n * ave, min + (n + 1) * ave]
            alist.append(temp)

        subs.append(alist)

    return subs
