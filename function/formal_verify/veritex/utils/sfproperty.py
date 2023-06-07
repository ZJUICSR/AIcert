"""
These functions are used to construct safety properties for DNNs

Authors: Xiaodong Yang, xiaodong.yang@vanderbilt.edu
License: BSD 3-Clause


"""

import sys
from veritex.sets.cubedomain import CubeDomain
from veritex.sets.cubelattice import CubeLattice
from veritex.sets.vzono import VzonoFFNN

class Property:
    """
    A class for the safety property of a neural network

        Attributes:
            lbs (list): Lower bound of the input domain
            ubs (list): Upper bound of the input domain
            set_type (str): Name of the set representation
            input_set (FVIM or Flattice): Input set constructed by a set representation
            unsafe_domains (list): A set of unsafe output domains of the neural network
            input ranges (list): Entire input range to the neural network
    """
    def __init__(self, input_domain: list, unsafe_output_domains: list, input_ranges=None, set_type='FVIM'):
        """
        Construct the attributes for a Property object

        Parameters:
            input_domain (list): Lower and Upper bounds of the input domain
            unsafe_output_domains (list): Unsafe output domains using sets of linear inequalities
            input_ranges (list): Entire input range to the network
            set_type (str): Name of the set representation that is used to construct the input set
        """

        assert len(input_domain)!=0
        self.lbs = input_domain[0]
        self.ubs = input_domain[1]

        self.set_type = set_type
        self.construct_input()
        self.unsafe_domains = unsafe_output_domains
        self.input_ranges = input_ranges


    def construct_input(self):
        """
        Construct the input set with a set representation.
        """

        if self.set_type == 'FVIM':
            box = CubeDomain(self.lbs, self.ubs)
            self.input_set = box.to_FVIM()
        elif self.set_type == 'FlatticeFFNN':
            box = CubeLattice(self.lbs, self.ubs)
            self.input_set = box.to_FlatticeFFNN()
        elif self.set_type == 'FlatticeCNN':
            box = CubeLattice(self.lbs, self.ubs)
            self.input_set = box.to_FlatticeCNN()
        elif self.set_type == 'Vzono':
            self.input_set = VzonoFFNN()
            self.input_set.create_from_bounds(self.lbs, self.ubs)
        else:
            sys.exit("This set type is not supported.")
