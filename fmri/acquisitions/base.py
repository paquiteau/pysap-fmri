#!/usr/bin/env python3
import os
import numpy as np
from  mapvbvd import mapVBVD
import functools


class BaseFMRIAcquisition:
    """
    Base Acquisition class,
    hold data and meta_data of sampling pattern and parameters. 
    """
    def __init__(self,kspace_data, kspace_loc):
        self.kspace_data = kspace_data
        self.kspace_data = kspace_loc

    def save(self, filename):
        np.savez(filename, kspace_data=self.kspace_data, kspace_loc=self.kspace_loc)



class CartesianAcquisition(BaseFMRIAcquisition):
    """ Cartesisan fMRI Acquisition. """
    pass
