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
    def __init__(self, data_file, trajectory_file, load=True):
        self._data_file = data_file
        self._trajectory_file = trajectory_file
        if load:
            self.load()

    def load(self):
        pass

    @property
    def kspace_data(self):
        pass

    @property
    def kspace_loc(self):
        pass
    def get_data_op(self):
        pass


class CartesianAcquisition(BaseFMRIAcquisition):
    """ Cartesisan fMRI Acquisition. """
    pass
