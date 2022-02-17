# -*- coding: utf-8 -*-
"""Base module to import and  load fMRI data."""

import pickle
import numpy as np


class BaseFMRIAcquisition:
    """Base Acquisition class.

    Hold data and meta_data of sampling pattern and parameters.
    """

    def __init__(self, kspace_data, kspace_loc):
        self.kspace_data = kspace_data
        self.kspace_loc = kspace_loc

    def save(self, filename):
        """Save data."""
        np.savez(filename, kspace_data=self.kspace_data,
                 kspace_loc=self.kspace_loc)


class CartesianAcquisition(BaseFMRIAcquisition):
    """Cartesisan fMRI Acquisition."""
    pass
