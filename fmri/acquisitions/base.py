# -*- coding: utf-8 -*-
"""Base module to import and  load fMRI data."""

import pickle


class BaseFMRIAcquisition:
    """Base Acquisition class.

    Hold data and meta_data of sampling pattern and parameters.
    """

    def __init__(self, kspace_data, kspace_loc):
        self.kspace_data = kspace_data
        self.kspace_data = kspace_loc

    def save(self, filename):
        """Save data."""
        np.savez(filename, kspace_data=self.kspace_data,
                 kspace_loc=self.kspace_loc)

    def save_pickle(self, filename):
        """Save object."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filename):
        """Load pickled file."""
        filepickle = open(filename, "rb")
        return pickle.load(filepickle)


class CartesianAcquisition(BaseFMRIAcquisition):
    """Cartesisan fMRI Acquisition."""
    pass
