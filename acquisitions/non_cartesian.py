import functools
import os.path
import warnings

import numpy as np
from mapvbvd import mapVBVD
from mri.operators.utils import normalize_frequency_locations
from sparkling.utils.gradient import get_kspace_loc_from_gradfile

from .base import BaseFMRIAcquisition


class NonCartesianAcquisition(BaseFMRIAcquisition):
    """
    Acquisition class for Non Cartesian Acquisition, repeated in time.
    """
    def __init__(self,*args,**kwargs):
        super(self).__init__(*args, load=True,  **kwargs,)
        self.fov = self._twix_obj.

    def load(self):
        return self.load_from_twix()

    def load_from_twix(self):
        self._twix_obj = mapVBVD(self._data_file)
        traj_name = self._twix_obj.hdr['Config']['tFree']
        if self._trajectory_file is None:
            self._trajectory_file = os.path.join(os.path.dirname(self._data_file),traj_name)

        if traj_name not in self._trajectory_file:
            warnings.warn("The trajectory file is probably not the same as ")
    

    @functools.lru_cache
    @property
    def kspace_data(self):
        self._twix_obj.image.flagRemoveOS = False
        self._twix_obj.image.squeeze = True
        a = self._twix_obj.image[""]
        a = np.swapaxes(a, 1, 2)
        return a.T

    @functools.lru_cache
    @property
    def kspace_loc(self, normalize=True):
        shots = get_kspace_loc_from_gradfile(self._trajectory_file,
                                                        dwell_t,
                                                        num_adc_samples)[0])
        shots = np.reshape(shots, (shots.shape[0] * shots.shape[1], shots.shape[2]))
        if normalize:
            sample_locations = normalize_frequency_locations(shots, Kmax=self.kmax)
        return sample_locations


class CompressAcquisition(BaseFMRIAcquisition):
    """
    Acquisition class for compress acquisition both in space and time (eg 4D sparkling)
    """
    pass
