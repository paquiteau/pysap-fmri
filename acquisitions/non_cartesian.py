import functools
import os.path
import warnings

import numpy as np
from mapvbvd import mapVBVD
from mri.operators.utils import normalize_frequency_locations
from mri.operators.fourier import NonCartesianFFT
from sparkling.utils.gradient import get_kspace_loc_from_gradfile

from .base import BaseFMRIAcquisition

class NonCartesianAcquisition:
    """
    Acquisition class for Non Cartesian Acquisition, repeated in time.
    """
    def __init__(self, data_file, trajectory_file, load=True, normalize=True, bin_load_kwargs:dict=None):
        self._data_file = data_file
        self._trajectory_file = trajectory_file

        self._twix_obj = None
        if load:
            self._load_from_twix()
            if bin_load_kwargs is None:
                bin_load_kwargs = dict()
            self._load_from_bin(normalize, **bin_load_kwargs)

    def _load_from_bin(self, normalize=True, **kwargs):
        kspace_points, infos = get_kspace_loc_from_gradfile(self._trajectory_file, **kwargs)
        self.n_shots   = infos['num_shots']
        self.n_samples = infos['num_samples_per_shot']
        self.FOV       = infos['FOV']
        self.img_dims  = infos['dimension']
        self.img_size  = infos['img_size']
        self.OSF       = infos['min_osf']

        if normalize:
            self.kspace_loc = normalize_frequency_locations(kspace_points, Kmax=self.img_size/2*self.FOV)
        else:
            self.kspace_loc = kspace_points

    def _load_from_twix(self):
        self._twix_obj = mapVBVD(self._data_file)
        traj_name = self._twix_obj.hdr['Meas']['tFree']
        if self._trajectory_file is None:
            self._trajectory_file = os.path.join(os.path.dirname(self._data_file),traj_name)
        self.n_coils = self._twix_obj.hdr['Meas']['NChaMeas']
        if traj_name not in self._trajectory_file:
            warnings.warn("The trajectory file is probably not the same as ")

    @functools.cached_property
    def kspace_data(self):
        """ Get the sampled data in kspace. Format in Time x N_channel X N_shots x N_samples"""
        self._twix_obj.image.flagRemoveOS = False
        self._twix_obj.image.squeeze = True
        a = self._twix_obj.image[""]
        a = np.swapaxes(a, 1, 2)
        return a.T

    def get_operator(self):
        return NonCartesianFFT(self.kspace_loc, self.img_size, implementation="gpuNUFFT",n_coils=self.n_coils)

    def reconstruct(self, method="ADMM", solver_params):
        pass

class CompressAcquisition(BaseFMRIAcquisition):
    """
    Acquisition class for compress acquisition both in space and time (eg 4D sparkling)
    TODO
    """
    pass
