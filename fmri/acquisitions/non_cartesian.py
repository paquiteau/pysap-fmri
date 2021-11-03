import functools
import os.path
import warnings

import numpy as np
from mapvbvd import mapVBVD

from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
from mri.operators.utils import normalize_frequency_locations
from mri.operators.fourier.non_cartesian import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from sparkling.utils.gradient import get_kspace_loc_from_gradfile

from .base import BaseFMRIAcquisition

class SparklingAcquisition(BaseFMRIAcquisition):
    """
    Acquisition class for Non Cartesian Acquisition, repeated in time.

    Notes
    -----
        The sampling pattern is assumed to be constant in time, and store in a Sparkling compliant format.

    Parameters
    ----------
    data_file: str
        `.dat` file in twix format , containing the experimental data.
    trajectory_file: str
        `.bin` file containing kspace_locations of Sparkling acquisition
    smaps_file: str (optional, default None)
        sensitivity profiles of coils. If empty and calibration is not False, they will be computed.
    frame_slicer: slice (optional, default None)
        Time frame selector, use it to reduce the memory footprint by only importing a subset of slices.
    load_data: bool (optional, default True)
        Flag to load data.
    normalize: bool (optional, default True)
        Flag to normalize the kspace frequencies between -0.5 and 0.5
    bin_load_kwargs: dict (optional, default None)
        Extra argument to load trajectory files.
        If not provided, the trajectory file will be imported once to extract correct parameters.
    calibration_kwargs: dict (optional, default None)
        Calibration parameters use to estimate the smaps if not provided.
    """
    def __init__(self, data_file, trajectory_file, smaps_file=None,
                 frame_slicer=None, load_data=True, normalize=True, calibrate=True,
                 calibration_kwargs:dict=None, bin_load_kwargs:dict=None) -> None:
        self._data_file = data_file
        self._trajectory_file = trajectory_file
        self._smaps_file = smaps_file
        if smaps_file is not None:
            calibrate = False
        self._twix_obj = None
        self.n_coils   = None
        self.n_frames  = None

        self.frame_slicer = frame_slicer
        # load .dat infos
        if load_data:
            _twix_obj = mapVBVD(self._data_file)
            traj_name = _twix_obj.hdr['Meas']['tFree']
            if self._trajectory_file is None:
                self._trajectory_file = os.path.join(os.path.dirname(self._data_file),traj_name)
            self.n_coils = _twix_obj.hdr['Meas']['NChaMeas']
            if traj_name not in self._trajectory_file:
                warnings.warn("The trajectory file specified in data_file is probably not the same as the one provided, using the latter.")
            self._twix_obj = _twix_obj
        print(".dat file red")
        # load .bin data and infos.
        if bin_load_kwargs is None:
            # HACK: the infos and data are retrieve twice to ensure the correct sampling rate.
            kspace_points, infos = get_kspace_loc_from_gradfile(self._trajectory_file)
            kspace_points, _     = get_kspace_loc_from_gradfile(self._trajectory_file,
                                                                dwell_time=0.01/infos['min_osf'],
                                                                num_adc_samples=infos['num_samples_per_shot']*infos['min_osf'])
        else:
            kspace_points, infos = get_kspace_loc_from_gradfile(self._trajectory_file, **bin_load_kwargs)

        self.n_shots   = infos['num_shots']
        self.n_samples = infos['num_samples_per_shot']
        self.FOV       = infos['FOV']
        self.DIM       = infos['dimension']
        self.img_size  = infos['img_size']
        self.OSF       = infos['min_osf']

        if normalize:
            self.kspace_loc = normalize_frequency_locations(kspace_points, Kmax=self.img_size/2*self.FOV)
        else:
            self.kspace_loc = kspace_points
        self.kspace_loc = np.reshape(self.kspace_loc, (self.kspace_loc.shape[0]*self.kspace_loc.shape[1],
                                                       self.kspace_loc.shape[2]))
        if calibrate:
            try:
                self.smaps = np.load(self._smaps_file)
            except:
                warnings.warn("Failed to load the smaps, using self calibration instead")
                self.smaps = self._computed_smaps(n_cpu=16)

    def _computed_smaps(self, use_slices=None, thresh=0.1, mode='gridding',
                  method='linear', density_comp=None, n_cpu=1,):
        """" Compute the sensitivity maps from fMRI data:
        a subset of frame is retrieve from the data, flatten and used for the estimation"""
        if use_slices is None:
            select_kspace = self.kspace_data
        else:
            select_kspace = self.kspace_data[use_slices,...]
        n_frame = select_kspace.shape[0]
        select_kspace = np.moveaxis(select_kspace,0,-1)
        select_kspace = np.reshape(select_kspace, (select_kspace.shape[0],
                                   select_kspace.shape[1]*select_kspace.shape[2]))
       # select_kspace = np.moveaxis(select_kspace,0,-1)
        samples = np.tile(self.kspace_loc, (n_frame,1))
        print(self.img_size, select_kspace.shape,samples.shape, samples.min(axis=0).shape)
        Smaps, _ = get_Smaps(select_kspace,
                             img_shape=self.img_size,
                             samples=samples,
                             min_samples=samples.min(axis=0),
                             max_samples=samples.max(axis=0),
                             thresh=(thresh,)*self.DIM,
                             mode=mode,
                             method=method,
                             density_comp=density_comp,
                             n_cpu=n_cpu)

        return Smaps


    @property
    @functools.lru_cache(maxsize=None)
    def kspace_data(self) -> np.ndarray:
        """ Get the sampled data in kspace.  N_frame x N_channel x OSF.N_shots.N_samples"""
        self._twix_obj.image.flagRemoveOS = False
        self._twix_obj.image.squeeze = True
        if self.frame_slicer:
            a = self._twix_obj.image[:,:,:,self.frame_slicer]
        else:
            a = self._twix_obj.image[""]
        a = np.swapaxes(a, 1, 2)
        self.n_coils = a.shape[2]
        self.n_frames = a.shape[3]
        a = np.reshape(a, (a.shape[0]*a.shape[1],a.shape[2],a.shape[3]))
        print("kspace_data imported")
        return a.T

    def get_fourier_operator(self, implementation='gpuNUFFT'):
        density_comp=estimate_density_compensation(self.kspace_loc,self.img_size,implementation=implementation)

        return NonCartesianFFT(samples=self.kspace_loc,
                               shape=self.img_size,
                               n_coils=self.n_coils,
                               implementation=implementation,
                               density_comp=density_comp
                               )

    def __repr__(self) -> str :
        return "SparklingAcquisition(\n"\
               f"shots={self.n_shots}\n"\
               f"samples/shots={self.n_samples}\n"\
               f"coils={self.n_coils}\n"\
               f"timeframes={self.n_frames}\n"\
               f"img_dim={self.img_dims}\n"\
               f"img_size={self.img_size}\n"\
               f"FOV={self.FOV}\n"\
               f"Oversampling={self.OSF}\n"\
               f")"

class CompressAcquisition(BaseFMRIAcquisition):
    """
    Acquisition class for compress acquisition both in space and time (eg 4D sparkling)
    TODO
    """
    pass
