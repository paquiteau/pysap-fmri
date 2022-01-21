import functools
import os
import warnings
import pickle
import numpy as np
from mapvbvd import mapVBVD
from mri.operators.fourier.non_cartesian import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from mri.operators.utils import normalize_frequency_locations
from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
from sparkling.utils.gradient import get_kspace_loc_from_gradfile

from .base import BaseFMRIAcquisition
from .utils import add_phase_kspace
from ..utils import MAX_CPU_CORE


class SparklingAcquisition(BaseFMRIAcquisition):
    """
    Acquisition class for Non Cartesian Acquisition, repeated in time.

    Notes
    -----
        The sampling pattern is assumed to be constant in time, and store in a Sparkling compliant format.

    Attributes
    ----------
    kspace_data: ndarray
        data acquired, shaped: (n_frames, n_coils, n_shots*n_samples)
    kspace_loc: ndarray
        samples points for the acquired data, shape (n_frame, n_shots*n_samples, DIM)
    n_frames: int
        Number of temporal frames of data acquisition
    n_coils: int
        Number of coils
    n_shots: int
        Number of shots
    n_samples: int
        Number of samples per shots
    FOV: ndarray
        Field of view in meters, for each dimension of the image
    img_shape: ndarray
        Size in pixel/voxel of the image
    OSF: int
        Oversampling factor
    density_comp: ndarray
        Density compensation estimation, computed and cached on the fly.
    """

    def __init__(self, data_file, trajectory_file, smaps_file=None, shifts=None,
                 frame_slicer=None, normalize=True, bin_load_kwargs: dict=None) -> None:
        """
        Init SparklingAcquisition.

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
        self._data_file = data_file
        self._traj_file = trajectory_file
        self._smaps_file = smaps_file
        self.shifts = shifts

        # load .dat infos
        _twix_obj = mapVBVD(self._data_file)
        traj_name = _twix_obj.hdr['Meas']['tFree']
        if self._traj_file is None:
            self._traj_file = os.path.join(
                os.path.dirname(self._data_file), traj_name)

        if traj_name not in self._traj_file:
            warnings.warn(
                "The trajectory file specified in data_file is probably not the same as the one provided, using the latter.")
        _twix_obj = _twix_obj
        _twix_obj.image.flagRemoveOS = False
        _twix_obj.image.squeeze = True

        if len(_twix_obj.image.sqzSize) > 4:
            raise ValueError("Data import with averaging is not available.")
        if frame_slicer is not None:
            self.kspace_data = _twix_obj.image[:, :, :, frame_slicer]
        else:
            self.kspace_data = _twix_obj.image[""]

        self.kspace_data = self.kspace_data.swapaxes(1, 2)
        self.kspace_data = self.kspace_data.swapaxes(0, 1)

        self.kspace_data = np.reshape(self.kspace_data,
                                      (self.kspace_data.shape[0]*self.kspace_data.shape[1],
                                       self.kspace_data.shape[2], self.kspace_data.shape[3]))
        self.kspace_data = self.kspace_data.T

        # load .bin data and infos.
        if bin_load_kwargs is None:
            # HACK: the infos and data are retrieve twice to ensure the correct sampling rate.
            self.kspace_loc, infos = get_kspace_loc_from_gradfile(
                self._traj_file)
            self.kspace_loc, _ = get_kspace_loc_from_gradfile(self._traj_file,
                                                              dwell_time=0.01/float(infos['min_osf']),
                                                              num_adc_samples=infos['num_samples_per_shot']*infos['min_osf'])
        else:
            self.kspace_loc, infos = get_kspace_loc_from_gradfile(
                self._traj_file, **bin_load_kwargs)

        self.kspace_loc = np.reshape(self.kspace_loc,
                                     (self.kspace_loc.shape[0]*self.kspace_loc.shape[1],
                                      self.kspace_loc.shape[2]))

        self.n_shots = infos['num_shots']
        self.n_samples = infos['num_samples_per_shot']
        self.FOV = np.asarray(infos['FOV'])
        self.DIM = infos['dimension']
        self.img_shape = infos['img_size']
        self.OSF = infos['min_osf']
        self.n_coils = int(_twix_obj.hdr['Meas']['NChaMeas'])
        self.n_frames = self.kspace_data.shape[0]

        if normalize:
            self.kspace_loc = normalize_frequency_locations(
                self.kspace_loc, Kmax=self.img_shape/(2*self.FOV))

        self.kspace_data = add_phase_kspace(
            self.kspace_data, self.kspace_loc, shifts=shifts)

    def get_smaps(self, use_rep=0, thresh=0.1, window_fun=None, mode='gridding',
                method='linear', density_comp=None, n_cpu=MAX_CPU_CORE, ssos=False, **kwargs):
        """Get Smaps from the center of kspace."""
        if type(thresh) is float:
            thresh = (thresh,)*self.DIM
        data = self.kspace_data[use_rep, ...]

        smaps, sos = get_Smaps(data,
                               img_shape=self.img_shape,
                               samples=self.kspace_loc,
                               thresh=thresh,
                               min_samples=self.kspace_loc.min(axis=0),
                               max_samples=self.kspace_loc.max(axis=0),
                               mode=mode,
                               window_fun=window_fun,
                               method=method,
                               density_comp=density_comp,
                               n_cpu=n_cpu,
                               fourier_op_kwargs=kwargs)
        if ssos:
            return smaps, ssos
        return smaps

    @property
    @functools.lru_cache(maxsize=None)
    def density_comp(self):
        """
        Estimate the density compensation.

        The results is cached for performance.
        """
        return estimate_density_compensation(self.kspace_loc, self.img_shape)

    def get_fourier_operator(self, implementation='gpuNUFFT', **kwargs):
        """Get the fourier operator associated to sampling pattern."""
        return NonCartesianFFT(samples=self.kspace_loc,
                               shape=self.img_shape,
                               n_coils=self.n_coils,
                               implementation=implementation,
                               density_comp=self.density_comp if implementation == 'gpuNUFFT' else None,
                               **kwargs,
                               )

    def save(self, filename):
        """Save the kspace_data and sampling pattern."""
        np.savez(filename, kspace_data=self.kspace_data,
                 kspace_loc=self.kspace_loc)

    def save_pickle(self, filename):
        """Save object."""
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=4)

    def __repr__(self):
        return "SparklingAcquisition(\n"\
               f"shots={self.n_shots}\n"\
               f"samples/shots={self.n_samples}\n"\
               f"coils={self.n_coils}\n"\
               f"timeframes={self.n_frames}\n"\
               f"img_dim={self.DIM}\n"\
               f"img_shape={self.img_shape}\n"\
               f"FOV={self.FOV}\n"\
               f"Oversampling={self.OSF}\n"\
               f")"


class CompressAcquisition(BaseFMRIAcquisition):
    """
    Acquisition class for compress acquisition both in space and time (eg 4D sparkling)
    TODO
    """
    pass
