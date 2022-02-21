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

from .utils import add_phase_kspace
from ..utils import MAX_CPU_CORE
from ..operators.fourier import SpaceFourierMulti


class BaseSparklingAcquisition:
    """
    Acquisition class for Non Cartesian Acquisition, repeated in time.

    Notes
    -----
        The sampling pattern is assumed to be constant in time,
        and store in a Sparkling compliant format.

    Parameters
    ----------
    data_file: str
        `.dat` file in twix format , containing the experimental data.
    trajectory_file: str
        `.bin` file containing kspace_locations of Sparkling acquisition
    smaps_file: str (optional, default None)
        sensitivity profiles of coils.
        If empty and calibration is not False, they will be computed.
    frame_slicer: tuple (optional, default None)
        Time frame selector, to only import a subset of slices.
    load_data: bool (optional, default True)
        Flag to load data.
    normalize: bool (optional, default True)
        Flag to normalize the kspace frequencies between -0.5 and 0.5
    bin_load_kwargs: dict (optional, default None)
        Extra argument to load trajectory files.
        If not provided, auto extract the parameters.
    calibration_kwargs: dict (optional, default None)
        Calibration parameters use to estimate the smaps if not provided.

    Attributes
    ----------
    kspace_data: ndarray
        Acquired data (n_frames, n_coils, n_shots*n_samples)
    kspace_loc: ndarray
        Samples locations (n_frame, n_shots*n_samples, DIM)
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
    """

    def __init__(
            self,
            data_file,
            trajectory_file,
            frame_slicer=None,
            shifts=None,
            normalize=True,
            n_shot_per_frame=-1,
            bin_load_kwargs=None):

        self._data_file = data_file
        self._traj_file = trajectory_file
        self.shifts = shifts
        self._load_kspace_loc(normalize, bin_load_kwargs)
        self._load_kspace_data(frame_slicer, n_shot_per_frame)

    def save_pickle(self, filename):
        """Save object."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filename):
        """Load pickled file."""
        filepickle = open(filename, "rb")
        return pickle.load(filepickle)

    def _load_kspace_data(self, frame_slicer, n_shot_per_frame=-1):
        """Load data from .dat file."""
        _twix_obj = mapVBVD(self._data_file)
        traj_name = _twix_obj.hdr["Meas"]["tFree"]
        if self._traj_file is None:
            self._traj_file = os.path.join(
                os.path.dirname(self._data_file), traj_name)

        if traj_name not in self._traj_file:
            warnings.warn(
                "The trajectory file specified in data_file is probably not "
                "the same as the one provided, using the latter."
            )
        _twix_obj.image.flagRemoveOS = False
        _twix_obj.image.squeeze = True

        if len(_twix_obj.image.sqzSize) > 4:
            raise ValueError(
                "Data import with averaging is not available.")

        if frame_slicer is not None:
            if n_shot_per_frame == -1:
                m = np.s_[:, :, :, frame_slicer[0]:frame_slicer[1]]
            else:
                m = np.s_[:,
                          :,
                          n_shot_per_frame * frame_slicer[0]:
                          n_shot_per_frame * frame_slicer[1]
                          ]
        else:
            m = ""
        self.kspace_data = np.ascontiguousarray((_twix_obj.image[m]).T)
        # add the frame dimension manually.
        if n_shot_per_frame != -1:
            self.kspace_data = np.reshape(
                self.kspace_data,
                (frame_slicer[1] - frame_slicer[0],
                 n_shot_per_frame,
                 *self.kspace_data.shape[1:])
            )

        self.kspace_data = self.kspace_data.swapaxes(1, 2)
        s = self.kspace_data.shape
        self.kspace_data = self.kspace_data.reshape(*s[:2], np.prod(s[2:]))
        self.n_coils = int(_twix_obj.hdr["Meas"]["NChaMeas"])
        self.n_frames = len(self.kspace_data)

    def _load_kspace_loc(self, normalize=True, bin_load_kwargs=None):
        """Load k-space trajectory from .bin file."""
        if bin_load_kwargs is None:
            # HACK: the infos and data are retrieve twice to ensure the correct
            # sampling rate.
            self.kspace_loc, infos = get_kspace_loc_from_gradfile(
                self._traj_file)
            self.kspace_loc, _ = get_kspace_loc_from_gradfile(
                self._traj_file,
                dwell_time=0.01 / float(infos["min_osf"]),
                num_adc_samples=infos["num_samples_per_shot"] *
                infos["min_osf"],
            )
        else:
            self.kspace_loc, infos = get_kspace_loc_from_gradfile(
                self._traj_file, **bin_load_kwargs
            )

        self.n_shots = infos["num_shots"]
        self.n_samples = infos["num_samples_per_shot"]
        self.FOV = np.asarray(infos["FOV"])
        self.DIM = infos["dimension"]
        self.img_shape = infos["img_size"]
        self.OSF = infos["min_osf"]
        if normalize:
            self.kspace_loc = normalize_frequency_locations(
                self.kspace_loc,
                Kmax=self.img_shape / (2 * self.FOV)
            )

    def get_fourier_operator(self, **kwargs):
        print(self.img_shape, self.n_coils, self.kspace_data.shape)
        return SpaceFourierMulti(
            self.img_shape,
            self.n_coils,
            samples=self.kspace_data,
            n_frames=self.n_frames,
            **kwargs,
        )


class SparklingAcquisition(BaseSparklingAcquisition):
    """Sparkling fMRI reconstruction using a repeating sampling pattern."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_frames = len(self.kspace_data)
        self.kspace_loc = self.kspace_loc.reshape(
            (np.prod(self.kspace_loc.shape[:2]), self.kspace_loc.shape[2]))

        print(self.kspace_data.shape)
        print(self.kspace_loc.shape)

        if len(self.shifts) != self.kspace_loc.shape[-1]:
            raise ValueError("Dimension mismatch")
        phi = np.zeros((self.kspace_loc.shape[0]))
        for i in range(self.kspace_loc.shape[-1]):
            phi += self.kspace_loc[0, ..., i] * self.shifts[i]
        self.kspace_data *= np.exp(-2 * np.pi * 1j * phi)

    def get_smaps(
        self,
        use_rep=0,
        thresh=0.1,
        window_fun=None,
        mode="gridding",
        method="linear",
        density_comp=None,
        n_cpu=MAX_CPU_CORE,
        ssos=False,
        **kwargs,
    ):
        """Get Smaps from the center of kspace."""
        if type(thresh) is float:
            thresh = (thresh,) * self.DIM
        data = self.kspace_data[use_rep, ...]

        smaps, sos = get_Smaps(
            data,
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
            fourier_op_kwargs=kwargs,
        )
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

    def save(self, filename):
        """Save the kspace_data and sampling pattern."""
        np.savez(filename, kspace_data=self.kspace_data,
                 kspace_loc=self.kspace_loc)

    def save_pickle(self, filename):
        """Save object."""
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=4)

    def __repr__(self):
        return (
            "SparklingAcquisition(\n"
            f"shots={self.n_shots}\n"
            f"samples/shots={self.n_samples}\n"
            f"coils={self.n_coils}\n"
            f"timeframes={self.n_frames}\n"
            f"img_dim={self.DIM}\n"
            f"img_shape={self.img_shape}\n"
            f"FOV={self.FOV}\n"
            f"Oversampling={self.OSF}\n"
            f")"
        )


class CompressedAcquisition(BaseSparklingAcquisition):
    """
    Acquisition class for compressed acquisition both in space and time.

    The kspace data and location is loaded as single anatomical volume and
    partitionned in frames later.

    See Also
    --------
        BaseSparklingAcquisition: parent class
        SparklingAcquisition: sister class, for constant-in-time trajectories.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_shot_per_frame = kwargs['n_shot_per_frame']
        frame_slicer = np.asarray(kwargs['frame_slicer'])
        # also extract the frame slice for the locations.
        self.kspace_loc = self.kspace_loc[n_shot_per_frame * frame_slicer[0]:
                                          n_shot_per_frame * frame_slicer[1],
                                          ...]

        self.kspace_loc = self.kspace_loc.reshape(
            np.prod(self.kspace_loc.shape[:2]),
            *self.kspace_loc.shape[2:])

        self.kspace_loc = self.kspace_loc.reshape(
            self.n_frames,
            len(self.kspace_loc) // self.n_frames,
            *self.kspace_loc.shape[1:]
        )

        print(self.kspace_data.shape)
        print(self.kspace_loc.shape)

        if len(self.shifts) != self.kspace_loc.shape[-1]:
            raise ValueError("Dimension mismatch")
        phi = np.zeros(self.kspace_loc.shape[:-1])
        for i in range(self.kspace_loc.shape[-1]):
            phi[...] += self.kspace_loc[..., i] * self.shifts[i]
        self.kspace_data *= np.exp(-2 * np.pi * 1j * phi)[:, None, ...]

    def __repr__(self):
        """Show basic infos about the acquisition."""
        return (
            "CompressedAcquisition(\n"
            f"shots={self.n_shots}\n"
            f"samples/shots={self.n_samples}\n"
            f"coils={self.n_coils}\n"
            f"timeframes={self.n_frames}\n"
            f"img_dim={self.DIM}\n"
            f"img_shape={self.img_shape}\n"
            f"FOV={self.FOV}\n"
            f"Oversampling={self.OSF}\n"
            f")"
        )
