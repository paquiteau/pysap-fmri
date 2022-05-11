"""Facilitate import of raw fMRI data."""
import functools
import os
import warnings
import pickle
import numpy as np
from mapvbvd import mapVBVD
from sparkling.utils.gradient import get_kspace_loc_from_gradfile
from mri.operators.fourier.utils import normalize_frequency_locations
from ..utils import MAX_CPU_CORE
from ..operators import SpaceFourier


class BaseAcquisition:
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
    frame_slicer: tuple (optional, default None)
        Time frame selector, to only import a subset of slices.
    shifts: array_like
        The coordinates shift for the frequency locations.
    n_shot_per_frame: int, default -1
        The number of shot to associate to each temporal frame.
    normalize: bool (optional, default True)
        Flag to normalize the kspace frequencies between -0.5 and 0.5
    bin_load_kwargs: dict (optional, default None)
        Extra argument to load trajectory files.
        If not provided, auto extract the parameters.

    Attributes
    ----------
    data: ndarray
        Acquired data (n_frames, n_coils, n_shots*n_samples)
    samples: ndarray
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
        self._load_samples(normalize, bin_load_kwargs)
        self._load_data(frame_slicer, n_shot_per_frame)

    def save(self, filename):
        """Save object."""
        with open(filename, "wb") as filepickle:
            pickle.dump(self, filepickle)

    @classmethod
    def load(cls, filename):
        """Load pickled file."""
        with open(filename, "rb") as filepickle:
            return pickle.load(filepickle)

    def _load_data(self, frame_slicer, n_shot_per_frame=-1):
        """Load data from .dat file.

        Parameters
        ----------
        frame_slicer: tuple
            A 2-tuple for the start and end frame to extract from the data.
        n_shot_per_frame: int, default=-1
            Number of shot to attribute to each frame.
            If `-1` assumed a repeated trajectory.
        """
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

        # only relevant frames are loaded from the raw data
        if frame_slicer is not None:
            if n_shot_per_frame == -1:
                slicer = np.s_[:, :, :, frame_slicer[0]:frame_slicer[1]]
            else:
                slicer = np.s_[:,
                               :,
                               n_shot_per_frame * frame_slicer[0]:
                               n_shot_per_frame * frame_slicer[1]
                               ]
        else:
            slicer = ""
        self.data = np.ascontiguousarray((_twix_obj.image[slicer]).T,
                                                dtype="complex64")
        # If only one frame is acquired reshape accordingly.
        # add the frame dimension manually.
        if n_shot_per_frame != -1:
            self.data = np.reshape(
                self.data,
                (frame_slicer[1] - frame_slicer[0],
                 n_shot_per_frame,
                 *self.data.shape[1:])
            )

        self.data = self.data.swapaxes(1, 2)
        shape = self.data.shape
        self.data = self.data.reshape(*shape[:2],
                                                    np.prod(shape[2:]))
        if frame_slicer[1] - frame_slicer[0] == 1:
            self.data = np.reshape(
                self.data,
                (1, (self.data.shape[0] * self.data.shape[1]),
                 *self.data.shape[2:])
            )
        self.n_coils = int(_twix_obj.hdr["Meas"]["NChaMeas"])
        self.n_frames = len(self.data)

    def _load_samples(self, normalize=True, bin_load_kwargs=None):
        """Load k-space trajectory from .bin file."""
        if bin_load_kwargs is None:
            # HACK: the infos and data are retrieve twice to ensure the correct
            # sampling rate.
            self.samples, infos = get_kspace_loc_from_gradfile(
                self._traj_file)
            self.samples, _ = get_kspace_loc_from_gradfile(
                self._traj_file,
                dwell_time=0.01 / float(infos["min_osf"]),
                num_adc_samples=(
                    infos["num_samples_per_shot"] * infos["min_osf"]
                ),
            )
        else:
            self.samples, infos = get_kspace_loc_from_gradfile(
                self._traj_file, **bin_load_kwargs
            )

        self.n_shots = infos["num_shots"]
        self.n_samples_shot = infos["num_samples_per_shot"]
        self.FOV = np.asarray(infos["FOV"])
        self.DIM = infos["dimension"]
        self.img_shape = infos["img_size"]
        self.OSF = infos["min_osf"]
        if normalize:
            self.samples = normalize_frequency_locations(
                self.samples,
                Kmax=self.img_shape / (2 * self.FOV)
            )
        self.samples = self.samples.astype(np.float32)
        self.samples = self.samples.reshape(
            (np.prod(self.samples.shape[:2]), self.samples.shape[2]))

    def get_fourier_operator(self, **kwargs):
        """Return the fourier operator associated with this acquisition."""
        return SpaceFourier(
            samples=self.samples,
            shape=self.img_shape,
            n_frames=self.n_frames,
            n_coils=self.n_coils,
            **kwargs,
        )


class RepeatedAcquisition(BaseAcquisition):
    """fMRI reconstruction using a repeating sampling pattern."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.shifts) != self.samples.shape[-1]:
            raise ValueError("Dimension mismatch")

        phi = np.zeros((self.samples.shape[0]))
        for i in range(self.samples.shape[-1]):
            phi += self.samples[..., i] * self.shifts[i]
        phi = np.exp(-2 * np.pi * 1j * phi, dtype="complex64")
        if self.n_frames == 1:
            self.data = self.data * phi[None, :, None]
        else:
            self.data = self.data * phi

    def save(self, filename):
        """Save the data and sampling pattern."""
        np.savez(filename, data=self.data,
                 samples=self.samples)

    def save_pickle(self, filename):
        """Save object."""
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=4)

    def __repr__(self):
        """Display the main characteristic of the acquisition."""
        return (
            "SparklingAcquisition(\n"
            f"shots={self.n_shots}\n"
            f"samples/shots={self.n_samples_shot}\n"
            f"coils={self.n_coils}\n"
            f"timeframes={self.n_frames}\n"
            f"img_dim={self.DIM}\n"
            f"img_shape={self.img_shape}\n"
            f"FOV={self.FOV}\n"
            f"Oversampling={self.OSF}\n"
            f")"
        )


class CompressedAcquisition(BaseAcquisition):
    """
    Acquisition class for compressed acquisition both in space and time.

    The kspace data and location is loaded as single anatomical volume and
    partitionned in frames later.

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
    n_shot_per_frame: int, default -1
        The number of shot to associate to each temporal frame.
    normalize: bool (optional, default True)
        Flag to normalize the kspace frequencies between -0.5 and 0.5
    bin_load_kwargs: dict (optional, default None)
        Extra argument to load trajectory files.
        If not provided, auto extract the parameters.
    calibration_kwargs: dict (optional, default None)
        Calibration parameters use to estimate the smaps if not provided.

    See Also
    --------
        BaseSparklingAcquisition: parent class
        RepeatedAcquisition: sister class, for constant-in-time trajectories.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_shot_frame = kwargs['n_shot_per_frame']
        fr_slice = kwargs['frame_slicer']
        # also extract the frame slice for the locations.
        nsf = self.n_shot_frame * self.n_samples_shot * self.OSF
        self.samples = self.samples[nsf * fr_slice[0]:nsf * fr_slice[1]]
        self.samples = self.samples.reshape((self.n_frames, nsf, -1))
        if len(self.shifts) != self.samples.shape[-1]:
            raise ValueError("Dimension mismatch")

        phi = np.zeros(self.samples.shape[:-1])
        for i in range(self.samples.shape[-1]):
            phi += self.samples[..., i] * self.shifts[i]
        self.data *= np.exp(-2 * np.pi * 1j * phi,
                                   dtype="complex64")[:, None, ...]

    def __repr__(self):
        """Display the main characteristic of the acquisition."""
        return (
            "CompressedAcquisition(\n"
            f"shots={self.n_shots}\n"
            f"samples/shots={self.n_samples_shot}\n"
            f"shots/frames={self.n_shot_frame}\n"
            f"coils={self.n_coils}\n"
            f"timeframes={self.n_frames}\n"
            f"img_dim={self.DIM}\n"
            f"img_shape={self.img_shape}\n"
            f"FOV={self.FOV}\n"
            f"Oversampling={self.OSF}\n"
            f")"
        )
