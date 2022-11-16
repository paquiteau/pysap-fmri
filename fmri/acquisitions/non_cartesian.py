"""Processing of non cartesian data, in the Sparkling format."""
import warnings

import mapvbvd
import numpy as np
from mri.operators.fourier.utils.processing import normalize_frequency_locations
from mri.operators.fourier.orc_wrapper import compute_orc_coefficients

from sparkling.utils.gradient import get_kspace_loc_from_gradfile
from mri.operators.fourier.utils import normalize_frequency_locations
from ..utils import MAX_CPU_CORE
from ..operators import NonCartesianSpaceFourier


def process_raw_data(
    twix_obj,
    n_shot_per_frame: int = 0,
    frame_range: tuple = (0, 0),
):
    """Process the raw import data of twix format.

    The data is imported and reshaped

    Parameters
    ----------
    twix_obj : mapvbvd.MapVBVD
        The twix import object
    n_shot_per_frame : int
        the number of shot to put in each temporal frame,
        only use for non repeating trajectories.
    frame_range : tuple
        A range of frame to import specifies as (start, stop, [step])
    repeating_trajectory : bool
        If the trajectory associated is a repeating pattern. In this case the
        data is already formatted along the coil dimension.

    Returns
    -------
    data: np.ndarray
        The acquired data in shape (n_frame, n_coil, n_samples)
    """
    twix_obj.image.flagRemoveOS = False
    twix_obj.image.squeeze = True

    # only relevant frames are loaded from the raw data
    if frame_range != (0, 0):
        if n_shot_per_frame > 0:
            frame_range_shots = [n_shot_per_frame * f for f in frame_range]
            slicer = np.s_[:, :, slice(*frame_range_shots)]
        else:
            slicer = np.s_[:, :, :, slice(*frame_range)]
    else:
        slicer = ""
    data = np.ascontiguousarray((twix_obj.image[slicer]).T, dtype="complex64")

    if n_shot_per_frame > 0:
        data = np.reshape(data, (-1, n_shot_per_frame, *data.shape[1:]))

    def _load_samples(self, normalize=True, bin_load_kwargs=None):
        """Load k-space trajectory from .bin file."""
        if bin_load_kwargs is None:
            # HACK: the infos and data are retrieve twice to ensure the correct
            # sampling rate.
            self.samples, infos = get_kspace_loc_from_gradfile(self._traj_file)
            self.samples, _ = get_kspace_loc_from_gradfile(
                self._traj_file,
                dwell_time=0.01 / float(infos["min_osf"]),
                num_adc_samples=(infos["num_samples_per_shot"] * infos["min_osf"]),
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
                self.samples, Kmax=self.img_shape / (2 * self.FOV)
            )
        self.samples = self.samples.astype(np.float32)
        self.samples = self.samples.reshape(
            (np.prod(self.samples.shape[:2]), self.samples.shape[2])
        )

    def get_fourier_operator(self, **kwargs):
        """Return the fourier operator associated with this acquisition."""
        return NonCartesianSpaceFourier(
            samples=self.samples,
            shape=self.img_shape,
            n_frames=self.n_frames,
            n_coils=self.n_coils,
            **kwargs,
        )


def process_raw_samples(
    samples_file: str,
    n_shot_per_frame: int = 0,
    bin_load_kwargs: dict = None,
    frame_range: tuple = (0, 0),
):
    """Process the raw samples location file.

    The trajectory file is loaded following the Sparkling data format.

    Parameters
    ----------
    samples_file : str
        filepath for the samples location, in `.bin` format.
    n_shot_per_frame : int
        the number of shot per frame, use for fmri. it should a positive integer.
        If 0, a single frame is provided.
    frame_range : tuple
        A tuple specifing the range of frame to import.
    normalize : {'unit', 'pi'} default 'pi'
        The normalizaion convention for the samples.
        - If `'pi'` the samples are normalized in [-pi, pi]
        - If `'unit'` the samples are normalized in [-0.5, 0.5]

    bin_load_kwargs : dict
        argument to load the bin file.

    Returns
    -------
    samples: np.ndarray
        the samples location with shape (n_frames, n_shot_per_frame, dim)
    acq_infos: dict
        Partial infos about the acquisition.
    """

    if bin_load_kwargs is None:
        # the data needs to be parsed once to retrieve the right
        # import parameters from infos.
        samples, infos = get_kspace_loc_from_gradfile(samples_file)
        samples, _ = get_kspace_loc_from_gradfile(
            samples_file,
            dwell_time=0.01 / float(infos["min_osf"]),
            num_adc_samples=infos["num_samples_per_shot"] * infos["min_osf"],
        )
    else:
        samples, infos = get_kspace_loc_from_gradfile(samples_file, **bin_load_kwargs)
    nsps = infos["num_samples_per_shot"] * infos["min_osf"]
    samples = normalize_frequency_locations(
        samples,
        Kmax=infos["img_size"] / (2 * infos["FOV"]),
    )
    samples = samples.astype(np.float32)
    samples = np.reshape(samples, (-1, samples.shape[-1]))
    if n_shot_per_frame > 0:
        samples = np.reshape(
            samples,
            (samples.shape[0] // (n_shot_per_frame * nsps), -1, samples.shape[1]),
        )
    if frame_range != (0, 0) and samples.ndim == 3:
        samples = samples[range(*frame_range), ...]

    acq_infos = {
        "shape": np.asarray(infos["img_size"]),
        "fov": np.asarray(infos["FOV"]),
        "repeating": samples.ndim == 2,
        "TE": infos["TE"],
        "osf": infos["min_osf"],
    }

    return np.ascontiguousarray(samples), acq_infos


def process_raw_acquisition(
    data_file: str,
    samples_file: str,
    smaps_file: str = None,
    b0_file: str = None,
    shifts: tuple = None,
    n_shot_per_frame: int = 0,
    frame_range: tuple = (0, 0),
    normalize: str = "pi",
    save_to: str = "",
):
    """Process the data and samples files to create a acquisition object.

    The processing is done as follow:
    1. load the samples locations.0000

    Parameters
    ----------
    data_file : str
        The file containing the data, it should be a `.dat` following the twix format.
    samples_file : str
        The samples location, in Sparkling format.
    shifts : tuple
        The coordinat shift to apply to the data.
    n_shot_per_frame : int
        The number of shot to allow per frame.
    save_to: str
        If not empty, the path to save the newly create acquisition.

    Returns
    -------
    Acquisition: the newly created acquisition.
    """

    twix_obj = mapvbvd.mapVBVD(data_file)
    traj_name = twix_obj.hdr["Meas"]["tFree"]
    if traj_name not in samples_file:
        warnings.warn(
            "The trajectory file specified in data file does not "
            "have the same name than the one specified by "
            "`samples_file`."
        )

    # process data and samples
    samples, acq_info_d = process_raw_samples(
        samples_file,
        n_shot_per_frame=n_shot_per_frame,
        frame_range=frame_range,
    )

    data = process_raw_data(
        twix_obj,
        frame_range=frame_range,
        n_shot_per_frame=n_shot_per_frame,
    )
    # shift frequencies.
    #
    if len(shifts) != samples.shape[-1]:
        raise ValueError("Dimension mismatch")
    phi = np.zeros((samples.shape[:-1]))
    for i in range(samples.shape[-1]):
        phi += samples[..., i] * shifts[i]
    phi = np.exp(-2 * np.pi * 1j * phi, dtype="complex64")
    if phi.ndim == 1:
        data *= phi
    else:
        data *= phi[:, None, :]

    smaps = np.load(smaps_file) if smaps_file is not None else None
    b0_map = np.load(b0_file) if b0_file is not None else None

    if normalize == "pi":
        samples *= 2 * np.pi

    # Export the data structure
    n_frames, n_coils, n_samples_per_frame = data.shape
    acq_info = AcquisitionInfo(
        normalize=normalize,
        n_frames=n_frames,
        n_coils=n_coils,
        n_shot_per_frame=n_shot_per_frame,
        n_samples_per_frame=n_samples_per_frame,
        n_samples_per_shot=n_samples_per_frame // n_shot_per_frame,
        **acq_info_d,
    )

    acq = Acquisition(
        infos=acq_info,
        samples=samples,
        data=data,
        density=None,
        smaps=smaps,
    )
    if save_to != "":
        acq.save(save_to)
    return acq
