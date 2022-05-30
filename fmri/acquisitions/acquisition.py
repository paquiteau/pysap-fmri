"""Pre process, create, save and load acquisition data."""
from dataclasses import dataclass
import warnings
import numpy as np

import h5py

@dataclass
class AcquisitionInfo:
    """Informations about an acquisition.
    ----------
    """
    shape: np.ndarray = None
    """The shape of the image space."""
    fov: np.ndarray = None
    """The field of view of the image space, specified in meters for
    each dimension."""
    n_samples_per_shot: int = 1
    """The number of samples per shot."""
    n_shot_per_frame: int = 1
    """The number of shot per frame."""
    n_samples_per_frame: int = 1
    """The  number of samples per frame."""
    n_coils: int = 1
    """The number of coil available."""
    n_frames: int = 1
    """The number of frames available."""
    n_interpolator: int = 0
    """The number of interpolator used for the field correction."""
    TE: float = 0.0
    """The TE for each shot."""
    osf: int = 0
    """The ADC oversampling factor"""
    normalize: str = "unit"
    """The normalization convention for the samples."""
    repeating: bool = False
    """If the trajectory is repeating at each frame."""

    @property
    def ndim(self):
        return len(self.shape)


@dataclass
class Acquisition:
    """The information about """
    infos: AcquisitionInfo = None
    """Information about the acquisition"""
    samples: np.ndarray = None
    """The samples locations ([n_frames] x n_samples x ndim)"""
    data: np.ndarray = None
    """The samples values ([n_frames] x n_coils x n_samples)"""
    density: np.ndarray = None
    """The samples density compensation values (n_samples)"""
    smaps: np.ndarray = None
    """The smaps (n_coils x *shape)"""
    b0_map: np.ndarray = None
    """The Field inhomogeneity map."""
    image_field_correction: np.ndarray = None
    """The image-side field correction for each interpolator."""
    kspace_field_correction: np.ndarray = None
    """The kspace-side field corection for each interpolator."""


    def __repr__(self):
        ret = f"Acquisition(\n"
        max_len= max((len(k) for k in self.__dict__))
        for attr_name, attr_val in self.infos.__dict__.items():
            ret += " "*4 + f"{attr_name}={attr_val},\n"
        for arr_name, arr_val in self.__dict__.items():
            if arr_name == "infos": continue
            if hasattr(arr_val, "shape") and hasattr(arr_val, "dtype") :
                ret += f"{arr_name:{max_len}}: {arr_val.dtype}{arr_val.shape},\n"
            else:
                ret += f"{arr_name:{max_len}}: {arr_val},\n"
        ret = ret[:-2] + ")"
        return ret

    @classmethod
    def load(cls, filename:str, frame_range=(0,0), no_data=False, no_smaps=False):
        """Load an acquisition from the file.

        Parameters
        ---------
        filename: str
            The filename of the function
        frame_range: tuple
            A 2 or 3 element tuple, use to select a range of temporal frames.
        no_data: bool
            If the data should not be loaded.
        no_smaps: bool
            If the smaps should not be loaded.
        Returns
        -------
        Acquisition: the acquistion instance loaded with data.

        """
        fhandle = h5py.File(filename, 'r')

        infos = AcquisitionInfo()
        for k in infos.__dict__.keys():
            setattr(infos,k, fhandle.attrs[k])
        s_frame = ()
        if frame_range != (0,0):
            s_frame = np.s_[slice(*frame_range), ...]
            infos.n_frames = len(range(*frame_range))
        # initialise empty slice selector for all fields.
        arr_dict = {k : ()  for k in cls.__annotations__ if k != "infos"}
        arr_dict['data'] = s_frame
        if not infos.repeating:
            arr_dict['samples'] = s_frame
            arr_dict['density'] = s_frame
        if no_data: arr_dict.pop("data")
        if no_smaps: arr_dict.pop("smaps")

        for arr_name, arr_slice in arr_dict.items():
            try:
                arr_dict[arr_name] = fhandle[arr_name][arr_slice]
            except:
                arr_dict[arr_name] = None
        return cls(infos=infos, **arr_dict)

    def save(self, filename:str) -> None:
        """Save the data to disk in a archive file.
        """
        fhandle = h5py.File(filename, 'a')
        # set the metadata
        for key, val in self.infos.__dict__.items():
            fhandle.attrs[key] = val

        fhandle.create_dataset('data',
                               data=self.data,
                               compression="gzip",
                               )
        fhandle.create_dataset('samples',
                               data=self.samples,
                               compression="gzip",
                               )
        if self.smaps is not None:
            fhandle.create_dataset('smaps',
                                   data=self.samples,
                                   compression="gzip",
                                   )
        if self.density is not None:
            fhandle.create_dataset('density',
                                   compression="gzip",
                                   data=self.density)
        fhandle.close()

    def extract_frames(self, frame_range):
        if self.samples.ndim == 3:
            samples = self.samples[slice(*frame_range), ...]
        data = self.data[slice(*frame_range), ...]


        return Acquisition(
            infos=AcquisitionInfo(*self.infos.__dict__),
            data=data,
            samples=samples,
            density=self.density,
            smaps=self.smaps,
        )
