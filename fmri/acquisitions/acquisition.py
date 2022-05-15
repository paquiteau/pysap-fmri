"""Pre process, create, save and load acquisition data."""
from dataclasses import dataclass
import numpy as np

import h5py

@dataclass(frozen=True)
class AcquisitionInfo:
    """Informations about an acquisition."""
    shape: tuple = None
    """The shape of the image space."""
    fov: np.ndarray = None
    """The field of view of the image space, specified in meters for
    each dimension."""
    n_samples: int = 1
    """The number of samples per shot."""
    n_coils: int = 1
    """The number of coil available."""
    n_frames: int = 1
    """The number of frames available."""
    normalize: str = "unit"
    """The normalization convention for the samples."""
    repeating: bool = False
    """If the trajectory is repeating at each frame."""
    @property
    def ndim(self):
        return len(self.shape)

@dataclass(frozen=False)
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

    @classmethod
    def load(cls, filename:str, frame_range=(0,0)):
        """Load an acquisition from the file.

        Parameters
        ---------
        filename: str
            The filename of the function
        frame_range: tuple
            A 2 or 3 element tuple, use to select a range of temporal frames.

        Returns
        -------
        Acquisition: the acquistion instance loaded with data.

        """
        fhandle = h5py.File(filename, 'r')
        infos = AcquisitionInfo(*fhandle.attrs.__dict__)
        if frame_range != (0,0):
            data = fhandle['data'][slice(*frame_range), ...]
        else:
            data = fhandle['data'][()]

        if frame_range != (0,0) and not infos.repeating:
            samples = fhandle['samples'][slice(*frame_range), ...]
        else:
            samples = fhandle['samples'][()]

        density = fhandle['density'][()] if 'density' in fhandle.keys() else None
        smaps = fhandle['smaps'][()] if 'smaps' in fhandle.keys() else None

        return cls(infos=infos,
              data=data,
              samples=samples,
              smaps=smaps,
              density=density)


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
