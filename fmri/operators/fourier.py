"""Fourier Operator for fMRI data."""
import abc
import numpy as np
import scipy as sp
import cupy as cp

MRI_CUFINUFFT_AVAILABLE = True
try:
    from mriCufinufft import MRICufiNUFFT
except ImportError:
    MRI_CUFINUFFT_AVAILABLE = False


class SpaceFourierBase(abc.ABC):
    """Spatial Fourier Transform on fMRI data.

    Parameters
    ----------
    shape: tuple
        Dimensions of the FFT
    n_frames: int
        Number of frames for the reconstruction
    n_coils: int
        Number of coils for pMRI, default 1.
    smaps: np.ndarray
        Sensitivity Maps, shared across time.

    Attributes
    ----------
    fourier_ops: list
        List of Fourier Operator
    """

    def __init__(self, shape, n_coils=1, n_frames=1, smaps=None):

        if (
            smaps is not None
            and n_coils != len(smaps)
            and smaps.shape[:-1] == tuple(shape)
        ):
            raise ValueError("smaps should  have dimension n_coils x shape")

        self.n_frames = n_frames
        self.n_coils = n_coils
        self.smaps = smaps
        self.shape = shape
        self.fourier_ops = []

    def op(self, data):
        """Forward Operator method."""
        adj_data = np.squeeze(
            np.zeros((self.n_frames, self.n_coils, *self.shape), dtype="complex64")
        )
        for i in range(self.n_frames):
            adj_data[i] = self.fourier_ops[i].op(data[i])
        return adj_data

    def adj_op(self, adj_data):
        """Adjoint Operator method."""
        if self.smaps is None:
            data = np.squeeze(
                np.zeros(
                    (self.n_frames, self.n_coils, *self.shape), dtype=adj_data.dtype
                )
            )
        else:

            data = np.squeeze(
                np.zeros((self.n_frames, *self.shape), dtype=adj_data.dtype)
            )

        for i in range(self.n_frames):
            data[i] = self.fourier_ops[i].adj_op(adj_data[i])
        return data


class NonCartesianSpaceFourier(SpaceFourierBase):
    """Spatial Fourier Transform on fMRI data.

    Parameters
    ----------
    samples: np.ndarray
        2D or 3D array of samples coordinates for non cartesian fourier
    n_frames: int
        Number of frames for the reconstruction
    n_coils: int
        Number of coils for pMRI, default 1.
    smaps: np.ndarray
        Sensitivity Maps, shared across time.
    estimate_density: 'gpu' | 'cpu'
        Method to estimate the density compensation.

    Attributes
    ----------
    fourier_ops: list
        List of NonCartesianFFT Operator
    """

    def __init__(
        self,
        samples,
        shape,
        n_coils=1,
        n_frames=1,
        smaps=None,
        smaps_cached=True,
        estimate_density=True,
        **kwargs
    ):
        if not MRI_CUFINUFFT_AVAILABLE:
            raise RuntimeError("MRICufinufft is not available.")

        super().__init__(shape, n_coils, n_frames, smaps)

        if samples.ndim == 2 and n_frames == 0:
            raise ValueError(
                "2D array of samples provided, but n_frames is not specified."
            )
        if samples.ndim == 2:
            self.samples = np.repeat(samples[None, ...], n_frames, axis=0)
            self.n_samples_per_frame = samples.shape[0]
        elif samples.ndim == 3:
            self.samples = samples
            self.n_samples_per_frame = samples.shape[1]
        else:
            raise ValueError("samples array should be 2D or 3D.")

        if density is True and samples.ndim == 2:
            density = MRICufiNUFFT.estimate_density(samples, self.shape, n_iter=20)
        else:
            density = density
        self.fourier_ops = []
        for i in range(n_frames):
            self.fourier_ops.append(
                MRICufiNUFFT(
                    self.samples[i],
                    shape,
                    n_coils=n_coils,
                    smaps=cp.array(smaps, copy=False)
                    if smaps is not None and smaps_cached
                    else smaps,
                    smaps_cached=smaps_cached,
                    density=density,
                    **kwargs,
                )
            )

    @classmethod
    def from_acquisition(cls, acquisition, orc=True, **kwargs):
        if "density" in kwargs.keys():
            density = kwargs.pop("density")
        else:
            density = acquisition.density
        fop = cls(
            samples=acquisition.samples,
            shape=acquisition.infos.shape,
            n_coils=acquisition.infos.n_coils,
            n_frames=acquisition.infos.n_frames,
            smaps=acquisition.smaps,
            density=density,
            **kwargs,
        )
        if acquisition.infos.n_interpolator > 0 and orc:
            fop.add_field_correction(
                acquisition.b0_map,
                acquisition.image_field_correction,
                acquisition.kspace_field_correction,
            )
        return fop

    def add_field_correction(self, field_map, image_field_cor, kspace_field_cor):
        """Add off-resonance field correction to each frames."""
        if field_map.ndim == 2:
            # use the same correction for all frame
            image_field_cor_d = cp.array(image_field_cor)
            kspace_field_cor_d = cp.array(kspace_field_cor)
            n_bins = image_field_cor.shape[-1]
            range_w = (np.min(field_map), np.max(field_map))
            scale = (range_w[1] - range_w[0]) / n_bins
            scale = scale if (scale != 0) else 1
            indices = np.around((field_map - range_w[0]) / scale).astype(int)
            indices = np.clip(indices, 0, n_bins - 1)

            for i in range(self.n_frames):
                self.fourier_ops[i] = MRIFourierCorrected(
                    self.fourier_ops[i], kspace_field_cor_d, image_field_cor_d, indices
                )
        else:
            raise NotImplementedError


class TimeFourier:
    """Temporal Fourier Transform on fMRI data."""

    def __init__(self, roi=None):
        super().__init__()
        self.roi = roi

    def op(self, x):
        """Forward Operator method..

        Apply the fourier transform on the time axis, voxel wise.
        """
        y = np.zeros_like(x)
        if self.roi is not None:
            y[:, self.roi] = sp.fft.ifftshift(
                sp.fft.fft(
                    sp.fft.fftshift(x[:, self.roi], axes=0), axis=0, norm="ortho"
                ),
                axes=0,
            )
        else:
            y = sp.fft.ifftshift(
                sp.fft.fft(sp.fft.fftshift(x, axes=0), axis=0, norm="ortho"), axes=0
            )
        return y

    def adj_op(self, x):
        """Adjoint Operator method.

        Apply the Inverse fourier transform on the time axis, voxel wise
        """
        y = np.zeros_like(x)
        if self.roi is not None:
            y[:, self.roi] = sp.fft.fftshift(
                sp.fft.ifft(
                    sp.fft.ifftshift(x[:, self.roi], axes=0), axis=0, norm="ortho"
                ),
                axes=0,
            )
        else:
            y = sp.fft.fftshift(
                sp.fft.ifft(sp.fft.ifftshift(x, axes=0), axis=0, norm="ortho"), axes=0
            )
        return y
