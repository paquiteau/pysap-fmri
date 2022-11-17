"""Operator for classical FFT."""
import numpy as np
import scipy as sp

from .utils import DimensionMismatchError, validate_mask, validate_smaps


class FFT:
    """FFT operator for MRI data.

    Parameters
    ----------
    shape: tuple
        Dimensions of the FFT
    mask: np.ndarray
        ND array sampling mask
    n_coils: int
        Number of coils for pMRI, default 1.
    smaps: np.ndarray
        Sensitivity Maps, shared across time.
    """

    def __init__(self, shape, mask=None, n_coils=1, smaps=None, n_jobs=-1):
        self.shape = shape
        self.n_coils = n_coils
        self.smaps = validate_smaps(shape, n_coils, smaps)
        self.mask = validate_mask(shape, mask=mask)
        self.n_jobs = n_jobs

    def op(self, img):
        """Forward fourier operator.

        Parameters
        ----------
        img: np.ndarray
            Array with shape (self.n_coils, self.shape )

        Return
        ------
        np.ndarray:
            The mask fourier transform of
        """
        if self.n_coils > 1 and self.n_coils != img.shape[0]:
            raise DimensionMismatchError("expected shape: [n_coils, Nx, Ny, Nz]")

        if self.n_coils == 1:
            return self.mask * sp.fft.ifftshift(
                sp.fft.fftn(
                    sp.fft.fftshift(img),
                    norm="ortho",
                    workers=self.n_jobs,
                )
            )
        axes = tuple(np.arange(1, img.ndim))
        res = self.mask * sp.fft.ifftshift(
            sp.fft.fftn(
                sp.fft.fftshift(img, axes=axes),
                axes=axes,
                norm="ortho",
                workers=self.n_jobs,
            ),
            axes=axes,
        )
        if self.smaps is not None:
            return self.smaps * res
        return res

    def adj_op(self, data):
        """Adjoint fourier operator."""
        if self.n_coils > 1 and self.n_coils != data.shape[0]:
            raise DimensionMismatchError("expected shape: [n_coils, Nx, Ny, Nz]")

        if self.n_coils == 1:
            return sp.fft.fftshift(
                sp.fft.ifftn(
                    sp.fft.ifftshift(self.mask * data),
                    norm="ortho",
                    workers=self.n_jobs,
                )
            )
        x = data * self.mask
        axes = tuple(np.arange(1, x.ndim))
        res = sp.fft.fftshift(
            sp.fft.ifftn(
                sp.fft.ifftshift(x, axes=axes),
                axes=axes,
                norm="ortho",
                workers=self.n_jobs,
            ),
            axes=axes,
        )
        if self.smaps is not None:
            return np.sum(self.smaps.T.conjugate() * res, axis=0)
