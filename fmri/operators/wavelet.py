"""
Wavelet operator, build around PyWavelet.

"""

from modopt.opt.linear import LinearParent
import pywt
from joblib import Parallel, delayed, cpu_count
import numpy as np


class WaveletTransform(LinearParent):
    """
    2D and 3D wavelet transform class.

    This is a light wrapper around PyWavelet, with multicoil support.

    Parameters
    ----------
    wavelet_name: str
        the wavelet name to be used during the decomposition.
    shape: tuple[int,...]
        Shape of the input data. The shape should be a tuple of length 2 or 3.
        It should not contains coils or batch dimension.
    nb_scales: int, default 4
        the number of scales in the decomposition.
    n_coils: int, default 1
        the number of coils for multichannel reconstruction
    n_jobs: int, default 1
        the number of cores to use for multichannel.
    backend: str, default "threading"
        the backend to use for parallel multichannel linear operation.
    verbose: int, default 0
        the verbosity level.

    Attributes
    ----------
    nb_scale: int
        number of scale decomposed in wavelet space.
    n_jobs: int
        number of jobs for parallel computation
    n_coils: int
        number of coils use f
    backend: str
        Backend use for parallel computation
    verbose: int
        Verbosity level
    """

    def __init__(
        self,
        wavelet_name,
        shape,
        level=4,
        n_coils=1,
        n_jobs=1,
        decimated=True,
        backend="threading",
        mode="symmetric",
    ):
        if wavelet_name not in pywt.wavelist(kind="all"):
            raise ValueError(
                "Invalid wavelet name. Check ``pywt.waveletlist(kind='all')``"
            )

        self.wavelet_name = wavelet_name
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.coeffs_shape = None
        self.n_jobs = n_jobs

        self.wave_conf = {"wavelet": self.wavelet_name, "mode": mode, "level": level}
        if not decimated:
            raise NotImplementedError(
                "Undecimated Wavelet Transform is not implemented yet."
            )
        if len(shape) > 1:
            self.dwt = pywt.wavedecn
            self.idwt = pywt.waverecn
            self._pywt_fun = "wavedecn"
        else:
            self.dwt = pywt.wavedec
            self.idwt = pywt.waverec
            self._pywt_fun = "wavedec"

        self.n_coils = n_coils
        if self.n_coils == 1 and self.n_jobs != 1:
            print("Making n_jobs = 1 for WaveletN as n_coils = 1")
            self.n_jobs = 1
        self.backend = backend
        n_proc = self.n_jobs
        if n_proc < 0:
            n_proc = cpu_count() + self.n_jobs + 1

    def op(self, data):
        """Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        if self.n_coils > 1:
            coeffs, self.coeffs_slices, self.coeffs_shape = zip(
                *Parallel(
                    n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
                )(delayed(self._op)(data[i]) for i in np.arange(self.n_coils))
            )
            coeffs = np.asarray(coeffs)
        else:
            coeffs, self.coeffs_slices, self.coeffs_shape = self._op(data)
        return coeffs

    def _op(self, data):
        """single coil wavelet transform."""

        return pywt.ravel_coeffs(self.idwt(data, **self.wave_conf))

    def adj_op(self, coeffs):
        """Define the wavelet adjoint operator.

        This method returns the reconstructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.n_coils > 1:
            images = Parallel(
                n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose
            )(
                delayed(self._adj_op)(coeffs[i], self.coeffs_shape[i])
                for i in np.arange(self.n_coils)
            )
            images = np.asarray(images)
        else:
            images = self._adj_op(coeffs, self.coeffs_shape)
        return images

    def _adj_op(self, coeffs):
        """Single coil inverse wavelet transform."""
        return self.idwt(
            pywt.unravel_coeffs(
                coeffs, self.coeff_slices, self.coeff_shapes, self._pywt_fun
            ),
            **self.wave_conf
        )
