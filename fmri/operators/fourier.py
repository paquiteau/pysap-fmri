import numpy as np
import scipy as sp

from mri.operators.base import OperatorBase
from mri.operators.fourier.cartesian import FFT
from mri.operators.fourier.non_cartesian import NonCartesianFFT


class SpaceFourier(OperatorBase):
    """ Spatial Fourier Transform on fMRI data """

    def __init__(self, shape, samples, n_coils, fourier_type="FFT", **kwargs) -> None:
        super().__init__()
        fourier_type=fourier_type.upper()
        if fourier_type == "FFT":
            self.fourier_op = FFT(shape, n_coils=n_coils, samples=samples, **kwargs)
        elif fourier_type == "GPUNUFFT":
            self.fourier_op = NonCartesianFFT(samples, shape, n_coils=n_coils, implementation="gpu", **kwargs)
        elif fourier_type == "NUFFT":
            self.fourier_op = NonCartesianFFT(samples, shape, n_coils=n_coils, implementation="cpu", **kwargs)
        else:
            raise NotImplementedError(f"{fourier_type} is not a valid transform")

    def op(self, x):
        return self.fourier_op.op(x[""])

    def adj_op(self, x):
        return self.fourier_op.adj_op(x[""])


class TimeFourier(OperatorBase):
    """ Temporal Fourier Transform on fMRI data """

    def init__(self, n_frames, n_coils):
        super().__init__()

    def op(self, x):
        return sp.fft.fft(x[""],norm="ortho")

    def adj_op(self, x):
        return sp.fft.ifft(x[""],norm="ortho")
