import numpy as np
import scipy as sp

from mri.operators.base import OperatorBase
from mri.operators.fourier.cartesian import FFT
from mri.operators.fourier.non_cartesian import NonCartesianFFT



class SpaceFourier(OperatorBase):
    """ Spatial Fourier Transform on fMRI data """

    def __init__(self, shape, n_coils, n_frames, samples, fourier_type="FFT", **kwargs) -> None:
        super().__init__()
        self.img_shape = shape
        self.n_samples = len(samples)
        self.n_coils = n_coils
        self.n_frames = n_frames
        self.shape = np.array([self.n_frames, self.n_coils, *self.img_shape])
        if fourier_type == "FFT":
            self.fourier_op = FFT(shape, n_coils=n_coils, samples=samples, **kwargs)
        elif fourier_type == "gpuNUFFT":
            self.fourier_op = NonCartesianFFT(samples, shape, n_coils=n_coils, implementation="gpuNUFFT", **kwargs)
        elif fourier_type == "NUFFT":
            self.fourier_op = NonCartesianFFT(samples, shape, n_coils=n_coils, implementation="cpu", **kwargs)
        else:
            raise NotImplementedError(f"{fourier_type} is not a valid transform")

    def op(self, x):
        # x is a n_frame x n_coils x shape array
        y = np.zeros((self.n_frames, self.n_coils, self.n_samples),dtype = x.dtype)

        for i_frame in range(x.shape[0]):
            y[i_frame, ...] = self.fourier_op.op(x[i_frame,...])
        return y


    def adj_op(self, y):
        x = np.zeros((self.n_frames, self.n_coils, *self.img_shape),dtype = y.dtype)
        for i_frame in range(self.n_frames):
            x[i_frame,...] = self.fourier_op.adj_op(y[i_frame,...])
        return x



class TimeFourier(OperatorBase):
    """ Temporal Fourier Transform on fMRI data """

    def init__(self, n_frames, n_coils):
        super().__init__()

    def op(self, x):
        return sp.fft.fft(x,axes=0, norm="ortho")

    def adj_op(self, x):
        return sp.fft.ifft(x,axes=0, norm="ortho")
