"""Fourier Operator for fMRI data."""
import joblib
import numpy as np
import scipy as sp
from mri.operators.base import OperatorBase
from mri.operators.fourier.cartesian import FFT
from mri.operators.fourier.non_cartesian import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation_gpu


class SpaceFourier(OperatorBase):
    """Spatial Fourier Transform on fMRI data."""

    def __init__(self, shape, n_coils, n_frames,
                 samples, fourier_type="FFT", class_grad=False, **kwargs) -> None:
        super().__init__()
        self.img_shape = shape
        self.n_samples = len(samples)
        self.n_coils = n_coils
        self.n_frames = n_frames
        self.shape = np.array([self.n_frames, self.n_coils, *self.img_shape])
        self.class_grad = class_grad
        self.fourier_type = fourier_type
        if fourier_type == "FFT":
            self.spatial_op = FFT(shape, n_coils=n_coils,
                                  samples=samples, **kwargs)
        elif fourier_type == "gpuNUFFT":
            self.spatial_op = NonCartesianFFT(
                samples,
                shape,
                n_coils=n_coils,
                implementation="gpuNUFFT",
                **kwargs)

        elif fourier_type == "NUFFT":
            self.spatial_op = NonCartesianFFT(
                samples,
                shape,
                n_coils=n_coils,
                implementation="cpu",
                **kwargs)
        else:
            raise NotImplementedError(
                f"{fourier_type} is not a valid transform")

    def op(self, x):
        """Forward Operator method."""
        # x is a n_frame x n_coils x shape array
        y = np.zeros((self.n_frames, self.n_coils,
                     self.n_samples), dtype=x.dtype)

        for i_frame in range(x.shape[0]):
            y[i_frame, ...] = self.spatial_op.op(x[i_frame, ...])
        return y

    def adj_op(self, y):
        """Adjoint Operator method."""
        if getattr(self.spatial_op.impl, 'uses_sense', False):
            x = np.zeros((self.n_frames, *self.img_shape), dtype=y.dtype)
        else:
            x = np.zeros((self.n_frames, self.n_coils,
                         *self.img_shape), dtype=y.dtype)
        for i_frame in range(self.n_frames):
            x[i_frame] = self.spatial_op.adj_op(y[i_frame, ...])
        return x

    def data_consistency(self, x, obs_data):
        """Compute data Consistency Operation.

        Compute adj_op(op(x) - obs_data)
        """
        if getattr(self.spatial_op.impl, 'uses_sense', False):
            gradient = np.zeros(
                (self.n_frames, *self.img_shape), dtype=x.dtype)
        else:
            gradient = np.zeros((self.n_frames, self.n_coils,
                                 *self.img_shape), dtype=x.dtype)

        if not self.class_grad and self.fourier_type == "gpuNUFFT": # not stable yet
            for i_frame in range(self.n_frames):
                gradient[i_frame] = self.spatial_op.impl.data_consistency(
                    x[i_frame], obs_data[i_frame])
        else:
            for i_frame in range(self.n_frames):
                gradient[i_frame] = self.spatial_op.adj_op(
                    self.spatial_op.op(x[i_frame]) - obs_data[i_frame])

        return gradient

class SpaceFourierMulti(OperatorBase):
    """Operator for Fourier Transform non constant Kspace traj samples."""

    def __init__(self, shape, n_coils, samples, n_jobs,
                 fourier_type="FFT",  **kwargs):
        super().__init__()
        self.img_shape = shape
        self.n_samples = len(samples[0])
        self.n_frames = len(samples)
        self.n_coils = n_coils
        self.n_jobs = n_jobs
        self.shape = np.array([self.n_frames, self.n_coils, *self.img_shape])

        if fourier_type == "gpuNUFFT":
            # instanciate all fourier operator with different trajectories.
            self.fourier_ops = [None] * self.n_frames
            for i in range(self.n_frames):
                density_array = estimate_density_compensation_gpu(
                    samples[i], shape)
                self.fourier_ops[i] = NonCartesianFFT(samples[i],
                                                      shape,
                                                      n_coils=n_coils,
                                                      implementation="gpuNUFFT",
                                                      density_comp=density_array,
                                                      )
        else:
            raise NotImplementedError(
                f"{fourier_type} is not a valid transform")

    def op(self, x):
        """Forward Operator method."""
        # x is a n_frame x n_coils x shape array
        y = np.zeros((self.n_frames, self.n_coils,
                     self.n_samples), dtype=x.dtype)

        for i_frame in range(x.shape[0]):
            y[i_frame, ...] = self.fourier_ops[i_frame].op(x[i_frame, ...])
        return y

    def adj_op(self, y):
        """Adjoint Operator method."""
        if getattr(self.spatial_op.impl, 'uses_sense', False):
            x = np.zeros((self.n_frames, *self.img_shape), dtype=y.dtype)
        else:
            x = np.zeros((self.n_frames, self.n_coils,
                         *self.img_shape), dtype=y.dtype)
        for i_frame in range(self.n_frames):
            x[i_frame] = self.fourier_ops[i_frame].adj_op(y[i_frame, ...])
        return np.asarray(x)


class TimeFourier(OperatorBase):
    """Temporal Fourier Transform on fMRI data."""

    def __init__(self, roi=None):
        super().__init__()
        self.roi = roi

    def op(self, x):
        """Forward Operator method..

        Apply the fourier transform on the time axis, voxel wise
        """
        y = np.zeros_like(x)
        if self.roi is not None:
            y[:, self.roi] = sp.fft.ifftshift(
                sp.fft.fft(
                    sp.fft.fftshift(x[:, self.roi], axes=0),
                    axis=0, norm="ortho"),
                axes=0)
        else:
            y = sp.fft.ifftshift(
                sp.fft.fft(
                    sp.fft.fftshift(x, axes=0),
                    axis=0, norm="ortho"),
                axes=0)
        return y

    def adj_op(self, x):
        """Adjoint Operator method.

        Apply the Inverse fourier transform on the time axis, voxel wise
        """
        y = np.zeros_like(x)
        if self.roi is not None:
            y[:, self.roi] = sp.fft.fftshift(
                sp.fft.ifft(
                    sp.fft.ifftshift(x[:, self.roi], axes=0),
                    axis=0, norm="ortho"),
                axes=0)
        else:
            y = sp.fft.fftshift(
                sp.fft.ifft(
                    sp.fft.ifftshift(x, axes=0),
                    axis=0, norm="ortho"),
                axes=0)
        return y
