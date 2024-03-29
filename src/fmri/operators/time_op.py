"""Operator applied voxel wise on the timeserie data."""

import numpy as np
import scipy as sp
import pywt

from .wavelet import WaveletTransform


class TimeOperator:
    """Operator applied voxel wise on the timeserie data."""

    def __init__(self, op, adj_op, shape, n_frames):
        """Initialize the operator.

        Parameters
        ----------
        op : function
            The function that will be applied to the data.
        adj_op : function
            The adjoint of the function that will be applied to the data.
        """
        self.op_func = op
        self.adj_op_func = adj_op
        self.shape = shape
        self.n_frames = n_frames
        self.n_frames_sparse = n_frames

    def op(self, data):
        """Apply the forward operator."""
        data_flatten = data.reshape(self.n_frames, -1)
        data_ret = np.zeros(
            (self.n_frames_sparse, np.prod(self.shape)), dtype=data.dtype
        )

        for i in range(np.prod(self.shape)):
            data_ret[:, i] = self.op_func(data_flatten[:, i])
        data_ret = data_ret.reshape(self.n_frames_sparse, *self.shape)
        return data_ret

    def adj_op(self, data):
        """Apply the adjoint operator."""
        data_ret = np.zeros((self.n_frames, np.prod(self.shape)), dtype=data.dtype)
        data_ = np.reshape(data, (self.n_frames_sparse, np.prod(self.shape)))
        for i in range(np.prod(self.shape)):
            data_ret[:, i] = self.adj_op_func(data_[:, i])
        data_ret = data_ret.reshape((self.n_frames, *self.shape))
        return data_ret


class WaveletTimeOperator(TimeOperator):
    """Apply a 1D Wavelet on the time dimension."""

    def __init__(
        self,
        n_frames,
        wavelet_name,
        shape,
        level=4,
        n_jobs=1,
        backend="threading",
        mode="symmetric",
    ):
        """1D Wavelet Transform applied on each voxel on the time axis.

        Parameters
        ----------
        wavelet : Wavelet object
            The wavelet object that will be applied to the data.
        shape: tuple
            The shape of the data.
        n_frames: int
            The number of frames in the data.
        **kwargs: dict
            Extra arguments for the wavelet object.
        """
        # TODO Use np.prod(shape) as n_coils and benchmark.
        self._wavelet_op = WaveletTransform(
            wavelet_name,
            shape=n_frames,
            level=level,
            n_coils=1,
            decimated=True,
            backend=backend,
            mode=mode,
        )
        self.shape = shape
        self.n_frames = n_frames
        self.n_frames_sparse = pywt.wavedecn_size(
            pywt.wavedecn_shapes((n_frames,), wavelet_name, mode, level)
        )
        self.op_func = self._wavelet_op.op
        self.adj_op_func = self._wavelet_op.adj_op


class TimeFourier:
    """Temporal Fourier Transform on fMRI data."""

    def __init__(self, time_axis=0):
        super().__init__()
        self.time_axis = time_axis

    def op(self, x):
        """Forward Operator method..

        Apply the fourier transform on the time axis, voxel wise.
        Assuming the time dimension is the first one.

        """
        y = sp.fft.ifftshift(
            sp.fft.fft(
                sp.fft.fftshift(x.reshape(x.shape[0], -1), axes=self.time_axis),
                axis=self.time_axis,
                norm="ortho",
            ),
            axes=self.time_axis,
        ).reshape(x.shape)
        return y

    def adj_op(self, x):
        """Adjoint Operator method.

        Apply the Inverse fourier transform on the time axis, voxel wise
        """
        y = sp.fft.fftshift(
            sp.fft.ifft(
                sp.fft.ifftshift(x.reshape(x.shape[0], -1), axes=self.time_axis),
                axis=self.time_axis,
                norm="ortho",
            ),
            axes=self.time_axis,
        ).reshape(x.shape)
        return y
