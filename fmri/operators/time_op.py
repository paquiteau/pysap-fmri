"""Operator applied voxel wise on the timeserie data."""
import numpy as np
import scipy as sp
import pysap
from pysap.base.utils import flatten, unflatten


class TimeOperator:
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
        self.n_frame_op = n_frames

    def op(self, data):
        """Apply the forward operator."""
        data_flatten = data.reshape(self.n_frames, -1)
        #data_ret = [None] * np.prod(self.shape)
        #data_ret = np.zeros((self.n_frames, np.prod(self.shape)), dtype=data.dtype)
        data_ret = np.zeros((self.n_frames_op, np.prod(self.shape)), dtype=data.dtype)

        for i in range(np.prod(self.shape)):
            print(self.op_func(data_flatten[:, i]))
            data_ret[:, i] = self.op_func(data_flatten[:, i])[0]
        return data_ret

    def adj_op(self, data):
        """Apply the adjoint operator."""
        data_ret = np.zeros((self.n_frames, np.prod(self.shape)), dtype=data.dtype)

        for i in range(np.prod(self.shape)):
            data_ret[:, i] = self.adj_op_func(data[:, i])
        data_ret = data_ret.reshape(self.shape)
        return data_ret

class WaveletTimeOperator(TimeOperator):
    def __init__(self, wavelet_name, shape, n_frames, **kwargs):
        """Initialize the operator.

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
        self.wavelet = pysap.load_transform(wavelet_name)(**kwargs)
        self.shape = shape
        self.n_frames = n_frames
        self.op_func = self._op_method
        self.adj_op_func = self._adj_op_method

        self.n_frames_op = len(self._op_method(np.zeros((self.n_frames))))
        print(self.n_frames_op)

    def _op_method(self, data):
        self.wavelet.data = data
        self.wavelet.analysis()
 #       print(self.wavelet.analysis_data.shape)
        return flatten(self.wavelet.analysis_data)

    def _adj_op_method(self, data):
        self.wavelet.analysis_data = unflatten(data, self.wavelet.coeffs_shape)
        return self.wavelet.synthesis()


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
