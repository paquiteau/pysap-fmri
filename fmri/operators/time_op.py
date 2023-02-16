"""Operator applied voxel wise on the timeserie data."""
import numpy as np
import pysap
from pysap.utils import flatten, unflatten


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

    def op(self, data):
        """Apply the forward operator."""
        data_flatten = data.reshape(self.n_frames, -1)
        data_ret = [None] * self.n_frames

        for i in range(self.n_frames):
            data_ret[i] = self.op_func(data_flatten[i])
        data_ret = np.array(data_ret)
        print(data_ret.shape)
        return data_ret

    def adj_op(self, data):
        """Apply the adjoint operator."""
        data_ret = np.zeros((self.n_frames, *self.shape))

        for i in range(self.n_frames):
            data_ret[i] = self.adj_op_func(data[i])
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

    def _op_method(self, data):
        self.wavelet.data = data
        self.wavelet.analysis()
        print(self.wavelet.analysis_data.shape)
        return flatten(self.wavelet.analysis_data)

    def _adj_op_method(self, data):
        self.wavelet.analysis_data = unflatten(data, self.wavelet.coeffs_shape)
        return self.wavelet.synthesis()


class FourierTimeOperator(TimeOperator):
    def __init__(self, shape, n_frames, **kwargs):
        """Initialize the operator.

        Parameters
        ----------
        shape: tuple
            The shape of the data.
        n_frames: int
            The number of frames in the data.
        **kwargs: dict
            Extra arguments for the wavelet object.
        """
        self.shape = shape
        self.n_frames = n_frames
        self.op_func = self._op_method
        self.adj_op_func = self._adj_op_method

    def _op_method(self, data):
        return np.fft.ifftshift(np.fft.fft(np.fft.fftshift(data), norm="ortho"))

    def _adj_op_method(self, data):
        return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data), norm="ortho"))
