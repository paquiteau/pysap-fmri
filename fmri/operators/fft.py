import numpy as np
import scipy as sp


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
    def __init__(self, shape, mask, n_coils=1, smaps=None):
        self.shape

    def op():

    def adj_op():


class StackedFFT(FFT):
    """FFT operator

    Parameters
    ----------
    shape: tuple
        Dimensions of the FFT
    slices: np.ndarray
        ND array sampling mask
    n_coils: int
        Number of coils for pMRI, default 1.
    smaps: np.ndarray
        Sensitivity Maps, shared across time.
    """

    def op():

    def adj_op():
