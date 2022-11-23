"""Add Activations."""
import numpy as np
import scipy as sp

from fmri.utils import DimensionMismatchError


def repeat_volume(array, n_frames, axis=0):
    """Repeat array n_frames times along the first or last axis."""
    return np.repeat(np.expand_dims(array, axis), n_frames, axis=axis)


def original_glover(t, n1=6.0, t1=0.9, a2=0.35, n2=12.0, t2=0.9):
    """Compute the value of the HRF at instant t.

    Based on the canonical definition of Glover [1_]

    Parameters
    ----------
    t: sample point (in second)
    n1, t1, a2, n2, t2 : see ref.

    References
    ----------
    [1] G. H. Glover, "Deconvolution of Impulse Response in Event-Related BOLD fMRI1,"
        NeuroImage, vol. 9, no. 4, pp. 416–429, Apr. 1999, doi: 10.1006/nimg.1998.0419.
    """
    gamma1 = t**n1 * np.exp(-t / t1)
    gamma2 = t**n2 * np.exp(-t / t2)

    c1 = gamma1.max()
    c2 = gamma2.max()

    return gamma1 / c1 - a2 * gamma2 / c2


def add_activations(volume_sequence, event_intensities, voxel_locations, TR=1):
    """Add activations to the volumee sequence.

    Parameters
    ----------
    volume_sequence: np.ndarray
        A N_frame x Volume_shape array.
    event_intensities: np.ndarray
        A N_frames x N_Voxel array.
        Each row should contains a variable intensity of activation.
        This will be convolve with a canonical HRF.
    voxel_locations: np.ndarray
        A Volume_dim array.
        Give the locations where to apply the event intensities.
    TR: float
        time between two frames.
    """
    if volume_sequence.shape[0] != event_intensities.shape[0]:
        raise DimensionMismatchError(
            "activation pattern does not fully cover time course."
        )
    if event_intensities.shape[1] != np.sum(voxel_locations):
        raise DimensionMismatchError(
            "there is not enought activation pattern for the given voxels"
        )

    volume_sequence_activated = volume_sequence.copy()

    t = np.arange(0, 32, TR)
    hrf = original_glover(t)

    activations = np.zeros_like(event_intensities)

    # convolve each timeseries with HRF.
    for i in range(len(activations)):
        activations[i, :] = sp.signal.convolve(
            event_intensities[i, :],
            hrf,
            mode="same",
            method="direct",  # don't use fft
        )

    volume_sequence_activated[:, voxel_locations] = activations

    return volume_sequence_activated
