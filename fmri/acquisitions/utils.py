import numpy as np


def add_phase_kspace(kspace_data, kspace_loc, shifts):
    """
    Add phase shifts to kspace.

    Shifts should be provided in the same units as the pixels.
    """
    if len(shifts) != kspace_loc.shape[-1]:
        raise ValueError("Dimension mismatch between shift and kspace locs!")
    phi = np.zeros((1, *kspace_loc.shape[:-1]))
    for i in range(kspace_loc.shape[-1]):
        phi += kspace_loc[..., i] * shifts[i]
    phase = np.exp(-2 * np.pi * 1j * phi)
    return kspace_data * phase
