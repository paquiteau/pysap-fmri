import numpy as np

def add_phase_kspace(kspace_data, kspace_loc, shifts=None):
    """ add phase shifts to kspace. shifts should be provided in the size of the pixels """
    if shifts is None:
        shifts = (0,) * kspace_loc.shape[1]
    if len(shifts) != kspace_loc.shape[1]:
        raise ValueError("Dimension mismatch between shift and kspace locations! "
                         "Ensure that shifts are right")
    phi = np.zeros_like(kspace_data)
    for i in range(kspace_loc.shape[1]):
        phi += kspace_loc[:, i] * shifts[i]
    phase = np.exp(-2 * np.pi * 1j * phi)
    return kspace_data * phase