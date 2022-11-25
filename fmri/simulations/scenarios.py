"""A collection of ready to use scenario of fMRI."""
import numpy as np

from .activations import add_activations, repeat_volume
from .extras import birdcage_maps
from .noise import add_temporal_gaussian_noise, g_factor_map
from .phantoms import idx_in_ellipse, mr_ellipsoid_parameters, mr_shepp_logan_t2_star
from .utils import validate_rng


def _base_case(shape, n_coils, n_frames, gmap=True, smaps=True, dtype=np.float32):
    """Return base data for simulation."""
    phantom = mr_shepp_logan_t2_star(shape, 7).astype(dtype)
    fphantom = repeat_volume(phantom, n_frames)

    if gmap is True:
        gmap = g_factor_map(shape).astype(dtype)
    elif isinstance(gmap, np.ndarray):
        pass
    else:
        gmap = None

    if smaps is True:
        smaps = birdcage_maps((n_coils, *shape)).astype(dtype)
    elif isinstance(smaps, np.ndarray):
        pass
    else:
        smaps = None
    return {
        "phantom": fphantom,
        "gmap": gmap,
        "smaps": smaps,
    }


def _add_phantom_noisy(scenario, snr, rng, type="gaussian"):

    if snr > 0:
        energy = np.max(scenario["phantom"])
        if type == "gaussian":
            noisy_phantom = add_temporal_gaussian_noise(
                scenario["phantom"],
                sigma=energy / snr,
                g_factor_map=scenario["gmap"],
                rng=rng,
            )
        else:
            raise NotImplementedError()
    else:
        noisy_phantom = scenario["phantom"]

    scenario["phantom_noisy"] = noisy_phantom
    return scenario


def noisy_constant(
    shape,
    n_coils,
    n_frames,
    snr,
    gmap=True,
    smaps=True,
    rng=42,
    dtype=np.float32,
):
    """Return a basic scenario where the serie is constant in time + gaussian noise."""
    scenario = _base_case(shape, n_coils, n_frames, gmap, smaps, dtype=dtype)

    return _add_phantom_noisy(scenario, snr, rng)


def block_design(
    shape,
    n_coils,
    n_frames,
    TR=1,
    block_on=5,
    block_off=5,
    block_p=0.99,
    snr=0,
    gmap=True,
    smaps=True,
    rng=42,
    dtype=np.float32,
):
    """Return simulation phantom where block design have been added."""
    scenario = _base_case(shape, n_coils, n_frames, gmap, smaps, dtype=dtype)

    rng = validate_rng(rng)
    # use a "tumor" ellipse region of the brain,
    # they have same T2 and chi parameter as gray-matter
    E = mr_ellipsoid_parameters()[10]
    roi_idx = idx_in_ellipse(E, shape)

    block_size = block_on + block_off
    block_proba = np.zeros(n_frames)

    for i in range((n_frames // block_size)):
        block_proba[i * block_size + block_off : (i + 1) * block_size] = block_p
        block_proba[i * block_size : i * block_size + block_off] = 1 - block_p

    voxel_event = np.zeros(shape=(n_frames, np.sum(roi_idx)))
    for idx in range(voxel_event.shape[1]):
        voxel_event[:, idx] = rng.random(n_frames) < block_proba

    activated_phantom = add_activations(scenario["phantom"], voxel_event, roi_idx)
    scenario["phantom_static"] = scenario["phantom"][0]
    scenario["roi"] = roi_idx
    scenario["phantom"] = activated_phantom
    scenario["activations"] = voxel_event

    return _add_phantom_noisy(scenario, snr, rng)


# TODO: add scenario with block design activation
# TODO: add scenario with motion.
#
# TODO: add possibility to chain/compose scenarios (Factory design pattern?)
