import time

import matplotlib.pyplot as plt
import numpy as np

from ..utils import fmri_ssos
from .utils import normalize


def flat_matrix_view(fmri_img, ax=None, figsize=None, cmap="gray"):
    """Represent the fmri data as a 2d matirx."""
    if ax is None:
        fig, ax = plt.subplots((1, 1), figsize=figsize)

    ax.imshow(
        np.reshape(
            fmri_ssos(abs(fmri_img)), (fmri_img.shape[0], np.prod(fmri_img.shape[1:]))
        ),
        cmap=cmap,
    )
    return ax


def dynamic_img(fmri_img, fps: float = 2, normalize=True):
    """Dynamic plot of fmri data."""

    fmri_img = np.abs(fmri_img)
    if normalize:
        fmri_img *= 255.0 / fmri_img.max()

    fig, ax = plt.subplots()
    obj_show = ax.imshow(np.zeros_like(fmri_img[0, :]))
    for img in fmri_img:
        obj_show.set_data(img)
        time.sleep(1.0 / fps)
        plt.draw()
        plt.show()


def carrousel(
    fmri_img,
    ax=None,
    frame_slicer=None,
    colorbar=False,
    pad=1,
    layout=None,
    normalized=False,
        mode="portrait",
):
    """
    Display frames in a single plot.

    Returns
    -------
    fig: figure object.
    """
    f_size = np.array(fmri_img.shape[1:])
    if frame_slicer is None:
        frame_slicer = slice(0, min(len(fmri_img), 10))
    index_select = np.arange(len(fmri_img))[frame_slicer]
    to_show = fmri_img[frame_slicer, ...]
    n_plots = len(to_show)
    if layout is None and len(to_show) == 10:
        n_rows, n_cols = 2, 5
    elif layout is None:
        n_cols = np.ceil(np.sqrt(n_plots)).astype(np.int)
        n_rows = np.floor(np.sqrt(n_plots)).astype(np.int)
        if n_cols * n_rows < n_plots:
            n_cols += 1
    else:
        n_rows, n_cols = layout

    if mode == "portrait" and n_rows < n_cols:
        n_rows, n_cols = n_cols, n_rows

    vignette = np.empty((f_size + pad) * np.array((n_rows, n_cols)) - pad)
    if ax is None:
        fig, ax = plt.subplots()
    vignette[:] = np.NaN
    for i in range(n_rows):
        for j in range(n_cols):
            if j + i * n_cols >= len(to_show):
                break
            if normalized:
                show = normalize(abs(to_show[i * n_cols + j]))
            else:
                show = abs(to_show[i * n_cols + j])
            vignette[
                i * (f_size[0] + pad):(i + 1) * (f_size[0]) + i * pad,
                j * (f_size[1] + pad):(j + 1) * f_size[1] + j * pad] = show
            ax.text(
                (j + 0.01) * (f_size[1] + pad),
                (i + 0.05) * (f_size[0] + pad),
                f'{index_select[i*n_cols+j]}', color='red')
    m = ax.imshow(vignette)
    ax.axis('off')
    if colorbar:
        fig.colorbar(m)
    return ax


def make_movie(filename, array, share_norm=True, fps=2, **kwargs):
    """Make a movie from a n_frames x N x N array."""
    import imageio.v2 as imageio

    array_val = abs(array)

    min_val = np.min(array_val, axis=None if share_norm else (1, 2))
    max_val = np.max(array_val, axis=None if share_norm else (1, 2))
    array_val = 255 * (array_val - min_val) / (max_val - min_val)
    array_val = np.uint8(array_val)

    imageio.mimsave(filename, array_val, fps=fps, **kwargs)
