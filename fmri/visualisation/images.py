import time

import matplotlib.pyplot as plt
import numpy as np

from ..utils import fmri_ssos


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


def fit_grid(n_tiles):
    """Give the number of row and columns to optimally fit n_tiles."""
    n_rows = int(np.sqrt(n_tiles))
    n_cols = n_rows
    while n_rows * n_cols < n_tiles:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1
    return n_rows, n_cols


def mosaic(array, axis=-1, samples=-1, n_rows=-1, n_cols=-1, img_w=3, fig=None):
    """Plot a 3D array as a mosaic grid of 2D images."""
    if array.ndim != 3:
        raise ValueError("Only 3D array are supported.")
    if axis < 0:
        axis = 3 + axis

    slicer = [slice(None), slice(None), slice(None)]
    axis_label = ["x", "y", "z"]

    if samples == -1:
        samples_loc = np.arange(array.shape[axis])
        step = 1
    else:
        step = array.shape[axis] // (samples + 1)

        samples_loc = np.arange(1, samples + 1) * step

    n_samples = len(samples_loc)
    array_list = [array[(*slicer[:axis], s, *slicer[axis + 1 :])] for s in samples_loc]

    if n_rows == -1 and n_cols != -1:
        while n_rows * n_cols < n_samples:
            n_rows += 1
    elif n_rows != -1 and n_cols == -1:
        while n_rows * n_cols < n_samples:
            n_cols += 1
    elif n_rows == -1 and n_cols == -1:
        n_rows, n_cols = fit_grid(n_samples)
    print(array_list[0].shape)
    aspect_ratio = array_list[0].shape[0] / array_list[0].shape[1]

    fig = plt.figure(num=fig, figsize=(n_cols * img_w, n_rows * img_w * aspect_ratio))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        hspace=0.01,
        wspace=0.01,
    )
    axs_2d = gs.subplots(squeeze=False)

    axs = axs_2d.flatten()
    for i, img in enumerate(array_list):
        ax = axs[i]
        ax.axis("off")
        if np.any(np.iscomplex(img)):
            ax.imshow(abs(img))
        else:
            ax.imshow(img)
        ax.text(
            0.05,
            0.95,
            f"{axis_label[axis]}={step*i}",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
    return fig


def make_movie(filename, array, share_norm=True, fps=2, **kwargs):
    """Make a movie from a n_frames x N x N array."""
    import imageio.v2 as imageio

    array_val = abs(array)

    min_val = np.min(array_val, axis=None if share_norm else (1, 2))
    max_val = np.max(array_val, axis=None if share_norm else (1, 2))
    array_val = 255 * (array_val - min_val) / (max_val - min_val)
    array_val = np.uint8(array_val)

    imageio.mimsave(filename, array_val, fps=fps, **kwargs)
