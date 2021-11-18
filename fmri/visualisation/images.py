import time

import numpy as np
import matplotlib.pyplot as plt

from ..utils import fmri_ssos


def flat_matrix_view(fmri_img, ax=None,figsize=None,cmap="gray"):
    """ represent the fmri data as a 2d matrix, where all the spatial dimension have been flatten out."""
    if ax is None:
        fig, ax = plt.subplots((1,1),figsize=figsize)

    ax.imshow(np.reshape(fmri_ssos(abs(fmri_img)),(fmri_img.shape[0],np.prod(fmri_img.shape[1:]))),cmap=cmap)
    return ax

def dynamic_img(fmri_img, fps:float=2, normalize=True):
    """ dynamic plot of fmri data"""

    fmri_img = np.abs(fmri_img)
    if normalize:
        fmri_img *= (255.0/fmri_img.max())

    fig,ax = plt.subplots()
    obj_show = ax.imshow(np.zeros_like(fmri_img[0,:]))
    for img in fmri_img:
        obj_show.set_data(img)
        time.sleep(1./fps)
        plt.draw()
        plt.show()

def carrousel(fmri_img, frame_slicer=None, colorbar=False, padding=1, layout=None):
    """ Display frames in a single plot. """
    frame_size = np.array(fmri_img.shape[1:])
    if frame_slicer is None:
        frame_slicer=slice(0,min(len(fmri_img),10))
    index_select = np.arange(len(fmri_img))[frame_slicer]
    to_show = fmri_img[frame_slicer,...]
    N_plots = len(to_show)
    if layout is None and len(to_show) == 10:
        N_row, N_cols = 2,5
    elif layout is None:
        N_cols = np.ceil(np.sqrt(N_plots)).astype(np.int)
        N_row = np.floor(np.sqrt(N_plots)).astype(np.int)
    else:
        N_row, N_cols = layout
    vignette = np.empty((frame_size+padding)*np.array((N_row,N_cols))-padding)
    fig,ax = plt.subplots()
    vignette[:] = np.NaN
    for i in range(N_row):
        for j in range(N_cols):
            if j + i*N_cols >= len(to_show):
                break
            vignette[i*(frame_size[0]+padding):(i+1)*(frame_size[0])+ i*padding,
                     j*(frame_size[1]+padding):(j+1)*frame_size[1]+j*padding] = abs(to_show[i*N_cols+j,...])
            ax.text((j+0.01)*(frame_size[1]+padding), (i+0.05)*(frame_size[0]+padding), f'{index_select[i*N_cols+j]}',color='red')
    m = ax.imshow(vignette)
    ax.axis('off')
    if colorbar:
        fig.colorbar(m)
    return fig
