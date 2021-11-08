import time

import numpy as np
import matplotlib.pyplot as plt

from ..utils import fmri_ssos


def flat_matrix_view(fmri_img, ax=None):
    """ represent the fmri data as a 2d matrix, where all the spatial dimension have been flatten out."""
    if ax is None:
        fig, ax = plt.subplots(1,1)

    ax.imshow(np.reshape(fmri_ssos(abs(fmri_img)),(fmri_img.shape[0],np.prod(fmri_img.shape[2:]))),cmap="gray")
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
