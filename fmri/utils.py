import numpy as np

def ssos(img,axis=0):
    """ return the square root sum of square """
    return np.sqrt(np.sum(np.square(img),axis))

def fmri_ssos(img):
    return ssos(img,axis=1)
