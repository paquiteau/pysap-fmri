"""
Proximity Operator for Total Variation in 1D.
"""
import numpy as np

NUMBA_AVAILABLE = True
try:
    import numba as nb
except ImportError:
    NUMBA_AVAILABLE = False


def linearizedTautString(y, lmbd, x):
    """Linearized Taut String algorithm.

    Follows the algorithm described in [1]_, analoguous to the Condat algorithm [2]_.
    Parameters
    ----------
    y: np.ndarray
        Input data.
    lambda: float
        Regularization parameter.
    x: np.ndarray
        Return value

    Returns
    -------
    np.ndarray
        Output data.

    References
    ----------
    .. [1] https://github.com/albarji/proxTV/blob/master/src/TVL1opt.cpp
    .. [2] Condat, L. (2013). A direct algorithm for 1D total variation denoising.
    """

    i = 0
    mnHeight = mxHeight = 0
    mn = y[0] - lmbd
    mx = y[0] + lmbd
    lastBreak = -1
    mnBreak = mxBreak = 0
    N = len(y)
    while i < N:
        while i < N - 1:
            mnHeight += mn - y[i]
            if lmbd < mnHeight:
                i = mnBreak + 1
                x[lastBreak + 1 : mnBreak + 1] = mn
                lastBreak = mnBreak
                mn = y[i]
                mx = 2 * lmbd + mn
                mxHeight = lmbd
                mnHeight = -lmbd
                mnBreak = mxBreak = i
                i += 1
                continue
            mxHeight += mx - y[i]
            if -lmbd > mxHeight:
                i = mxBreak + 1
                x[lastBreak + 1 : mxBreak + 1] = mx
                lastBreak = mxBreak
                mx = y[i]
                mn = mx - 2 * lmbd
                mnHeight = lmbd
                mxHeight = -lmbd
                mnBreak = mxBreak = i
                i += 1
                continue
            if mxHeight > lmbd:
                mx += (lmbd - mxHeight) / (i - lastBreak)
                mxHeight = lmbd
                mxBreak = i
            if mnHeight <= -lmbd:
                mn += (-lmbd - mnHeight) / (i - lastBreak)
                mnHeight = -lmbd
                mnBreak = i
            i += 1
        mnHeight += mn - y[i]
        if mnHeight > 0:
            i = mnBreak + 1
            x[lastBreak + 1 : mnBreak + 1] = mn
            lastBreak = mnBreak
            mn = y[i]
            mx = 2 * lmbd + mn
            mxHeight = mnHeight = -lmbd
            mnBreak = mxBreak = i
            continue
        mxHeight += mx - y[i]
        if mxHeight < 0:
            i = mxBreak + 1
            x[lastBreak + 1 : mxBreak + 1] = mx
            lastBreak = mxBreak
            mx = y[i]
            mn = mx - 2 * lmbd
            mnHeight = mxHeight = lmbd
            mnBreak = mxBreak = i
            continue
        if mnHeight <= 0:
            mn += (-mnHeight) / (i - lastBreak)
        i += 1
    x[lastBreak + 1 :] = mn
    return x


if NUMBA_AVAILABLE:
    linearizedTautString = nb.njit(nopython=True)(linearizedTautString)
    prange = nb.prange
else:
    prange = range


def prox_tv1d(y, lmbd):
    """Proximity operator for Total Variation in 1D.

    Parameters
    ----------
    y: np.ndarray
        2D Input data.
    lambda: float
        Regularization parameter.

    Returns
    -------
    np.ndarray
        Output data.

    Notes
    -----
    For best performance, use a Fortran contiguous array.
    """
    x = np.zeros_like(y)
    for i in prange(y.shape[1]):
        linearizedTautString(y[:, i], lmbd[i], x[:, i])
    return x.reshape(y.shape)


if NUMBA_AVAILABLE:
    prox_tv1d = nb.njit(nopython=True, parallel=True)(prox_tv1d)
