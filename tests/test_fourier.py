#!/usr/bin/env python3
import numpy as np
import numpy.testing as npt

import pytest

from fmri.operators.fourier import (
    CartesianSpaceFourier,
    RepeatOperator,
    FFT_Sense,
)


@pytest.mark.parametrize("n_coils, uses_sense", [(3, True), (3, False), (1, False)])
@pytest.mark.parametrize("n_frames", [1, 5])
@pytest.mark.parametrize("shape", [(10, 10), (10, 10, 10)])
def test_fourier_op(shape, n_frames, n_coils, uses_sense):
    if uses_sense:
        img = np.random.randn(n_frames, *shape) + 1j * np.random.randn(n_frames, *shape)
    elif n_coils > 1:
        img = np.random.randn(n_frames, n_coils, *shape) + 1j * np.random.randn(
            n_frames, n_coils, *shape
        )
    else:
        img = np.random.randn(n_frames, *shape) + 1j * np.random.randn(n_frames, *shape)
    mask = np.random.randn(n_frames, *shape) > 0

    if uses_sense:
        smaps = np.random.randn(n_coils, *shape) + 1j * np.random.randn(n_coils, *shape)
    else:
        smaps = None
    fourier_op = CartesianSpaceFourier(
        shape, n_frames=n_frames, n_coils=n_coils, mask=mask, smaps=smaps
    )
    repeat_op = RepeatOperator(
        [
            FFT_Sense(shape, n_coils=n_coils, mask=mask[i], smaps=smaps)
            for i in range(n_frames)
        ]
    )

    ksp = fourier_op.op(img)

    ksp_repeat = repeat_op.op(img)

    npt.assert_allclose(ksp, ksp_repeat)
