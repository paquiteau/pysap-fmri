"""
Base class for Reconstructors.

See Also
--------
fmri.reconstructors.frame_base
fmri.reconstructors.full
"""

import warnings

from modopt.opt.linear import Identity


class BaseFMRIReconstructor:
    """Common attributes and methods for fMRI reconstruction.

    Attributes
    ----------
    fourier_op: OperatorBase
        Operator for the fourier transform of each frame
    space_linear_op: OperatorBase
        Linear operator (eg Wavelet) using for the spatial regularisation
    time_linear_op: OperatorBase
        Linear operator (eg Wavelet) using for the time regularisation
    space_prox_op: OperatorBase
        Proximal Operator for the spatial regularisation
    time_prox_op: OperatorBase
        Proximal Operator for the time regularisation
    opt_name: "pogm" or "fista"
        Optimisation algorithm to use
    grad_formulation: "synthesis" or "analysis"
        Determines in which framework the problem will be solve.
    """

    def __init__(
        self,
        fourier_op,
        space_linear_op,
        space_prox_op=None,
        time_linear_op=None,
        time_prox_op=None,
        verbose=0,
    ):
        self.fourier_op = fourier_op
        self.space_linear_op = space_linear_op or Identity
        self.time_linear_op = time_linear_op or Identity
        self.verbose = verbose

        if space_prox_op is None:
            warnings.warn(
                "The in space regulariser is not set. Setting to identity. "
                "Note that optimization is just a gradient descent in space"
            )
            self.space_prox_op = Identity()
        else:
            self.space_prox_op = space_prox_op

        if time_prox_op is None:
            warnings.warn(
                "The in-time regularizer is not set. Setting to identity. "
                "Note that frame will be reconstruct independently."
            )
            self.time_prox_op = Identity()
        else:
            self.time_prox_op = time_prox_op

    def reconstruct(self, *args, **kwargs):
        """Launch reconstruction."""
        raise NotImplementedError
