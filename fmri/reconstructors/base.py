"""
Base class for Reconstructors.

See Also:
---------
fmri.reconstructors.frame_base
fmri.reconstructors.full
"""
import warnings

from modopt.opt.linear import Identity

from .utils import initialize_opt


OPTIMIZERS = {'pogm': 'synthesis',
              'fista':  'analysis',
              None: None}


class BaseFMRIReconstructor(object):
    """This class hold common attributes and methods for fMRI reconstruction.

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
    smaps: ndarray, default None
        sensibility Maps for the reconstruction.
        If None, Smaps can be provided with the fourier operator.
        If no Smaps is available, performs a calibrationless reconstruction.
    """

    def __init__(self, fourier_op, space_linear_op, space_regularisation=None,
                 time_linear_op=None, time_regularisation=None, Smaps=None, optimizer='pogm', verbose=0,):
        """
        Create base reconstructor.

        Parameters:
        -----------
        fourier_op: Fourier operator
        space_linear_operator: Linear Operator
            Linear operator for the in space regularisation.
        space_regularisation: Proximity Operator, default Identity
            Proximity Operator for the in space regularisation.
        time_linear_op: Linear Operator
            Linear operator for the  in-time regularisation.
        time_regularisation: Proximity Operator, default Identity
            Proximity Operator the in-time regularisation
        Smaps: ndarray, default None
            Sensibility map for the reconstruction (if not set in the fourier operator)
        optimizer: "pogm" or "fista"
            Optimizer algorithm to use.
        verbose: int default=0
            verbosity level.
        """
        self.fourier_op = fourier_op
        self.space_linear_op = space_linear_op or Identity
        self.time_linear_op = time_linear_op or Identity
        self.opt_name = optimizer
        self.grad_formulation = OPTIMIZERS[optimizer]

        if Smaps is not None:
            if fourier_op.n_coils > 1:
                if Smaps.shape[0] != fourier_op.n_coils and Smaps.shape[1:] != fourier_op.shape:
                    raise("Smaps shape is different from the image dimensions.")
        self.smaps = Smaps
        self.verbose = verbose

        if space_regularisation is None:
            warnings.warn("The in space regulariser is not set. Setting to identity. "
                          "Note that optimization is just a gradient descent in space")
            self.space_prox_op = Identity()
        else:
            self.space_prox_op = space_regularisation

        if time_regularisation is None:
            warnings.warn("The in-time regularizer is not set. Setting to identity. "
                          "Note that frame will be reconstruct independently.")
            self.time_prox_op = Identity()
        else:
            self.time_prox_op = time_regularisation

    def reconstruct(self, kspace_data, *args, **kwargs):
        """Lauchn reconstruction."""
        raise NotImplementedError

    def initialize_opt(self, grad_op, x_init=None, synthesis_init=False,
                     opt_kwargs=None, metric_kwargs=None):
        """Initialize an  optimizer for a given gradient """
        return initialize_opt(opt_name=self.opt_name, grad_op=grad_op,
                         linear_op=self.space_linear_op,
                         prox_op=self.space_prox_op)
