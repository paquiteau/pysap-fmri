"""
Utility function for reconstructors.

See Also
--------
Modopt.opt.algorithms
"""

from modopt.opt.algorithms import POGM, ForwardBackward
from modopt.opt.cost import costObj
from modopt.base.backend import get_backend

OPTIMIZERS = {"pogm": "synthesis", "fista": "analysis", None: None}


def initialize_opt(
    opt_name,
    grad_op,
    linear_op,
    prox_op,
    x_init=None,
    synthesis_init=False,
    opt_kwargs=None,
    metric_kwargs=None,
    compute_backend="numpy",
):
    """
    Initialize an Optimizer with the suitable parameters.

    Parameters
    ----------
    grad_op: OperatorBase
        Gradient Operator for the data consistency
    x_init: ndarray, default None
        Initial value for the reconstruction. If None use a zero Array.
    synthesis_init: bool, default False
        Is the initial_value in the image space of the space_linear operator ?
    opt_kwargs: dict, default None
        Extra kwargs for the initialisation of Optimizer
    metric_kwargs: dict, default None
        Extra kwargs for the metric api of ModOpt

    Returns
    -------
    An Optimizer Instance to perform the reconstruction with.

    See Also
    --------
    Modopt.opt.algorithms

    """
    xp, _ = get_backend(compute_backend)
    if x_init is None:
        x_init = xp.squeeze(
            xp.zeros(
                (
                    (
                        grad_op.fourier_op.n_coils
                        if not grad_op.fourier_op.uses_sense
                        else 1
                    ),
                    *grad_op.fourier_op.shape,
                ),
                dtype="complex64",
            )
        )

    if not synthesis_init and hasattr(grad_op, "linear_op"):
        alpha_init = grad_op.linear_op.op(x_init)
    elif synthesis_init and not hasattr(grad_op, "linear_op"):
        x_init = linear_op.adj_op(x_init)
    elif not synthesis_init and hasattr(grad_op, "linear_op"):
        alpha_init = x_init
    opt_kwargs = opt_kwargs or dict()
    metric_kwargs = metric_kwargs or dict()

    beta = grad_op.inv_spec_rad
    if isinstance(beta, xp.ndarray):
        beta = beta.item()
    if opt_kwargs.get("cost", None) == "auto":
        opt_kwargs["cost"] = costObj([grad_op, prox_op], verbose=False)
    if opt_name == "pogm":
        opt = POGM(
            u=alpha_init,
            x=alpha_init,
            y=alpha_init,
            z=alpha_init,
            grad=grad_op,
            prox=prox_op,
            linear=linear_op,
            beta_param=beta,
            sigma_bar=opt_kwargs.pop("sigma_bar", 0.96),
            auto_iterate=opt_kwargs.pop("auto_iterate", False),
            compute_backend=compute_backend,
            **opt_kwargs,
            **metric_kwargs,
        )
    elif opt_name == "fista":
        opt = ForwardBackward(
            x=x_init,
            grad=grad_op,
            prox=prox_op,
            linear=linear_op,
            beta_param=beta,
            lambda_param=opt_kwargs.pop("lambda_param", 1.0),
            auto_iterate=opt_kwargs.pop("auto_iterate", False),
            **opt_kwargs,
            **metric_kwargs,
        )
    else:
        raise ValueError(f"Optimizer {opt_name} not implemented")
    return opt
