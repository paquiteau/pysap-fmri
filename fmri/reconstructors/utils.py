import numpy as np
from modopt.opt.algorithms import POGM, ForwardBackward

def initialize_opt(opt_name, grad_op, linear_op, prox_op, x_init=None, synthesis_init=False, opt_kwargs=None, metric_kwargs=None):
    if x_init is None:
        x_init = np.squeeze(np.zeros(grad_op.fourier_op.n_coils,*grad_op.fourier_op.shape,dtype="complex64"))

    if synthesis_init == False and  hasattr(grad_op,'linear_op'):
        alpha_init = grad_op.linear_op.op(x_init)
    elif not hasattr(grad_op,'linear_op'):
        x_init = linear_op.adj_op(x_init)
    elif synthesis_init == True:
        alpha_init = x_init
    opt_kwargs = opt_kwargs or dict()
    metric_kwargs = metric_kwargs or dict()

    beta = grad_op.inv_spec_rad
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
            sigma_bar=opt_kwargs.pop('sigma_bar',0.96),
            auto_iterate=opt_kwargs.pop("auto_iterate",False),
            **opt_kwargs,
            **metric_kwargs
        )
    elif opt_name == "fista":
        opt = ForwardBackward(
            x=x_init,
            grad=grad_op,
            prox=prox_op,
            linear=linear_op,
            beta_param=beta,
            lambda_param=opt_kwargs.pop("lambda_param",1.0),
            auto_iterate=opt_kwargs.pop("auto_iterate",False),
            **opt_kwargs,
            **metric_kwargs,
        )
    else:
        raise ValueError(f"Optimizer {opt_name} not implemented")
    return opt
