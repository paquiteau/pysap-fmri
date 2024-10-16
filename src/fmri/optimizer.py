"""Optimization algorithms for solving fMRI problems."""

import numpy as np

from modopt.opt.algorithms.base import SetUp
from modopt.opt.cost import costObj


class AccProxSVRG(SetUp):
    r"""Accelerated Proximal SVRG algorithm.

    Solve the problem
    .. math::
        \min_x \sum_i^n f_i(x) + g(x)


    Parameters
    ----------
    grad_ops:
        Gradient operator
    prox_op:
        Proximity operator
    cost: str, optional
        Cost function to use, by default "auto"
    step_size: float, optional
        Step size, by default 1.0
    auto_iterate: bool, optional
        Option to automatically iterate, by default True
    """

    def __init__(
        self,
        x,
        fourier_op_list,
        prox,
        cost="auto",
        step_size=1.0,
        auto_iterate=True,
        beta=None,
        batch_size=1,
        update_frequency=10,
        seed=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # Set the initial variable values
        self._check_input_data(x)
        self._x_old = self.xp.copy(x)
        self._x_new = self.xp.zeros_like(self._x_old)
        self._x_tld = self.xp.copy(x)

        self._y = self.xp.zeros_like(self._x_old)

        self._v = self.xp.zeros_like(self._x_old)
        self._v_tld = self.xp.zeros_like(self._x_old)

        self.step_size = step_size

        self.update_frequency = update_frequency
        self.batch_size = batch_size
        if beta is None:
            beta = (1 - np.sqrt(step_size)) / (1 + np.sqrt(step_size))
        self.beta = beta
        self.beta_update = None
        self._grad_ops = grad_list
        self._prox = prox

        self._rng = np.random.default_rng(seed)

        if cost == "auto":
            self._cost_func = costObj([*self._grad_ops, prox])
        else:
            self._cost_func = cost

    def _update(self):
        """Update the variables."""
        self._v_tld = self.xp.zeros_like(self._v_tld)
        # Compute the average gradient.
        for g in self._grad_ops:
            self._v_tld += g.get_grad(self._x_tld)
        self._v_tld /= len(self._grad_ops)

        self.xp.copyto(self._x_old, self._x_tld)
        self.xp.copyto(self._y, self._x_tld)
        for _ in range(self.update_frequency):
            gIk = self._rng.choice(self._grad_ops, size=self.batch_size, replace=False)
            self.xp.copyto(self._v, self._v_tld)
            self._v *= self.batch_size
            for g in gIk:
                self._v -= g.get_grad(self._x_tld)
                self._v += g.get_grad(self._y)
            self._v *= self.step_size / self.batch_size
            self.xp.copyto(self._x_new, self._y)
            self._x_new -= self._v  # Reuse the array
            self._x_new = self._prox.op(self._x_new, extra_factor=self.step_size)
            self.xp.copyto(self._v, self._x_new)
            self._v -= self._x_old  # Reuse the array
            self.xp.copyto(self._y, self._x_new)
            self._y += self.beta * self._v
            self.xp.copyto(self._x_old, self._x_new)
        self.xp.copyto(self._x_tld, self._x_new)

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self._cost_func.get_cost(self._x_tld)

    def iterate(self, max_iter=150, progbar=None):
        """Iterate the algorithm."""
        self._run_alg(max_iter, progbar)

        # retrieve metrics results
        self.retrieve_outputs()
        # rename outputs as attributes
        self.x_final = self._x_new

    def get_notify_observers_kwargs(self):
        """Notify observers.

        Return the mapping between the metrics call and the iterated
        variables.

        Returns
        -------
        dict
           The mapping between the iterated variables

        """
        return {
            "x_new_img": self._linear.adj_op(self._x_tld),
            "x_new": self._x_tld,
            "idx": self.idx,
        }

    def retrieve_outputs(self):
        """Retireve outputs.

        Declare the outputs of the algorithms as attributes: ``x_final``,
        ``y_final``, ``metrics``.

        """
        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics


class MS2GD(SetUp):
    r"""Accelerated Proximal SVRG algorithm.

    Solve the problem
    .. math::
        \min_x \sum_i^n f_i(x) + g(x)


    Parameters
    ----------
    grad_ops:
        Gradient operator
    prox_op:
        Proximity operator
    cost: str, optional
        Cost function to use, by default "auto"
    step_size: float, optional
        Step size, by default 1.0
    auto_iterate: bool, optional
        Option to automatically iterate, by default True
    """

    def __init__(
        self,
        x,
        grad_list,
        prox,
        cost="auto",
        step_size=1.0,
        auto_iterate=True,
        batch_size=1,
        update_frequency=10,
        seed=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # Set the initial variable values

        self.step_size = step_size

        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self._grad_ops = grad_list
        self._prox = prox

        self._rng = np.random.default_rng(seed)

        if cost == "auto":
            self._cost_func = costObj([*self._grad_ops, prox])
        else:
            self._cost_func = cost

        self._g = self.xp.zeros_like(x)
        self._g_sto = self.xp.zeros_like(x)
        self._y = self.xp.zeros_like(x)
        self._x = self.xp.copy(x)

    def _update(self):
        """Update the variables."""

        # Compute the average gradient.
        for g in self._grad_ops:
            self._g += g.get_grad(self._x)
        self._g /= len(self._grad_ops)
        self.xp.copyto(self._y, self._x)
        tk = self._rng.integers(1, self.update_frequency)
        for _ in range(tk):
            Ak = self._rng.choice(self._grad_ops, size=self.batch_size, replace=False)
            self.xp.copyto(self._g_sto, self._g)
            self._g_sto *= self.batch_size
            for g in Ak:
                self._g_sto -= g.get_grad(self._x)
                self._g_sto += g.get_grad(self._y)
            self._g_sto *= self.step_size / self.batch_size
            self._y = self._prox.op(self._y - self._g_sto, self.step_size)

        self.xp.copyto(self._x, self._y)
        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self._cost_func.get_cost(self._x_tld)

    def iterate(self, max_iter=150, progbar=None):
        """Iterate the algorithm."""
        self._run_alg(max_iter, progbar)

        # retrieve metrics results
        self.retrieve_outputs()
        # rename outputs as attributes
        self.x_final = self._x

    def get_notify_observers_kwargs(self):
        """Notify observers.

        Return the mapping between the metrics call and the iterated
        variables.

        Returns
        -------
        dict
           The mapping between the iterated variables

        """
        return {
            "x_new_img": self._linear.adj_op(self._x),
            "x_new": self._x,
            "idx": self.idx,
        }

    def retrieve_outputs(self):
        """Retrieve outputs.

        Declare the outputs of the algorithms as attributes: ``x_final``,
        ``y_final``, ``metrics``.

        """
        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics
