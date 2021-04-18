import numpy as np
import torch
from .utils import batchify, get_storage, set_storage, unique_rows
from .configuration import Configuration
config = Configuration()
del Configuration


__all__ = ['Lyapunov']

class Lyapunov(object):
    """A class for general Lyapunov functions.

    Parameters
    ----------
    discretization : ndarray
        A discrete grid on which to evaluate the Lyapunov function.
    lyapunov_function : callable or instance of `DeterministicFunction`
        The lyapunov function. Can be called with states and returns the
        corresponding values of the Lyapunov function.
    dynamics : a callable or an instance of `Function`
        The dynamics model. Can be either a deterministic function or something
        uncertain that includes error bounds.
    lipschitz_dynamics : ndarray or float
        The Lipschitz constant of the dynamics. Either globally, or locally
        for each point in the discretization (within a radius given by the
        discretization constant. This is the closed-loop Lipschitz constant
        including the policy!
    lipschitz_lyapunov : ndarray or float
        The Lipschitz constant of the lyapunov function. Either globally, or
        locally for each point in the discretization (within a radius given by
        the discretization constant.
    tau : float
        The discretization constant.
    policy : ndarray, optional
        The control policy used at each state (Same number of rows as the
        discretization).
    initial_set : ndarray, optional
        A boolean array of states that are known to be safe a priori.
    adaptive : bool, optional
        A boolean determining whether an adaptive discretization is used for
        stability verification.
    decrease_thresh: None or a real value. If None, the threshold is computed by self.threshold function.
    If it is a real value, the value is considered as the threshold.

    """

    def __init__(self, discretization, lyapunov_function, dynamics,
                 lipschitz_dynamics, lipschitz_lyapunov,
                 tau, policy, initial_set=None, adaptive=False, decrease_thresh=None):
        """Initialization, see `Lyapunov` for details."""
        super(Lyapunov, self).__init__()

        self.discretization = discretization
        self.policy = policy
        # Keep track of the safe sets
        self.safe_set = np.zeros(np.prod(discretization.num_points),
                                 dtype=bool)
        self.initial_safe_set = initial_set
        if initial_set is not None:
            self.safe_set[initial_set] = True
        # Discretization constant
        self.tau = tau
        self.decrease_thresh = decrease_thresh
        # Make sure dynamics are of standard framework
        self.dynamics = dynamics
        # Make sure Lyapunov fits into standard framework
        self.lyapunov_function = lyapunov_function
        # Storage for graph
        self._storage = dict()
        # Lyapunov values
        self.values = None
        self.c_max = torch.tensor(0, dtype=config.ptdtype)
        # self.feed_dict[self.c_max] = 0.
        self._lipschitz_dynamics = lipschitz_dynamics
        self._lipschitz_lyapunov = lipschitz_lyapunov
        self.update_values()
        self.adaptive = adaptive


    def lipschitz_dynamics(self, states):
        """Return the Lipschitz constant for given states and actions.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_dynamics is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_dynamics, '__call__'): # check if _lipschitz_dynamics is a function
            return self._lipschitz_dynamics(states)
        else:
            return self._lipschitz_dynamics


    def lipschitz_lyapunov(self, states):
        """Return the local Lipschitz constant at a given state.

        Parameters
        ----------
        states : ndarray or Tensor

        Returns
        -------
        lipschitz : float, ndarray or Tensor
            If lipschitz_lyapunov is a callable then returns local Lipschitz
            constants. Otherwise returns the Lipschitz constant as a scalar.
        """
        if hasattr(self._lipschitz_lyapunov, '__call__'):
            return self._lipschitz_lyapunov(states)
        else:
            return self._lipschitz_lyapunov


    def threshold(self, states, tau=None):
        """Return the safety threshold for the Lyapunov condition.
        meaning that v(x(t+1)) - v(x(t)) must be less than this threshold
        to ensure negativity of the dv

        Parameters
        ----------
        states : ndarray or torch.Tensor

        tau : np.float or torch.Tensor, optional
            discretization constant to consider.

        Returns
        -------
        lipschitz : np.float, ndarray or torch.Tensor
            Either the scalar threshold or local thresholds, depending on
            whether lipschitz_lyapunov and lipschitz_dynamics are local or not.
        """
        if tau is None:
            tau = self.tau
        lv = self.lipschitz_lyapunov(states)
        if hasattr(self._lipschitz_lyapunov, '__call__') and lv.shape[1] > 1:
            lv = torch.norm(lv, p=1, axis=1)
        lf = self.lipschitz_dynamics(states)
        return - lv * (1. + lf) * tau


    def is_safe(self, state):
        """Return a boolean array that indicates whether the state is safe using the current safe set.

        Parameters
        ----------
        state : ndarray

        Returns
        -------
        safe : boolean numpy array
            Is true if the corresponding state is inside the safe set.

        """
        return self.safe_set[self.discretization.state_to_index(state)]


    def update_values(self):
        """Update the discretized values when the Lyapunov function changes.
        self.values will be a 1D torch Tensor, (N, ) tensor of scalars where N is the number of
        points in the discretization.
        It also updates the self._storage
        """
    
        storage = get_storage(self._storage)
        if storage is None:
            pt_points = self.discretization.all_points
            pt_values = self.lyapunov_function(pt_points)
            storage = [('points', pt_points), ('values', pt_values)]
            set_storage(self._storage, storage)
        else:
            pt_points, pt_values = storage.values()
        pt_points = self.discretization.all_points
        self.values = torch.squeeze(self.lyapunov_function(pt_points))

    def check_decrease_condition(self, pt_states, policy, threshold):
        """ Check if the decrease condition is satisfied for the points on the dicretization for a given policy

        Parameters
        ----------
        pt_states: (N x d) pytorch tensors as the states of the system
        policy: A pytorch function that determines how actions are produced by the current states. If policy is None, the system
                is autonomous.
        threshold: (N x 1) negative values as the upper bound of the decrease condition of the Lyapunov function for each state

        Returns
        ----------
        decrease_condition: (N,) pytorch tensor representing if the decrease condition holds for each state
        """
        if policy is not None:
            actions = policy(pt_states)
            next_states = self.dynamics(pt_states, actions)
        else:
            next_states = self.dynamics(pt_states)
        decrease =  self.lyapunov_function(next_states) - self.lyapunov_function(pt_states)
        pt_negative = torch.squeeze(torch.lt(decrease, threshold))
        return pt_negative


    def update_safe_set(self, can_shrink=True):
        """Compute and update the safe set and c_max to determine the levelset.

        Parameters
        ----------
        can_shrink : bool, optional
            A boolean determining whether previously safe states other than the
            initial safe set must be verified again (i.e., can the safe set
            shrink in volume?)

        """

        if can_shrink:
            # Reset the safe set
            safe_set = np.zeros_like(self.safe_set, dtype=bool)
            if self.initial_safe_set is not None:
                safe_set[self.initial_safe_set] = True
        else:
            # Assume safe set cannot shrink
            safe_set = self.safe_set
        self.update_values()
        value_order = np.argsort(np.squeeze(self.values.detach().numpy())) # ordered indices based on the values of the Lyapunov function
        safe_set = safe_set[value_order]

        # Verify safety in batches
        batch_size = config.batch_size
        batch_generator = batchify((value_order, safe_set),
                                   batch_size)
        index_to_state = self.discretization.index_to_state

        #######################################################################

        for i, (indices, safe_batch) in batch_generator:
            states = index_to_state(indices)

            # Update the safety with the safe_batch result
            thresh = torch.tensor(self.decrease_thresh, dtype=config.ptdtype) if self.decrease_thresh is not None else self.threshold(torch.tensor(states, dtype=config.ptdtype), self.tau)
            negative = self.check_decrease_condition(torch.tensor(states, dtype=config.ptdtype), self.policy, threshold=thresh).detach().numpy().astype('int')
            negative = (negative == 1)
            safe_batch |= negative

            # Boolean array: argmin returns first element that is False
            # If all are safe then it returns 0
            bound = np.argmin(safe_batch)

            # Check if there are unsafe elements in the batch
            if bound > 0 or not safe_batch[0]:
                # Make sure all following points are labeled as unsafe (because the batch is ordered with the values of the Lyapunov function)
                safe_batch[bound:] = False
                break
        # The largest index of a safe value
        max_index = i + bound # i is the starting index of each batch and bound is the index inside a batch

        # Set placeholder for c_max to the corresponding value
        self.c_max = self.values[value_order[max_index]]

        # Restore the order of the safe set
        safe_nodes = value_order[safe_set]
        self.safe_set[:] = False
        self.safe_set[safe_nodes] = True

        # Ensure the initial safe set is kept
        if self.initial_safe_set is not None:
            self.safe_set[self.initial_safe_set] = True

