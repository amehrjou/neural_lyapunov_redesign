from __future__ import division, print_function

import sys
import os
import importlib
import numpy as np
import scipy
from scipy import signal
from matplotlib.colors import ListedColormap
import sys
sys.path.insert(0, '../')
from mars import config, DeterministicFunction, GridWorld
from mars.utils import dict2func
import torch
from mars import config, DeterministicFunction, GridWorld, PTPDNet
from mars.utils import concatenate_inputs
import pickle 

__all__ = ['build_system', 'InvertedPendulum', 'CartPole', 'VanDerPol', 'LyapunovNetwork', 'compute_roa', 'generate_trajectories', 'save_dict', 'load_dict']


def import_from_directory(library, path):
    """Import a library from a directory outside the path.

    Parameters
    ----------
    library: string
        The name of the library.
    path: string
        The path of the folder containing the library.

    """
    try:
        return importlib.import_module(library)
    except ImportError:
        module_path = os.path.abspath(path)
        version = sys.version_info

        if version.major == 2:
            f, filename, desc = imp.find_module(library, [module_path])
            return imp.load_module(library, f, filename, desc)
        else:
            sys.path.append(module_path)
            return importlib.import_module(library)

class LyapunovNetwork(DeterministicFunction):
    def __init__(self, input_dim, layer_dims, activations, eps=1e-6,
                initializer=torch.nn.init.xavier_uniform,
                name='lyapunov_network'):
        """
        initializer: a function that takes weights as intput and initialize them
        """
        super(LyapunovNetwork, self).__init__(name=name)
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.layer_dims = layer_dims
        self.eps = eps
        self.initializer = initializer
        self.net = PTPDNet(self.input_dim, self.layer_dims, self.activations, self.initializer, self.eps)

    def eval(self, x):
        return self.net(x)

    def __call__(self, x):
        return self.eval(x)


class Andrea(DeterministicFunction):
    """
    The system proposed by Andrea for the ODE-ROA project
    Parameters. When the system is autonomous, we use a dummy action
    for consistency.
    --------------
    delta: A constant that changes the ROA of the system.
    """
    def __init__(self, delta, dt= 1/80, normalization=None):
        super(Andrea, self).__init__(name='Andrea')
        self.delta = delta
        self.dt = dt
        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]


    def normalize(self, state):
        """Normalize states."""
        if self.normalization is None:
            return state
        Tx_inv = np.diag(self.inv_norm)
        state = torch.mm(state, torch.tensor(Tx_inv))
        return state


    def denormalize(self, state):
        """De-normalize states."""
        if self.normalization is None:
            return state
        Tx = np.diag(self.normalization)
        state = torch.mm(state, torch.tensor(Tx))
        return state


    @concatenate_inputs(start=1)
    def eval(self, state):
        """Evaluate the dynamics.
    
        Parameters
        ----------
        state: ndarray or Tensor
            normalized states of the system.

        Returns
        -------
        normalized next state: Tensor
            The normalized next state after applying the dynamics for one timestep.
            
        """
        # Denormalize
        state = self.denormalize(state)
        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state)
            state = state + dt * state_derivative
        return self.normalize(state)


    def ode(self, state):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """

        x1, x2 = torch.split(state, [1, 1], dim=1)
        x1dot = -x2 - 3/2 * x1**2 - 1/2 * x1**3 + self.delta
        x2dot = 3*x1 - x2 - x2**2
        state_derivative = torch.cat((x1dot, x2dot), dim=1)

        # Normalize
        return state_derivative

    def linearize(self):
        raise NotImplementedError
        

class DuffingOscillator(DeterministicFunction):
    """
    Parameters
    --------------
    mass(float): mass
    k_linear(float): linear stiffness 
    k_nonlinear(float): nonlinear stiffness
    damping(float) damping coefficient
    """

    def __init__(self, mass, k_linear, k_nonlinear, damping, dt= 1/80, normalization=None):
        super(DuffingOscillator, self).__init__()
        self.mass = mass
        self.k_linear = k_linear
        self.k_nonlinear = k_nonlinear
        self.damping = damping
        self.dt = dt
        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = torch.mm(state, torch.tensor(Tx_inv))

        if action is not None:
            action = torch.mm(action, torch.tensor(Tu_inv))

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)
        state = torch.mm(state, torch.tensor(Tx))
        if action is not None:
            action = torch.mm(action, torch.tensor(Tu))

        return state, action

    @concatenate_inputs(start=1)
    def eval(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        state, action = torch.split(state_action, [2, 1], dim=1)
        state, action = self.denormalize(state, action)

        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]


    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """

        position, velocity = torch.split(state, [1, 1], dim=1)
        x_ddot = 1 / self.mass * (- self.damping * velocity - self.k_linear * position - self.k_nonlinear *  position.pow(3) + action)
        state_derivative = torch.cat((velocity, x_ddot), dim=1)

        # Normalize
        return state_derivative

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        A : ndarray
            The state matrix.
        B : ndarray
            The action matrix.

        """
        A = np.array([[0, 1],
                        [-1 /self.mass * self.damping + -1 /self.mass * 3 * self.k_nonlinear, -self.damping / self.mass]])

        B = np.array([[0],
                    [-1/self.mass]])

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)

        A = np.linalg.multi_dot((Tx_inv, A, Tx))
        B = np.linalg.multi_dot((Tx_inv, B, Tu))

        sys = signal.StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.dt)
        return sysd.A, sysd.B

class InvertedPendulum(DeterministicFunction):
    """Inverted Pendulum.

    Parameters
    ----------
    mass : float
    length : float
    friction : float, optional
    dt : float, optional
        The sampling time.
    normalization : tuple, optional
        A tuple (Tx, Tu) of arrays used to normalize the state and actions. It
        is so that diag(Tx) *x_norm = x and diag(Tu) * u_norm = u.

    """

    def __init__(self, mass, length, friction=0, dt=1 / 80,
                 normalization=None):
        """Initialization; see `InvertedPendulum`."""
        super(InvertedPendulum, self).__init__()
        self.mass = mass
        self.length = length
        self.gravity = 9.81
        self.friction = friction
        self.dt = dt

        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    @property
    def inertia(self):
        """Return inertia of the pendulum."""
        return self.mass * self.length ** 2

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = torch.mm(state, torch.tensor(Tx_inv))

        if action is not None:
            action = torch.mm(action, torch.tensor(Tu_inv))

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)
        state = torch.mm(state, torch.tensor(Tx))
        if action is not None:
            action = torch.mm(action, torch.tensor(Tu))

        return state, action

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        A = np.array([[0, 1],
                      [gravity / length, -friction / inertia]],
                     dtype=config.np_dtype)

        B = np.array([[0],
                      [1 / inertia]],
                     dtype=config.np_dtype)

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)

            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        sys = signal.StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.dt)
        return sysd.A, sysd.B

    @concatenate_inputs(start=1)
    def eval(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        state, action = torch.split(state_action, [2, 1], dim=1)
        state, action = self.denormalize(state, action)

        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]

    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """
        # Physical dynamics
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        angle, angular_velocity = torch.split(state, [1, 1], dim=1)

        x_ddot = gravity / length * torch.sin(angle) + action / inertia

        if friction > 0:
            x_ddot -= friction / inertia * angular_velocity

        state_derivative = torch.cat((angular_velocity, x_ddot), dim=1)

        # Normalize
        return state_derivative


class CartPole(DeterministicFunction):
    """
    Parameters
    ----------
    pendulum_mass : float
    cart_mass : float
    length : float
    dt : float, optional
        The sampling period used for discretization.
    normalization : tuple, optional
        A tuple (Tx, Tu) of 1-D arrays or lists used to normalize the state and
        action, such that x = diag(Tx) * x_norm and u = diag(Tu) * u_norm.

    """
    
    def __init__(self, pendulum_mass, cart_mass, length, rot_friction=0.0, 
                dt=0.01, normalization=None):
        """Initialization; see `CartPole`.""" 
        super(CartPole, self).__init__()
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.length = length
        self.rot_friction = rot_friction
        self.dt = dt
        self.gravity = 9.81
        self.state_dim = 4
        self.action_dim = 1
        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = torch.mm(state, torch.tensor(Tx_inv))

        if action is not None:
            action = torch.mm(action, torch.tensor(Tu_inv))
        
        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)
        state = torch.mm(state, torch.tensor(Tx))
        if action is not None:
            action = torch.mm(action, torch.tensor(Tu))

        return state, action

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        Ad : ndarray
            The discrete-time state matrix.
        Bd : ndarray
            The discrete-time action matrix.

        """
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.rot_friction
        g = self.gravity

        A = np.array([[0, 0,                     1, 0                            ],
                    [0, 0,                     0, 1                            ],
                    [0, g * m / M,             0, -b / (M * L)                 ],
                    [0, g * (m + M) / (L * M), 0, -b * (m + M) / (m * M * L**2)]],
                    dtype=config.np_dtype)

        B = np.array([0, 0, 1 / M, 1 / (M * L)]).reshape((-1, self.action_dim))

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        Ad, Bd, _, _, _ = signal.cont2discrete((A, B, 0, 0), self.dt,
                                                method='zoh')
        return Ad, Bd

        
    @concatenate_inputs(start=1)
    def eval(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        state, action = torch.split(state_action, [4, 1], dim=1)
        state, action = self.denormalize(state, action)

        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]


    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.
        action: ndarray or Tensor
            Actions.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        # Physical dynamics
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.rot_friction
        g = self.gravity

        x, theta, v, omega = torch.split(state, [1, 1, 1, 1], dim=1)

        x_dot = v
        theta_dot = omega

        det = L*(M + m * torch.mul(torch.sin(theta), torch.sin(theta)))
        v_dot = (action - m * L * torch.mul(omega, omega) * torch.sin(theta) - b * omega * torch.cos(theta) + 0.5 * m * g * L * torch.sin(2 * theta)) * L / det
        omega_dot = (action * torch.cos(theta) - 0.5 * m * L * torch.mul(omega, omega) * torch.sin(2 * theta) - b * (m + M) * omega / (m * L)
                    + (m + M) * g * torch.sin(theta)) / det

        state_derivative = torch.cat((x_dot, theta_dot, v_dot, omega_dot), dim=1)

        return state_derivative
        
            
class VanDerPol(DeterministicFunction):
    """Van der Pol oscillator in reverse-time."""

    def __init__(self, damping=1, dt=0.01, normalization=None):
        """Initialization; see `VanDerPol`."""
        super(VanDerPol, self).__init__(name='VanDerPol')
        self.damping = damping
        self.dt = dt
        self.state_dim = 2
        self.action_dim = 0
        self.normalization = normalization
        if normalization is not None:
            self.normalization = np.array(normalization, dtype=config.np_dtype)
            self.inv_norm = self.normalization ** -1

    def normalize(self, state):
        """Normalize states."""
        if self.normalization is None:
            return state
        Tx_inv = np.diag(self.inv_norm)
        state = torch.mm(state, torch.tensor(Tx_inv))
        return state

    def denormalize(self, state):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state
        Tx = np.diag(self.normalization)
        state = torch.mm(state, torch.tensor(Tx))
        return state

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        Ad : ndarray
            The discrete-time state matrix.

        """
        A = np.array([[0, -1], [1, -1]], dtype=config.np_dtype)
        B = np.zeros([2, 1])
        if self.normalization is not None:
            Tx = np.diag(self.normalization)
            Tx_inv = np.diag(self.inv_norm)
            A = np.linalg.multi_dot((Tx_inv, A, Tx))
        B = np.zeros([2, 1])

        Ad, _, _, _, _ = signal.cont2discrete((A, B, 0, 0), self.dt, method='zoh')

        return Ad

    @concatenate_inputs(start=1)
    def eval(self, state):
        """Evaluate the dynamics.
        
        Parameters
        ----------
        state: ndarray or Tensor
            normalized states of the system.

        Returns
        -------
        normalized next state: Tensor
            The normalized next state after applying the dynamics for one timestep.
            
        """
        # Denormalize
        state = self.denormalize(state)
        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state)
            state = state + dt * state_derivative
        return self.normalize(state)

    def ode(self, state):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        # Physical dynamics
        damping = self.damping
        x, y = torch.split(state, [1, 1], dim=1)
        x_dot = - y
        y_dot = x + damping * (x ** 2 - 1) * y
        state_derivative = torch.cat((x_dot, y_dot), dim=1)
        return state_derivative

def compute_roa(grid, closed_loop_dynamics, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization.
    
    Parameters
    ----------
    grid: ndarray or a GridWorld instance
        The set of initial states to check for stability.
    closed_loop_dynamics: PT function
        Takes the current state and produces the next state.
    horizon: int
        How far the simulation of each state should go to check for stability (The longer, more accurate but more constly).
    tol: float,
        How large the gap between the final state and the origin can be (The larger, more states are considered as stable).
    equilibrium: ndarray
        The equilibrium wrt which the final state of the simulated trajetcories are compared.
    no_traj: Boolean
        If False, the simulated trajectories are kept and returned.

    Returns
    -------
    roa: ndarray
        Binary array where the points beloning to roa are labeled True and the rest are labeled False.
    trajectories: ndarray
        If no_traj is false, the simulated trajectories for all initial points of the provided grid are returned as a ndarray.
    
    """


    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex
        ndim = grid.ndim

    # Forward-simulate all trajectories from initial points in the discretization
    if no_traj:
        end_states = all_points
        for t in range(1, horizon):
            end_states = closed_loop_dynamics(end_states)
    else:
        trajectories = np.empty((nindex, ndim, horizon))
        trajectories[:, :, 0] = all_points
        with torch.no_grad():
            for t in range(1, horizon):
                trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
        end_states = trajectories[:, :, -1]

    if equilibrium is None:
        equilibrium = np.zeros((1, ndim))

    # Compute an approximate ROA as all states that end up "close" to 0
    dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories


def compute_roa_zero_v(grid, closed_loop_dynamics, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
    """Computes those states in the discritization grid that evolved to some stationary point (no necessarily the origin).
    
    Parameters
    ----------
    grid: ndarray or a GridWorld instance
        The set of initial states to check for stability.
    closed_loop_dynamics: PT function
        Takes the current state and produces the next state.
    horizon: int
        How far the simulation of each state should go to check for stability (The longer, more accurate but more constly).
    tol: float,
        How large the gap between the final state and the origin can be (The larger, more states are considered as stable).
    equilibrium: ndarray
        The equilibrium wrt which the final state of the simulated trajetcories are compared.
    no_traj: Boolean
        If False, the simulated trajectories are kept and returned.

    Returns
    -------
    roa: ndarray
        Binary array where the points beloning to roa are labeled True and the rest are labeled False.
    trajectories: ndarray
        If no_traj is false, the simulated trajectories for all initial points of the provided grid are returned as a ndarray.
    
    """
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex
        ndim = grid.ndim

    # Forward-simulate all trajectories from initial points in the discretization
    if no_traj:
        end_states_one_before = all_points
        for t in range(1, horizon-1):
            end_states_one_before = closed_loop_dynamics(end_states_one_before)
        end_states = closed_loop_dynamics(end_states_one_before)
    else:
        trajectories = np.empty((nindex, ndim, horizon))
        trajectories[:, :, 0] = all_points
        with torch.no_grad():
            for t in range(1, horizon):
                trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
        end_states_one_before = trajectories[:, :, -2]
        end_states = trajectories[:, :, -1]

    # Compute an approximate ROA as all states that end up "close" to 0
    dists = np.linalg.norm(end_states - end_states_one_before, ord=2, axis=1, keepdims=True).ravel()
    roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories

def monomials(x, deg):
    """Compute monomial features of `x' up to degree `deg'."""
    x = np.atleast_2d(np.copy(x))
    # 1-D features (x, y)
    Z = x
    if deg >= 2:
        # 2-D features (x^2, x * y, y^2)
        temp = np.empty([len(x), 3])
        temp[:, 0] = x[:, 0] ** 2
        temp[:, 1] = x[:, 0] * x[:, 1]
        temp[:, 2] = x[:, 1] ** 2
        Z = np.hstack((Z, temp))
    if deg >= 3:
        # 3-D features (x^3, x^2 * y, x * y^2, y^3)
        temp = np.empty([len(x), 4])
        temp[:, 0] = x[:, 0] ** 3
        temp[:, 1] = (x[:, 0] ** 2) * x[:, 1]
        temp[:, 2] = x[:, 0] * (x[:, 1] ** 2)
        temp[:, 3] = x[:, 1] ** 3
        Z = np.hstack((Z, temp))
    if deg >= 4:
        # 4-D features (x^4, x^3 * y, x^2 * y^2, x * y^3, y^4)
        temp = np.empty([len(x), 5])
        temp[:, 0] = x[:, 0] ** 4
        temp[:, 1] = (x[:, 0] ** 3) * x[:, 1]
        temp[:, 2] = (x[:, 0] ** 2) * (x[:, 1] ** 2)
        temp[:, 3] = x[:, 0] * (x[:, 1] ** 3)
        temp[:, 4] = x[:, 1] ** 4
        Z = np.hstack((Z, temp))
    return Z

def derivative_monomials(x, deg):
    """Compute derivatives of monomial features of `x' up to degree `deg'."""
    x = np.atleast_2d(np.copy(x))
    dim = x.shape[1]
    # 1-D features (x, y)
    Z = np.zeros([len(x), 2, dim])
    Z[:, 0, 0] = 1
    Z[:, 1, 1] = 1
    if deg >= 2:
        # 2-D features (x^2, x * y, y^2)
        temp = np.zeros([len(x), 3, dim])
        temp[:, 0, 0] = 2 * x[:, 0]
        temp[:, 1, 0] = x[:, 1]
        temp[:, 1, 1] = x[:, 0]
        temp[:, 2, 1] = 2 * x[:, 1]
        Z = np.concatenate((Z, temp), axis=1)
    if deg >= 3:
        # 3-D features (x^3, x^2 * y, x * y^2, y^3)
        temp = np.zeros([len(x), 4, dim])
        temp[:, 0, 0] = 3 * x[:, 0] ** 2
        temp[:, 1, 0] = 2 * x[:, 0] * x[:, 1]
        temp[:, 1, 1] = x[:, 0] ** 2
        temp[:, 2, 0] = x[:, 1] ** 2
        temp[:, 2, 1] = 2 * x[:, 0] * x[:, 1]
        temp[:, 3, 1] = 3 * x[:, 1] ** 2
        Z = np.concatenate((Z, temp), axis=1)
    return Z


def binary_cmap(color='red', alpha=1.):
    """Construct a binary colormap."""
    if color == 'red':
        color_code = (1., 0., 0., alpha)
    elif color == 'green':
        color_code = (0., 1., 0., alpha)
    elif color == 'blue':
        color_code = (0., 0., 1., alpha)
    else:
        color_code = color
    transparent_code = (1., 1., 1., 0.)
    return ListedColormap([transparent_code, color_code])



def balanced_class_weights(y_true, scale_by_total=True):
    """Compute class weights from class label counts."""
    y = y_true.astype(np.bool)
    nP = y.sum()
    nN = y.size - y.sum()
    class_counts = np.array([nN, nP])

    weights = np.ones_like(y, dtype=float)
    weights[ y] /= nP
    weights[~y] /= nN
    if scale_by_total:
        weights *= y.size

    return weights, class_counts


def generate_trajectories(states_init, closed_loop_dynamics, dt, horizon):
    if isinstance(states_init, np.ndarray):
        states_init = torch.tensor(np.copy(states_init), dtype=config.ptdtype)
    nindex = states_init.shape[0]
    ndim = states_init.shape[1]
    
    trajectories = torch.zeros((nindex, ndim, horizon+1), dtype=config.ptdtype)
    grad_field = torch.zeros((nindex, ndim, horizon), dtype=config.ptdtype)
    trajectories[:, :, 0] = states_init
    
    with torch.no_grad():
        for t in range(1, horizon+1):
            trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
            grad_field[:, :, t-1] = (trajectories[:, :, t] - trajectories[:, :, t-1]) / dt
    return trajectories[:,:, 0:-1], grad_field



def build_system(system_properties, dt):
    """
    Takes an instance of system_property class and return a 
    system class based on the type of the system.
    """
    s = system_properties
    if s.type == "pendulum":
        system = InvertedPendulum(s.m , s.L, s.b, dt, [s.state_norm, s.action_norm])
    return system

def save_dict(dict_obj, fullname):
    with open(fullname, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(fullname):
    with open(fullname, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    return loaded_obj

