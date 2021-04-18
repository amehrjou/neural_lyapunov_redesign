import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration import Configuration
config = Configuration()
del Configuration
from mars.utils import dict2func, concatenate_inputs, PT_loose_thresh



__all__ = ['QuadraticFunction', 'GridWorld', 'DeterministicFunction', 'ConstantFunction', 'PTNet', 'PTPDNet', 'LinearSystem', 'Saturation', 'TrainableController']

_EPS = np.finfo(config.np_dtype).eps

class DimensionError(Exception):
    pass

class Function(object):
    """Pytorch function baseclass.
    """

    def __init__(self, name='function'):
        super(Function, self).__init__()


    @abstractmethod
    def parameters(self):
        """Return the variables within the current scope."""
        pass

    @abstractmethod
    def eval(self, x):
        pass

    def __call__(self, x):
        return self.eval(x)

    def __add__(self, other):
        """Add this function to another."""
        return AddedFunction(self, other)

    def __mul__(self, other):
        """Multiply this function with another."""
        return MultipliedFunction(self, other)

    def __neg__(self):
        """Negate the function."""
        return MultipliedFunction(self, -1)

class DeterministicFunction(Function):
    """Base class for function approximators."""

    def __init__(self, **kwargs):
        """Initialization, see `Function` for details."""
        print(type(self))
        super(DeterministicFunction, self).__init__(**kwargs)

class ConstantFunction(DeterministicFunction):
    """A function with a constant value."""

    def __init__(self, constant, name='constant_function'):
        """Initialize, see `ConstantFunction`."""
        super(ConstantFunction, self).__init__(name=name)
        self.constant = constant

    def eval(self, points):
        return self.constant


class AddedFunction(Function):
    """A class for adding two individual functions.

    Parameters
    ----------
    fun1 : instance of Function or scalar
    fun2 : instance of Function or scalar

    """

    def __init__(self, fun1, fun2):
        """Initialization, see `AddedFunction`."""
        super(AddedFunction, self).__init__()

        if not isinstance(fun1, Function):
            fun1 = ConstantFunction(fun1)
        if not isinstance(fun2, Function):
            fun2 = ConstantFunction(fun2)

        self.fun1 = fun1
        self.fun2 = fun2

    @property
    def parameters(self):
        """Return the parameters."""
        return self.fun1.parameters + self.fun2.parameters

    @concatenate_inputs(start=1)
    def eval(self, points):
        """Evaluate the function."""
        return self.fun1(points) + self.fun2(points)


class LinearSystem(DeterministicFunction):
    """A linear system.

    y = A_1 * x + A_2 * x_2 ...

    Parameters
    ----------
    *matrices : list
        Can specify an arbitrary amount of matrices for the linear system. Each
        is multiplied by the corresponding state that is passed to evaluate.


        Parameters
    ----------

    """

    def __init__(self, matrices, name='linear_system'):
        """Initialize."""
        fun = lambda x: np.atleast_2d(x).astype(config.dtype)
        self.matrix = np.hstack(map(fun, matrices))
        self.output_dim, self.input_dim = self.matrix.shape

    @concatenate_inputs(start=1)
    def eval(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray or torch.Tensor
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        output : torch.Tensor
            The next states, achieved by applying the linear system 
            on the current states.

        """
        if not isinstance(self.matrix, torch.Tensor):
            self.matrix = torch.tensor(self.matrix, dtype=config.ptdtype)
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=config.ptdtype)
        return torch.mm(points, self.matrix.t())


class QuadraticFunction(DeterministicFunction):
    """A quadratic function.

    values(x) = x.T P x

    Parameters
    ----------
    P : np.array or torch.Tensor
        2d cost matrix for lyapunov function.

    Returns
    -------
    output : torch.Tensor
        The next states, achieved by applying the linear system 
        on the current states.

    """

    def __init__(self, P, name='quadratic'):
        if not isinstance(P, torch.Tensor):
            self.P = torch.tensor(P, dtype=config.ptdtype)
        self.ndim = self.P.shape[0]

    @concatenate_inputs(start=1)
    def eval(self, points):
        """Like evaluate, but returns a pytorch tensor instead.

        Parameters
        ------------
        points : n x dim pytorch tensor
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=config.ptdtype)
        quadratic = torch.diag(torch.mm(torch.mm(points, self.P), points.t())).reshape(-1, 1)
        return quadratic

    def gradient(self, points):
        """Return the gradient of the function."""
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=config.ptdtype)
        return torch.mm(points, (self.P + self.P.t()))


class Saturation(DeterministicFunction):
    """Saturate the output of a `DeterministicFunction`.

    Parameters
    ----------
    fun : instance of `DeterministicFunction`.
    lower : float or arraylike
        Lower bound. Passed to `tf.clip_by_value`.
    upper : float or arraylike
        Upper bound. Passed to `tf.clip_by_value`.

    Returns
    ----------
    output: torch.Tensor
        A torch function that is the thresholded version of
        the input torch function fun.
    """

    def __init__(self, fun, lower, upper, name='saturation'):
        """Initialization. See `Saturation`."""
        self.fun = fun
        self.lower = lower
        self.upper = upper
        self.input_dim = self.fun.input_dim
        self.output_dim = self.fun.output_dim

    @concatenate_inputs(start=1)
    def eval(self, points):
        """Evaluation, see `DeterministicFunction.evaluate`."""
        return torch.clamp(self.fun(points), self.lower, self.upper)


class GridWorld(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    """
    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int, copy=False)

        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                    'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                            / (self.num_points - 1)).astype(config.np_dtype)
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                        self.limits[:, 1] - self.offset),
                                        axis=1)

        # Statistics about the grid
        self.discrete_points = [np.linspace(low, up, n, dtype=config.np_dtype)
                                for (low, up), n in zip(self.limits,
                                                        self.num_points)]

        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)

        self.ndim = len(self.limits)
        self._all_points = None


    @property
    def all_points(self):
        """Return all the discrete points of the discretization.

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).

        """
        if self._all_points is None:
            mesh = np.meshgrid(*self.discrete_points, indexing='ij')
            points = np.column_stack(col.ravel() for col in mesh)
            self._all_points = points.astype(config.np_dtype)
        return self._all_points

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def sample_continuous(self, num_samples):
        """Sample uniformly at random from the continuous domain.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        limits = self.limits
        rand = np.random.uniform(0, 1, size=(num_samples, self.ndim))
        return rand * np.diff(limits, axis=1).T + self.offset

    def sample_discrete(self, num_samples, replace=False):
        """Sample uniformly at random from the discrete domain.

        Parameters
        ----------
        num_samples : int
        replace : bool, optional
            Whether to sample with replacement.

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        idx = np.random.choice(self.nindex, size=num_samples, replace=replace)
        return self.index_to_state(idx)

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray

        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray

        """
        states = np.atleast_2d(states).astype(config.np_dtype)
        states = states - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
                    out=states)
        return states


    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.

        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        ijk_index = ijk_index.astype(config.np_dtype)
        return ijk_index * self.unit_maxes + self.offset
    
    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.

        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) * (1. / self.unit_maxes)
        ijk_index = np.rint(states).astype(np.int32)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.

        """
        ind = []
        for i, (discrete, num_points) in enumerate(zip(self.discrete_points,
                                                       self.num_points)):
            idx = np.digitize(states[:, i], discrete)
            idx -= 1
            np.clip(idx, 0, num_points - 2, out=idx)

            ind.append(idx)
        return np.ravel_multi_index(ind, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.

        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        ijk_index = ijk_index.astype(config.np_dtype)
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.

        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)
    

class DeterministicFunction(Function):
    """Base class for function approximators."""

    def __init__(self, **kwargs):
        """Initialization, see `Function` for details."""
        super(DeterministicFunction, self).__init__(**kwargs)


class PTNet(nn.Module):
    """A pytorch based neural network"""

    def __init__(self, wb, activations):
        super(PTNet, self).__init__()
        self.num_layers = len(wb)
        self.wb = wb
        self.activations = activations
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.activations[i](self.wb[i](x))
        return x


class TrainableController(nn.Module):
    """A linear system.

    y = A_1 * x + A_2 * x_2 ...

    Parameters
    ----------
    *matrices : list
        Can specify an arbitrary amount of matrices for the linear system. Each
        is multiplied by the corresponding state that is passed to evaluate.

    """

    def __init__(self, init_matrix, name='linear_system', args=None):
        super(TrainableController, self).__init__()
        fun = lambda x: np.atleast_2d(x).astype(config.dtype)
        self.init_matrix = init_matrix
        self.matrix = np.hstack(map(fun, self.init_matrix))
        self.matrix = torch.nn.Parameter(torch.tensor(self.matrix, dtype=config.ptdtype), requires_grad=True)
        self.low_thresh_param = torch.nn.Parameter(torch.tensor(args['low_thresh'], dtype=config.ptdtype), requires_grad=False)
        self.high_thresh_param = torch.nn.Parameter(torch.tensor(args['high_thresh'], dtype=config.ptdtype), requires_grad=False)
        self.low_slope_param = torch.nn.Parameter(torch.tensor(args['low_slope'], dtype=config.ptdtype), requires_grad=True)
        self.high_slope_param = torch.nn.Parameter(torch.tensor(args['high_slope'], dtype=config.ptdtype), requires_grad=True)
        self.args = args
        self.output_dim, self.input_dim = self.matrix.shape

    def forward(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : tf.Tensor
            A 2D array with the function values at the points.

        """
        val = torch.mm(points, self.matrix.t())
        a = torch.tensor(self.args['low_thresh'], dtype=config.ptdtype).view(1, 1)
        b = torch.tensor(self.args['high_thresh'], dtype=config.ptdtype).view(1, 1)
        ma = torch.tensor(self.args['low_slope'], dtype=config.ptdtype).view(1, 1)
        mb = torch.tensor(self.args['high_slope'], dtype=config.ptdtype).view(1, 1)
        out = PT_loose_thresh(val, self.low_thresh_param, self.high_thresh_param, self.low_slope_param, self.high_slope_param)
        return out

class PTPDNet(nn.Module):
    """A pytorch based neural network that is always architecturally positive definite."""

    def __init__(self, input_dim, layer_dims, activations, initializer, eps=1e-6):
        super(PTPDNet, self).__init__()
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        activation_dict = {'relu': torch.relu, 'tanh': torch.tanh}
        self.activations = list(map(dict2func(activation_dict), activations))
        self.eps = eps
        self.initializer = initializer
        if layer_dims[0] < input_dim:
            raise ValueError('The first layer dimension must be at least the input dimension!')

        if np.all(np.diff(layer_dims) >= 0):
            self.output_dims = layer_dims
        else:
            raise ValueError('Each layer must maintain or increase the dimension of its input!')

        self.hidden_dims = np.zeros(self.num_layers, dtype=int)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)

        # For printing results nicely
        self.layer_partitions = np.zeros(self.num_layers, dtype=int)
        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            if dim_diff > 0:
                self.layer_partitions[i] = 2
            else:
                self.layer_partitions[i] = 1
        # creating layer weights
        self.W_posdef = []
        self.W = []
        w_posdef_ind = 0
        w_ind = 0
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            W_temp = torch.nn.Parameter(torch.zeros([self.hidden_dims[i], layer_input_dim], dtype=config.ptdtype), requires_grad=True)
            setattr(self, 'W_posdef_{}'.format(w_posdef_ind), self.initializer(W_temp))
            w_posdef_ind += 1
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                W_temp = torch.nn.Parameter(torch.zeros([dim_diff, layer_input_dim], dtype=config.ptdtype), requires_grad=True)
                setattr(self, 'W_{}'.format(w_ind), self.initializer(W_temp))
                w_ind += 1
    def forward(self, points):
        w_posdef_ind = 0
        w_ind = 0
        """Build the evaluation graph."""
        net = points
        if isinstance(net, np.ndarray):
            net = torch.tensor(net)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            W_temp = getattr(self, 'W_posdef_{}'.format(w_posdef_ind))
            kernel = torch.mm(W_temp.t(), W_temp) + self.eps * torch.eye(layer_input_dim, dtype=config.ptdtype)
            dim_diff = self.output_dims[i] - layer_input_dim
            w_posdef_ind += 1
            if dim_diff > 0:
                W_temp = getattr(self, 'W_{}'.format(w_ind))
                kernel = torch.cat([kernel, W_temp], dim=0)
                w_ind += 1
            layer_output = torch.mm(net, kernel.t())
            net = self.activations[i](layer_output)
        values = torch.sum(net.pow(2), dim=1)
        values = values.reshape(-1, 1)
        return values

