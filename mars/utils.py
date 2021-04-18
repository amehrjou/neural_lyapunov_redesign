from __future__ import absolute_import, division, print_function

import itertools
import inspect
from functools import wraps, partial

import numpy as np
import scipy.interpolate
import scipy.linalg
from future.builtins import zip, range
from future.backports import OrderedDict
import torch
from matplotlib.colors import ListedColormap

__all__ = ['PT_loose_thresh', 'PT_indicator', 'combinations', 'linearly_spaced_combinations', 'lqr', 'dlqr',
           'ellipse_bounds', 'concatenate_inputs', 'make_tf_fun',
           'with_scope', 'use_parent_scope', 'add_weight_constraint',
           'batchify', 'get_storage', 'set_storage', 'unique_rows',
           'gradient_clipping', 'binary_cmap', 'make_dataset_from_trajectories', ]


def unique_rows(array):
    """Return the unique rows of the array.

    Parameters
    ----------
    array : ndarray
        A 2D numpy array.

    Returns
    -------
    unique_array : ndarray
        A 2D numpy array that contains all the unique rows of array.
    """
    array = np.ascontiguousarray(array)
    # Combine all the rows into a single element of the flexible void datatype
    dtype = np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    combined_array = array.view(dtype=dtype)
    # Get all the unique rows of the combined array
    _, idx = np.unique(combined_array, return_index=True)
    return array[idx]
    

def get_storage(dictionary, index=None):
    """Get a unique storage point within a class method.

    Parameters
    ----------
    dictionary : dict
        A dictionary used for storage.
    index : hashable
        An index under which to load the element. Needs to be hashable.
        This is useful for functions which might be accessed with multiple
        different arguments.

    Returns
    -------
    storage : OrderedDict
        The storage object. Is None if no storage exists. Otherwise it
        returns the OrderedDict that was previously put in the storage.
    """
    # Use function name as storage name
    frame = inspect.currentframe()
    storage_name = inspect.getframeinfo(frame.f_back).function
    storage = dictionary.get(storage_name)
    if index is None:
        return storage
    elif storage is not None:
        # Return directly the indexed object
        try:
            return storage[index]
        except KeyError:
            pass


def set_storage(dictionary, name_value, index=None):
    """Set the storage point within a class method.

    Parameters
    ----------
    dictionary : dict
    name_value : tuple
        A list of tuples, where each tuple contains a string with the name
        of the storage object and the corresponding value that is to be put
        in storage. These are stored as OrderedDicts.
    index : hashable
        An index under which to store the element. Needs to be hashable.
        This is useful for functions which might be accessed with multiple
        different arguements.
    """
    # Use function name as storage name
    frame = inspect.currentframe()
    storage_name = inspect.getframeinfo(frame.f_back).function 
    storage = OrderedDict(name_value)
    if index is None:
        dictionary[storage_name] = storage
    else:
        # Make sure the storage is initialized
        if storage_name not in dictionary:
            dictionary[storage_name] = {}
        # Set the indexed storage
        dictionary[storage_name][index] = storage


def batchify(arrays, batch_size):
    """Yield the arrays in batches and in order.

    The last batch might be smaller than batch_size.

    Parameters
    ----------
    arrays : list of ndarray
        The arrays that we want to convert to batches.
    batch_size : int
        The size of each individual batch.
    """
    if not isinstance(arrays, (list, tuple)):
        arrays = (arrays,)
    # Iterate over array in batches
    for i, i_next in zip(itertools.count(start=0, step=batch_size),
                         itertools.count(start=batch_size, step=batch_size)):
        batches = [array[i:i_next] for array in arrays]
        # Break if there are no points left
        if batches[0].size:
            yield i, batches
        else:
            break



def combinations(arrays):
    """Return a single array with combinations of parameters.

    Parameters
    ----------
    arrays : list of np.array

    Returns
    -------
    array : np.array
        An array that contains all combinations of the input arrays
    """
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds : sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples : integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations : 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    bounds = np.atleast_2d(bounds)
    num_vars = len(bounds)
    num_samples = np.broadcast_to(num_samples, num_vars)

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]
    # Convert to 2-D array
    return combinations(inputs)


def lqr(a, b, q, r):
    """Compute the continuous time LQR-controller.

    The optimal control input is `u = -k.dot(x)`.

    Parameters
    ----------
    a : np.array
    b : np.array
    q : np.array
    r : np.array

    Returns
    -------
    k : np.array
        Controller matrix
    p : np.array
        Cost to go matrix
    """
    a, b, q, r = map(np.atleast_2d, (a, b, q, r))
    p = scipy.linalg.solve_continuous_are(a, b, q, r)

    # LQR gain
    k = np.linalg.solve(r, b.T.dot(p))

    return k, p


def dlqr(a, b, q, r):
    """Compute the discrete-time LQR controller.

    The optimal control input is `u = -k.dot(x)`.

    Parameters
    ----------
    a : np.array
    b : np.array
    q : np.array
    r : np.array

    Returns
    -------
    k : np.array
        Controller matrix
    p : np.array
        Cost to go matrix
    """
    a, b, q, r = map(np.atleast_2d, (a, b, q, r))
    p = scipy.linalg.solve_discrete_are(a, b, q, r)

    # LQR gain
    # k = (b.T * p * b + r)^-1 * (b.T * p * a)
    bp = b.T.dot(p)
    tmp1 = bp.dot(b)
    tmp1 += r
    tmp2 = bp.dot(a)
    k = np.linalg.solve(tmp1, tmp2)
    return k, p


def ellipse_bounds(P, level, n=100):
    """Compute the bounds of a 2D ellipse.

    The levelset of the ellipsoid is given by
    level = x' P x. Given the coordinates of the first
    dimension, this function computes the corresponding
    lower and upper values of the second dimension and
    removes any values of x0 that are outside of the ellipse.

    Parameters
    ----------
    P : np.array
        The matrix of the ellipsoid
    level : float
        The value of the levelset
    n : int
        Number of data points

    Returns
    -------
    x : np.array
        1D array of x positions of the ellipse
    yu : np.array
        The upper bound of the ellipse
    yl : np.array
        The lower bound of the ellipse

    Notes
    -----
    This can be used as
    ```plt.fill_between(*ellipse_bounds(P, level))```
    """
    # Round up to multiple of 2
    n += n % 2

    # Principal axes of ellipsoid
    eigval, eigvec = np.linalg.eig(P)
    eigvec *= np.sqrt(level / eigval)

    # set zero angle at maximum x
    angle = np.linspace(0, 2 * np.pi, n)[:, None]
    angle += np.arctan(eigvec[0, 1] / eigvec[0, 0])

    # Compute positions
    pos = np.cos(angle) * eigvec[:, 0] + np.sin(angle) * eigvec[:, 1]
    n /= 2

    # Return x-position (symmetric) and upper/lower bounds
    return pos[:n, 0], pos[:n, 1], pos[:n - 1:-1, 1]


def dict2func(d):
    return lambda x: d[x]


def concatenate_inputs(start=0):
    """Concatenate the numpy array inputs to the functions.

    Parameters
    ----------
    start : int, optional
        The attribute number at which to start concatenating.
    """
    def wrap(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            """Concatenate the input arguments."""
            nargs = len(args) - start
            torch_objects = torch.Tensor
            if any(isinstance(arg, torch_objects) for arg in args[start:]):
                # reduce number of function calls in graph
                if nargs == 1:
                    return function(*args, **kwargs)
                # concatenate extra arguments
                args = args[:start] + (torch.cat(args[start:], dim=1),)
                return function(*args, **kwargs)
            else:
                to_concatenate = list(map(np.atleast_2d, args[start:]))
                if nargs == 1:
                    concatenated = tuple(to_concatenate)
                else:
                    concatenated = (np.hstack(to_concatenate),)
                
                args = args[:start] + concatenated
                return function(*args, **kwargs)
        return wrapped_function

    return wrap


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


def get_batch_grad(f, inputs):
    """"Computes the gradient of a scalar-valued function
    with respect to its input
    
    Parameters
    ----------
    f : A torch scalar-valued function.
    inputs: torch Tensor (nsamples, input dimension)

    Returns
    ----------
    grads : (nsamples, input dimension), The gradient of f
            with respect to each of its inputs.
    """
    nsamples = inputs.shape[0]
    grads = torch.zeros_like(inputs)
    input_values = inputs.detach().numpy()
    for i in range(nsamples):
        x = torch.tensor(input_values[[i]], requires_grad=True)
        y = f(x)
        y.backward()
        grads[i] = x.grad
    return grads


def get_number_of_rows_and_columns(m):
    """ Takes m as the total number things and find r and c as
    they are closest to the square root of m
    
    Parameters
    ----------
    m : Total number of subplots

    Returns
    ----------
    r, c : the number of rows and columns

    """

    r = int(np.sqrt(m))
    c = m // r if np.mod(m, r) == 0 else m // r + 1
    return r, c


def save_lyapunov_nn(lyapunov_nn, full_path):
    """
    Get an instance of Lyapunov and store the weights of its lyapunov_function which is a neural network
    """
    model = lyapunov_nn.lyapunov_function.net
    torch.save(model.state_dict(), full_path)


def load_lyapunov_nn(lyapunov_nn, full_path):
    """
    Get an initialized instance of Lyapunov and replace its lyapunov_function which is 
    a neural network with the loaded one from the specified address.
    """
    loaded_state_dict = torch.load(full_path)
    lyapunov_nn.lyapunov_function.net.load_state_dict(loaded_state_dict)
    return lyapunov_nn

def print_no_newline(string): 
    """Print with replacement without going to the new line
    Useful for showing the progress of training or search
    """
    sys.stdout.write(string)
    sys.stdout.flush()

def compute_nrows_ncolumns(nplots):
    """
    Takes the total number of plots and calculate the number
    of rows and columns.
    """
    n_rows = int(np.sqrt(nplots)) + (np.sqrt(nplots) != int(np.sqrt(nplots))) * 1
    n_columns = int(nplots / n_rows) + (nplots / n_rows != int(nplots / n_rows)) * 1
    return n_rows, n_columns

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import numpy as np
import torch

def choose_initial_points(all_points, n_points):
    """
    Parameters
    ----------
    all_states: all available points in the grid to choose from
    n_states: number of points to choose
    
    
    Returns
    ----------
    chosen_states: chosen initial points
    """
    all_inds = np.arange(len(all_points))
    chosen_inds = np.random.choice(all_inds, n_points)
    chosen_states = np.atleast_2d(all_points[chosen_inds])
    return chosen_states


def generate_trajectories_from_inits(closed_loop_dynamics, initial_states, dt, n_reps, length, noise_std):
    """ Takes a list of initial states and outputs corresponding
    trajectories of the specified length.
    
    Parameters
    ----------
    init_states : iterable. n_envs = len(init_states) is the number of initial states (environments)
    n_reps: number of trajectories starting for each initial state (environment). For example,
    the more noisy trajectories from a particular intial state can be used to reduce noise
    by averaging.

    Returns
    -------
    dataset : is a dictionary whose "envs" entry gives a list of environment and the
            "trajectories" gives a list of size n_envs whose every element is a list of size n_reps whose every element is 
            of size L x  n_states of repititins of trajectories for a particular starting point
    """
    n_envs = len(initial_states)
    ddim = len(initial_states[0])
    trajectories = np.zeros((n_envs, ddim, length, n_reps))
    for r in range(n_reps):
        trajectories[:, :, :, r] = trajectory_simulator(closed_loop_dynamics, initial_states, 
                                                        dt, length, noise_std)
    return trajectories


def trajectory_simulator(close_loop_dynamics, initial_states, dt, length, noise_std):
    """Takes the closed loop dynamics and generate a trajectory with desired length
    and observation Gaussian noise with specified standard deviation.
    
    Parameters
    ----------
    close_loop_dynamics : a Torch function that represents the ode rhs.
    initial_states: initial states of the intented trajectories.
    dt: sampling time interval
    length: Length of the intended trajectory.
    noise_std: Standard deviation of the observation noise


    Returns
    -------
    trajectories : len(initial_states) x length ndarray of trajectories.
    """
    initial_states = np.atleast_2d(initial_states)
    n_envs = len(initial_states)
    ddim = len(initial_states[0])
    data = np.zeros((n_envs, ddim, length))
    data[:, :, 0] = initial_states
    for t in range(1, length):
        data[:, :, t] = close_loop_dynamics(data[:, :, t-1]).detach().numpy()
    data = data + np.random.normal(0, noise_std, data.shape) # zero-mean Guassian noise
    return data

def make_dataset_from_trajectories(closed_loop_dynamics, initial_states, dt, state_norm, length, n_reps, noise_std):
    """Make a dataset of trajectories which is compatible with ode package. It stores trajectories and meta info.
    
    Parameters
    ----------
    close_loop_dynamics : a Torch function that represents the ode rhs.
    initial_states: normalized initial states of the intented trajectories.
    dt: sampling time interval
    state_norm: Because the input points are normalized to [-1, 1], if state_norm is not None, 
        it denormalizes the value of states to their actual values which can be used for ode learning purpose.
        if state_norm is None, do not denormalize the trajectories.
    length: Length of the intended trajectory.
    noise_std: Standard deviation of the observation noise


    Returns
    -------
    dataset: A dictionary {"envs": ndarray of initial states, "trajectories": ndarray of trajectories, "meta_info": a dictionary of
                            information such as the length of the trajectories and the standard deviation of the observation noise.}
    """
    
    data = {"envs":initial_states, "trajectories":[]}
    data['meta_info'] = {'L':length, 'T': length * dt, 'obs_noise_std':noise_std}
    trajs = generate_trajectories_from_inits(closed_loop_dynamics, initial_states, dt, n_reps, length, noise_std)
    if state_norm is not None:
        Tx = np.diag(state_norm)
    else:
        Tx = np.eye(len(initial_states[0]))
    state = np.moveaxis(trajs, 1, 3)
    state = np.dot(state, Tx)
    trajs = np.moveaxis(state, 3, 1)
    data["trajectories"] = trajs
    return data
    


def PT_indicator(x, a, b):
    if a is None:
        out = torch.le(x, b).type(x.dtype)
    elif b is None:
        out = torch.ge(x, a).type(x.dtype)
    else:
        out = torch.le(x, b).type(x.dtype) * torch.ge(x, a).type(x.dtype)
    return out

def PT_loose_thresh(x, a, b, ma, mb):
    """
    All elements are torch tensor the same dtype as x
    ma: slope of the negative section
    mb: slope of the positive section
    a: left side of the middle segment
    b: right side of the middle segment
    """
    out = PT_indicator(x, a, b) * x + PT_indicator(x, b, None) * (b + mb * (x - b)) + PT_indicator(x, None, a) * (a + ma * (x-a))
    return out