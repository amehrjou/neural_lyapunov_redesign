import numpy as np
import sys
sys.path.append('../mars/')
import mars
from mars.configuration import Configuration
config = Configuration()
del Configuration
from examples.example_utils import LyapunovNetwork
from mars.utils import get_batch_grad
from examples.example_utils import balanced_class_weights
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def initialize_roa(grid, method='ball', cutoff_radius=0.0):
    """Takes a grid and mark the states within the ball of the specified radius as safe """
    initial_safe_set = np.linalg.norm(grid.all_points, ord=2, axis=1) <= cutoff_radius
    return initial_safe_set

def initialize_lyapunov_nn(grid, dynamics, L_dyn, policy, initial_safe_set, decrease_thresh, state_dim, layer_dims, layer_activations):
    """ Takes configuration of the neural net that acts as the lyapunov function 
    and outputs the initialized network"""
    lyapunov_function = LyapunovNetwork(state_dim, layer_dims, layer_activations, initializer=torch.nn.init.xavier_uniform)
    grad_lyapunov_nn = lambda x: get_batch_grad(lyapunov_function, x)
    L_v = lambda x: torch.norm(grad_lyapunov_nn(x), p=1, dim=1, keepdim=True)
    tau = np.sum(grid.unit_maxes) / 2
    closed_loop_dynamics = lambda states: dynamics(torch.tensor(states))
    lyapunov_nn = mars.Lyapunov(grid, lyapunov_function, dynamics, L_dyn, L_v, tau, policy, initial_safe_set, decrease_thresh=0)
    dv_nn = lambda x: lyapunov_function(closed_loop_dynamics(x)) - lyapunov_nn.lyapunov_function(x)
    return lyapunov_nn, grad_lyapunov_nn, dv_nn, L_v, tau

def initialize_lyapunov_quadratic(grid, P, dynamics, L_dyn, policy, initial_safe_set, decrease_thresh):
    lyapunov_function = mars.QuadraticFunction(P)
    grad_lyapunov = mars.LinearSystem((2 * P,))
    L_v = lambda x: torch.norm(grad_lyapunov(x), p=1, dim=1, keepdim=True)
    tau = np.sum(grid.unit_maxes) / 2
    lyapunov_pre = mars.Lyapunov(grid, lyapunov_function, dynamics, L_dyn, L_v, tau, policy, initial_safe_set)
    return lyapunov_pre, grad_lyapunov, L_v, tau

def pretrain_lyapunov_nn(grid, lyapunov_nn, target_lyapunov, batchsize, n_iters, learning_rate, verbose=False):
    """
    Takes initialized lyapunov_nn and pretrain it to match target_lyapunov. 
    target_lyapunov is usually a simple quadratic function.
    """
    ind_range = len(grid.all_points)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(lyapunov_nn.lyapunov_function.net.parameters(), lr=learning_rate)
    loss_monitor = []
    for i in tqdm(range(n_iters)):
        optimizer.zero_grad()
        batch_inds = np.random.choice(ind_range, batchsize, replace=False)
        states = grid.all_points[batch_inds, :]
        output_nn = lyapunov_nn.lyapunov_function(states)
        output_pre = target_lyapunov.lyapunov_function(states)
        loss = criterion(output_pre, output_nn)
        if verbose:
            print("loss: {}".format(loss))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_monitor.append(loss.detach().numpy())
    return lyapunov_nn

def train_lyapunov_nn(grid, lyapunov_nn, closed_loop_dynamics, batchsize, horizon, level_multiplier,
                      niters, learning_rate, Decrease_loss_coeff, Lipschitz_loss_coeff, difference_loss_coeff, fullpath_to_save_objectives=None, verbose=False):
    """
    grid: the state discritization in which everything happens
    lyapunov_nn: The pytorch neural network that implements the Lyapunov function
    batchsize: Batchsize for training the neural network
    horizon: How long the simulated trajectory must be to verify if a starting state belongs to the ROA
    level_multiplier: The multiplicative factor of the maximum stable levelset of the current Lyapunov function.
                    The larger this number is, the larger the gap will be from which the initial states are chosen.
    niters: number of optimization iterations to learn the shape of the Lyapunov function whose ROA covers
            the new stable initial states.
    learning_rate: The learning rate of training the neural network.
    lagrange_multiplier_factor: The weight of the component of the loss function that enforces the decrease condition
                                of the Lyapunov function.
    verbose: If True, the loss funtion is printing during learning.
    """
    
    optimizer = optim.SGD(lyapunov_nn.lyapunov_function.net.parameters(), lr=learning_rate)
    Lipschitz_est_func = lambda x, e: (lyapunov_nn.lyapunov_function(x + e) - 
                                       lyapunov_nn.lyapunov_function(x)) / e
    lyapunov_nn.update_values()
    lyapunov_nn.update_safe_set()
    c = lyapunov_nn.c_max.detach().numpy()
    idx_small = lyapunov_nn.values.detach().numpy().ravel() <= c
    idx_big = lyapunov_nn.values.detach().numpy().ravel() <= c * level_multiplier
    idx_gap = np.logical_and(idx_big, ~idx_small)
    roa_estimate = np.copy(lyapunov_nn.safe_set)
    gap_states = torch.tensor(grid.all_points[idx_gap])
    for _ in range(horizon):
        gap_states = closed_loop_dynamics(gap_states)
    gap_future_values = lyapunov_nn.lyapunov_function(gap_states)
    idx_gap_stable = (gap_future_values.detach().numpy() <= c).ravel()
    roa_estimate[idx_gap] |= (gap_future_values.detach().numpy() <= c).ravel()
    global_ind_gap_stable = np.copy(idx_gap)
    global_ind_gap_stable[idx_gap] &= (gap_future_values.detach().numpy() <= c).ravel()
    # Identify the class labels for our current ROA estimate and the expanded level set
    target_idx = idx_gap
    target_set = grid.all_points[target_idx]
    target_labels = roa_estimate[target_idx].reshape([-1, 1])
    safe_level = torch.tensor(c)
    dv_nn = lambda x: lyapunov_nn.lyapunov_function(closed_loop_dynamics(x)) - lyapunov_nn.lyapunov_function(x)
    all_objectives_record = {"classifier":np.zeros(niters), "decrease":np.zeros(niters), "Lipschitz":np.zeros(niters), "difference":np.zeros(niters)}
    for ind_in in tqdm(range(niters)):
        # Training step
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=False)
        target_states_batch = target_set[batch_inds]
        target_labels_batch = target_labels[batch_inds]
        class_weights, class_counts = balanced_class_weights(target_labels_batch)
        class_labels = torch.tensor(2 * target_labels_batch - 1, dtype=torch.float64)
        decision_distance = (1 - lyapunov_nn.lyapunov_function(target_states_batch)).reshape(-1, 1)
        classifier_loss = torch.max(- class_labels * decision_distance, torch.tensor(0, dtype=torch.float64))
        Decrease_Lagrange_multiplier = torch.ones_like(class_labels) * Decrease_loss_coeff
        decrease_loss = torch.tensor(target_labels_batch, dtype=torch.float64).reshape(-1, 1) * torch.max(dv_nn(target_states_batch), torch.tensor(0, dtype=torch.float64)).reshape(-1, 1)
        e = 0.1 # the direction to compute Lipschitz constant
        idx_for_Lipschitz_objective = np.random.choice(grid.all_points.shape[0], batchsize, replace=False)
        Lipschitz_loss = torch.norm(Lipschitz_est_func(grid.all_points[idx_for_Lipschitz_objective], e), p=2, dim=1)
        ## Difference loss
        idx_for_difference_objective = np.random.choice(grid.all_points.shape[0], batchsize, replace=False)
        temp_states = grid.all_points[idx_for_difference_objective]
        for step in range(10):
            temp_states = closed_loop_dynamics(temp_states)
        values_for_previous_stage = lyapunov_nn.lyapunov_function(temp_states).detach()
        difference_loss = lyapunov_nn.lyapunov_function(grid.all_points[idx_for_difference_objective]) - values_for_previous_stage
        difference_loss = torch.pow(difference_loss, 2)

        objective_classifier = torch.mean(classifier_loss)
        objective_decrease_condition = torch.mean(Decrease_Lagrange_multiplier * decrease_loss)
        objective_Lipschitz = torch.mean(Lipschitz_loss_coeff * Lipschitz_loss)
        objective_difference = torch.mean(difference_loss_coeff * difference_loss)
        objective = objective_classifier + objective_decrease_condition + objective_Lipschitz + objective_difference
        all_objectives_record["classifier"][ind_in], all_objectives_record["decrease"][ind_in], all_objectives_record["Lipschitz"][ind_in], all_objectives_record["difference"][ind_in]  = objective_classifier, objective_decrease_condition, objective_Lipschitz, objective_difference
        if verbose:
            print("Classifier los:{:8f}\nDecrease_loss:{:8f}\nLipschitz loss:{:8f}\nDifference loss:{:8f}".format(objective_classifier.detach().numpy(), objective_decrease_condition.detach().numpy(), objective_Lipschitz.detach().numpy(), objective_difference.detach().numpy()))
        objective.backward()
        optimizer.step()
    if fullpath_to_save_objectives is not None:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
        y_axis_values = np.concatenate([all_objectives_record["classifier"].reshape(niters, 1), all_objectives_record["decrease"].reshape(niters, 1), all_objectives_record["Lipschitz"].reshape(niters, 1), all_objectives_record["difference"].reshape(niters, 1)], 1)
        ax.plot(np.arange(0, niters).reshape(niters, 1), y_axis_values, linewidth=1)
        ax.legend(["Classifier", "Decrease", "Lipschitz", "difference"])
        ax.set_xlabel("iters", fontsize=20)
        ax.set_ylabel("objective value", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
        plt.xticks(np.linspace(0, niters, 10))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(fullpath_to_save_objectives, dpi=config.dpi)
        plt.close(fig)

    lyapunov_nn.update_values()
    lyapunov_nn.update_safe_set()
    print("cmax:{}".format(lyapunov_nn.c_max))
    return lyapunov_nn
    
def sample_around_roa(lyapunov_nn, expansion_factor, n_samples, method="gap"):
    """
    Takes the lyapunov_nn and expand its safe levelset such by the expansion_factor and choose
    samples from this expansion gap.
    
    
    Parameters
    ----------
    lyapunov_nn : lyapunov_nn class
    expansion_factor: a real number larger than 1.0
    n_samples: number of intended initial states to be chosen
    method: {"gap", etc}, a method to choose the samples. 
            "gap: chooses initial states from the gap"
    
    Returns
    -------
    A binary mask of the chosen indices. The size is the same as
    len(lyapunov_nn.discretization.all_points) and the where it is all False
    and only True for the chosen states.
    
    """

    grid = lyapunov_nn.discretization
    c = lyapunov_nn.c_max.detach().numpy()
    idx_small = lyapunov_nn.values.detach().numpy().ravel() <= c
    idx_big = lyapunov_nn.values.detach().numpy().ravel() <= c * expansion_factor
    idx_gap = np.logical_and(idx_big, ~idx_small)
    chosen_numerical_indx = np.random.choice(np.where(idx_gap == True)[0], n_samples, replace=False)
    idx_chosen = np.zeros_like(idx_gap) == 1
    idx_chosen[chosen_numerical_indx] = True
    return idx_chosen


def sample_blindly(grid, n_samples, method="uniform", rad=None):
    """
    Takes the grid and choose the initial states from that grid based on the provided  method.
    Note that the Lyapunov function or ROA is not used here for sampling.
    
    
    Parameters
    ----------
    grid : state discretization
    n_samples: number of intended initial states to be chosen
    method: {"uniform"}, determines the method of sampling from the grid.
    
    Returns
    -------
    A binary mask of the chosen indices. The size is the same as
    len(grid.all_points) where it is all False
    and only True for the chosen states.
    
    """
    if method == "uniform":
        chosen_numerical_indx = np.random.choice(grid.nindex, n_samples, replace=False)
        idx_chosen = np.zeros(grid.nindex) == 1
        idx_chosen[chosen_numerical_indx] = True
    elif method == "ball":
        idx_feasible = np.linalg.norm(grid.all_points, 2, axis=1) <= rad
        if sum(idx_feasible) < n_samples:
            raise ValueError("The number of chosen samples is larger than the size of the feasible set")
        else:
            feasible_numerical_idx = np.where(idx_feasible)[0]
            chosen_numerical_indx = np.random.choice(feasible_numerical_idx, n_samples, replace=False)
            idx_chosen = np.zeros(grid.nindex) == 1
            idx_chosen[chosen_numerical_indx] = True
    return idx_chosen
