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


def initialize_controller():
    pass

def pre_train_controller():
    pass

def train_controller(grid, policy, lyapunov_model, dynamics, batchsize, horizon, level_multiplier, 
                    niters, learning_rate, fullpath_to_save_objectives=None, verbose=False):
    """
    Take initial policy and train it to enhance the ROA
    """
    closed_loop_dynamics = lambda states: dynamics(torch.tensor(states), policy(torch.tensor(states))) # Close loop dynamics and true region of attraction
    optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
    lyapunov_model.update_values()
    lyapunov_model.update_safe_set()
    c = lyapunov_model.c_max.detach().numpy()
    idx_small = lyapunov_model.values.detach().numpy().ravel() <= c
    idx_big = lyapunov_model.values.detach().numpy().ravel() <= c * level_multiplier
    idx_gap = np.logical_and(idx_big, ~idx_small)
    roa_estimate = np.copy(lyapunov_model.safe_set)
    gap_states = torch.tensor(grid.all_points[idx_gap])
    target_idx = idx_gap
    target_set = grid.all_points[target_idx]
    safe_level = torch.tensor(c)
    dv_nn = lambda x: lyapunov_model.lyapunov_function(closed_loop_dynamics(x)) - lyapunov_model.lyapunov_function(x)
    all_objectives_record = {"policy":np.zeros(niters)}
    for ind_in in tqdm(range(niters)):
        optimizer.zero_grad()
        batch_inds = np.random.choice(target_set.shape[0], batchsize, replace=False)
        target_states_batch = target_set[batch_inds]
        target_states_values_batch = torch.max(dv_nn(target_states_batch), torch.tensor(0, dtype=torch.float64))
        policy_loss = target_states_values_batch
        objective = torch.mean(policy_loss)
        objective.backward()
        optimizer.step()
    return policy