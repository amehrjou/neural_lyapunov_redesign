import numpy as np
import sys
sys.path.append('..')
from examples.example_utils import build_system, VanDerPol, InvertedPendulum, LyapunovNetwork, compute_roa, balanced_class_weights, generate_trajectories, save_dict, load_dict

import mars
from mars import config, Saturation
from mars.visualization import plot_roa, plot_trajs_on_levelsets, plot_levelsets, plot_nested_roas, plot_phase_portrait, plot_scalar_function_over_trajectories
from mars.utils import print_no_newline, compute_nrows_ncolumns, str2bool
from mars.roa_tools import initialize_roa, initialize_lyapunov_nn, initialize_lyapunov_quadratic, pretrain_lyapunov_nn, sample_around_roa, train_lyapunov_nn
from examples.systems_config import all_systems 

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from mars.utils import get_batch_grad, save_lyapunov_nn, load_lyapunov_nn, make_dataset_from_trajectories
from mars.controller_tools import train_controller
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import pickle
import os
import warnings
import random
import argparse
warnings.filterwarnings("ignore")


## Test with arguments
def getArgs(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', default='pendulum1', type=str, help='pick a dynamical system')
    parser.add_argument('--dt', default=0.01, type=float, help='time interval of simulations')
    parser.add_argument('--grid_resolution', default=100, type=int, help='number of division in every dimension of the state grid')
    parser.add_argument('--repetition_use', default='denoise', type=str, help='how to use the repititions in each environment {denoise,learn}')
    parser.add_argument('--roa_gridsize', default=100, type=int, help='number of discrerized states in each dimension of the ROA grid')
    parser.add_argument('--roa_batchsize', default=16, type=int, help='the batchsize to train the neural network of ROA')
    parser.add_argument('--roa_pre_lr', default=0.001, type=float, help='learning rate to pretrain the neural network of ROA')
    parser.add_argument('--roa_train_lr', default=0.1, type=float, help='learning rate to train the neural network of ROA while growing')
    parser.add_argument('--roa_pre_iters', default=10000, type=int, help='number of iterations to pretrain the neural network of ROA')
    parser.add_argument('--roa_inner_iters', default=100, type=int, help='number of iterations to train the neural network of ROA at each growth stage')
    parser.add_argument('--roa_outer_iters', default=100, type=int, help='number of growth stages to train the neural network of ROA')
    parser.add_argument('--roa_level_multiplier', default=3.0, type=float, help='determines how much the levelset grows at each growth stage (> 1.0)')
    parser.add_argument('--roa_decrease_loss_coeff', default=200, type=float, help='the weight of the decrease condition in the total objective')
    parser.add_argument('--roa_lipschitz_loss_coeff', default=0.000, type=float, help='the weight of the Lipschitz condition in the total objective')
    parser.add_argument('--roa_difference_loss_coeff', default=0.000, type=float, help='the weight of the difference between the previous and current values of the Lyapunov function in the total objective')
    parser.add_argument('--roa_nn_sizes', default='[16, 16, 32]', type=str, help='number of neurons at each layer of the neural network of ROA')
    parser.add_argument('--roa_nn_activations', default="['tanh', 'tanh', 'tanh']", type=str, help='the activation of each layer of the neural network of ROA')
    parser.add_argument('--lyapunov_decrease_threshold', default=0.0, type=float, help='considered as satisfied if less than this threshold (must be negative in theory)')
    parser.add_argument('--controller_outer_iters', default=10, type=int, help='number of times that a new controller is discovered')
    parser.add_argument('--controller_inner_iters', default=1000, type=int, help='number of iterations to train the controller at each growth stage')
    parser.add_argument('--controller_traj_length', default=100, type=int, help='the length of the trajectory produced to train the controller')
    parser.add_argument('--controller_train_lr', default=0.01, type=float, help='learning rate to train the controller')
    parser.add_argument('--controller_batchsize', default=16, type=int, help='the batchsize to train the controller')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate for all learners')
    parser.add_argument('--verbose', default=False, type=str2bool, help='print out the state of the learners')
    parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use GPU or CPU')
    parser.add_argument('--exp_num', default=0, type=int, help='the number of the experiment. It determines the folder in which the results are saved or loaded from.')
    parser.add_argument('--image_save_format', default="png", type=str, help='png or pdf')
    return parser.parse_args(argv)

input_args_str = "\
--system pendulum1\
--dt 0.01\
--grid_resolution 100\
--lr 0.001\
--repetition_use denoise\
--roa_gridsize 100\
--roa_batchsize 16\
--roa_pre_lr 0.001\
--roa_pre_iters 20000\
--roa_inner_iters 100\
--roa_outer_iters 20\
--roa_train_lr 0.01\
--roa_level_multiplier 4.0\
--lyapunov_decrease_threshold 0.0\
--roa_nn_sizes [64,64,64]\
--roa_nn_activations ['tanh','tanh','tanh']\
--roa_decrease_loss_coeff 1000\
--roa_lipschitz_loss_coeff 0.0\
--roa_difference_loss_coeff 0.000\
--controller_inner_iters 100\
--controller_outer_iters 20\
--controller_traj_length 10\
--controller_train_lr 0.01\
--controller_batchsize 16\
--verbose True\
--image_save_format pdf\
--exp_num 54\
--use_cuda False"

input_args_temp = input_args_str.split("--")
input_args = []
for ind, twins in enumerate(input_args_temp[1:]):
    a, b = twins.split(" ")
    a = "--{}".format(a)
    input_args.append(a)
    input_args.append(b)
args = getArgs(input_args)


exp_num = args.exp_num
results_dir = '../results/exp_{}'.format(exp_num)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# Set random seed
# torch.manual_seed(0)
# np.random.seed(0)
# Choosing the system
dt = args.dt   # sampling time
# System properties
system_properties = all_systems[args.system]
system = build_system(system_properties, dt)
state_dim     = system_properties.state_dim
action_dim    = system_properties.action_dim
state_limits  = np.array([[-1., 1.]] * state_dim)
action_limits = np.array([[-1., 1.]] * action_dim)
num_states = args.grid_resolution # Number of states divisions each dimension

# Initialize system class and its linearization
system = build_system(system_properties, dt)
A, B = system.linearize()
dynamics = lambda x, y: system.eval(x, y)

# State grid
grid_limits = np.array([[-1., 1.], ] * state_dim)
state_discretization = mars.GridWorld(grid_limits, num_states)
tau = np.sum(state_discretization.unit_maxes) / 2

# Set initial safe set as a ball around the origin (in normalized coordinates)
cutoff_radius    = 0.1
initial_safe_set = np.linalg.norm(state_discretization.all_points, ord=2, axis=1) <= cutoff_radius

# LQR policy and its true ROA
Q = np.identity(state_dim).astype(config.dtype) * 0.001     # state cost matrix
R = np.identity(action_dim).astype(config.np_dtype) * 0.001  # action cost matrix
K, P_lqr = mars.utils.dlqr(A, B, Q, R)
print("LQR matrix:")
print(K)
# K = np.random.uniform(-1, 7, (1, 2)) # randomly designed linear controller for comparison with LQR
# policy = mars.LinearSystem(- K, name='policy')
# policy = mars.Saturation(policy, -1, 1)
policy = mars.TrainableController(-K, name='policy', args={'low_thresh':-0.2, 'high_thresh':0.2, 'low_slope':0.0, 'high_slope':0.0})
# policy = mars.Saturation(policy, -1, 1)
closed_loop_dynamics = lambda states: dynamics(torch.tensor(states), policy(torch.tensor(states))) # Close loop dynamics and true region of attraction
horizon = 200 # smaller tol requires longer horizon to give an accurate estimate of ROA
tol = 0.01 # how much close to origin must be x(T) to be considered as stable trajectory
roa_true, trajectories = compute_roa(state_discretization, closed_loop_dynamics, horizon, tol, no_traj=False) # True ROA with LQR policy

# Quadratic Lyapunov function for the LQR controller and its induced ROA
L_pol = lambda x: np.linalg.norm(-K, 1) # # Policy (linear)
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x) # Dynamics (linear approximation)
lyapunov_function = mars.QuadraticFunction(P_lqr)
grad_lyapunov_function = mars.LinearSystem((2 * P_lqr,))
dv_lqr = lambda x: lyapunov_lqr.lyapunov_function(closed_loop_dynamics(x))- lyapunov_lqr.lyapunov_function(x) # function : v(x(t)) - v(x(t+1)) useful for checking the decrease condition
L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=1, keepdim=True) # Lipschitz constant of the Lyapunov function
lyapunov_lqr = mars.Lyapunov(state_discretization, lyapunov_function, dynamics, L_dyn, L_v, tau, policy, initial_safe_set, decrease_thresh=0)
lyapunov_lqr.update_values()
lyapunov_lqr.update_safe_set()
roa_lqr = lyapunov_lqr.safe_set
grid_size = state_discretization.num_points

roa_true, trajectories = compute_roa(state_discretization, closed_loop_dynamics, horizon, tol, no_traj=False)
# ROA Visualization
grid_size = state_discretization.num_points
system.normalize
plot_limits = np.column_stack(( np.array([-1, -3]), np.array([1, 3])))
plot_state_labels = [r"$\theta$", r"$\omega$"]
plot_roa(roa_true.reshape(grid_size), plot_limits, plot_state_labels, os.path.join(results_dir, 'true_roa_{}.{}'.format(args.system, args.image_save_format)))

###### NN Lyapunov ######

## Pretraining
layer_dims = eval(args.roa_nn_sizes)
layer_activations = eval(args.roa_nn_activations)
decrease_thresh = args.lyapunov_decrease_threshold
# Initialize nn Lyapunov
lyapunov_nn, grad_lyapunov_nn, dv_nn, L_v, tau = initialize_lyapunov_nn(state_discretization, dynamics, L_dyn, policy, 
                                                    initial_safe_set, decrease_thresh, state_dim, layer_dims, layer_activations)
lyapunov_nn.update_values()
lyapunov_nn.update_safe_set()
roa_nn = lyapunov_nn.safe_set

# Plot ROAs
grid_size = state_discretization.num_points
plot_roa(lyapunov_nn.safe_set.reshape(grid_size), plot_limits, plot_state_labels, os.path.join(results_dir, 'roa_nn_preinit.{}'.format(args.image_save_format)))
plot_levelsets(grid_limits, 0.5, lyapunov_nn.lyapunov_function, plot_state_labels, res=32, figsize=(10,10), 
            nlevels=10, full_path=os.path.join(results_dir, 'levelsets_nn_pre_init.{}'.format(args.image_save_format)))


# Initialize quadratic Lyapunov
P = 0.1 * np.eye(state_dim)
lyapunov_pre, grad_lyapunov_pre, L_v_pre, tau = initialize_lyapunov_quadratic(state_discretization, P, dynamics, L_dyn, policy, 
                                                    initial_safe_set, decrease_thresh)
lyapunov_pre.update_values()
lyapunov_pre.update_safe_set()

save_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'untrained_lyapunov_nn.net'))
pretrain_lyapunov_nn(state_discretization, lyapunov_nn, lyapunov_pre, args.roa_batchsize, args.roa_pre_iters, args.roa_pre_lr, verbose=False)
# plotting pretrained lyapunov_nn
plot_levelsets(grid_limits, 0.5, lyapunov_pre.lyapunov_function, plot_state_labels, res=32, figsize=(10,10), 
            nlevels=10, full_path=os.path.join(results_dir, 'levelsets_quadratic_pre.{}'.format(args.image_save_format)))
save_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'pretrained_lyapunov_nn.net'))
plot_roa(lyapunov_nn.safe_set.reshape(grid_size), plot_limits, plot_state_labels, os.path.join(results_dir, 'roa_nn_pretrained.{}'.format(args.image_save_format)))
plot_levelsets(grid_limits, 0.5, lyapunov_pre.lyapunov_function, plot_state_labels, res=32, figsize=(10,10), 
            nlevels=10, full_path=os.path.join(results_dir, 'levelsets_quadratic_pre.{}'.format(args.image_save_format)))
plot_levelsets(grid_limits, 0.5, lyapunov_nn.lyapunov_function, plot_state_labels, res=32, figsize=(10,10), 
            nlevels=10, full_path=os.path.join(results_dir, 'levelsets_nn_pretrained.{}'.format(args.image_save_format)))
horizon = 100
grid = state_discretization
optimizer = optim.SGD(lyapunov_nn.lyapunov_function.net.parameters(), lr=args.roa_train_lr)
boundary_grad = []
roa_captured_ratio = []
Lipschitz_est_func = lambda x, e: (lyapunov_nn.lyapunov_function(x + e) - lyapunov_nn.lyapunov_function(x)) / e

# growing the roa
lyapunov_nn.update_safe_set()
c = lyapunov_nn.c_max.detach().numpy()
idx_small = lyapunov_nn.values.detach().numpy().ravel() <= c
idx_big = lyapunov_nn.values.detach().numpy().ravel() <= c * args.roa_level_multiplier
plot_limits = np.column_stack(( np.array([-1, -3]), np.array([1, 3])))
plot_nested_roas(grid.num_points, idx_small, idx_big, plot_limits, plot_state_labels, 
                ind_gap_stable=None, ind_true_roa=roa_true, 
                full_path=os.path.join(results_dir, 'nested_roas_before_training.{}'.format(args.image_save_format)))

policy_init = copy.deepcopy(policy)
policy_current = policy
monitor_dict = {"roa_info":{}}

# Monitor the training process
training_info = {"grid_size":{}, "roa_info":{}, "policy_info":{}}
training_info["grid_size"] = grid.nindex
training_info["roa_info"] = {"true_roa_sizes": [], "estimated_roa_sizes": [], "level_values": []}
training_info["policy_info"] = {"low_thresh_param" : [], "high_thresh_param" : [], "low_slope_param" : [], "high_slope_param" : []}


for g in range(args.controller_outer_iters):
    # Approximate ROA
    closed_loop_dynamics = lambda states: dynamics(torch.tensor(states), policy_current(torch.tensor(states))) # Close loop dynamics and true region of attraction
    roa_true, trajectories = compute_roa(state_discretization, closed_loop_dynamics, 200, tol, no_traj=False) # True ROA with LQR policy
    plot_roa(roa_true.reshape(grid_size), plot_limits, plot_state_labels, os.path.join(results_dir, 'true_roa_{}_policyiter_{}.{}'.format(args.system, g, args.image_save_format)))
    print("Stable states:{}".format(roa_true.sum()))
    training_info["roa_info"]["true_roa_sizes"].append(sum(roa_true))
    training_info["policy_info"]["low_thresh_param"].append(copy.deepcopy(policy.low_thresh_param.detach().numpy()))
    training_info["policy_info"]["high_thresh_param"].append(copy.deepcopy(policy.high_thresh_param.detach().numpy()))
    training_info["policy_info"]["low_slope_param"].append(copy.deepcopy(policy.low_slope_param.detach().numpy()))
    training_info["policy_info"]["high_slope_param"].append(copy.deepcopy(policy.high_slope_param.detach().numpy()))
    training_info["roa_info"]["estimated_roa_sizes"].append([])
    training_info["roa_info"]["level_values"].append([])
    training_info["roa_info"]["estimated_roa_sizes"][-1].append(copy.deepcopy(sum(lyapunov_nn.safe_set)))
    training_info["roa_info"]["level_values"][-1].append(copy.deepcopy(lyapunov_nn.c_max.detach().numpy()))
    save_dict(training_info, os.path.join(results_dir, "training_info_policyiter_init.npy"))
    for k in range(args.roa_outer_iters):
        lyapunov_nn = train_lyapunov_nn(grid, lyapunov_nn, closed_loop_dynamics, args.roa_batchsize, horizon, args.roa_level_multiplier,
                            args.roa_inner_iters, args.roa_train_lr, args.roa_decrease_loss_coeff, args.roa_lipschitz_loss_coeff, args.roa_difference_loss_coeff,
                            fullpath_to_save_objectives=os.path.join(results_dir, 'objectives_{}_policyiter_{}.{}'.format(k, g, args.image_save_format)), verbose=False)
        save_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'trained_lyapunov_nn_roa_iter_{}_policyiter_{}.net'.format(k, g)))
        lyapunov_nn.update_values()
        lyapunov_nn.update_safe_set()
        c = lyapunov_nn.c_max.detach().numpy()
        idx_small = lyapunov_nn.values.detach().numpy().ravel() <= c
        idx_big = lyapunov_nn.values.detach().numpy().ravel() <= c * args.roa_level_multiplier
        plot_nested_roas(grid.num_points, idx_small, idx_big, plot_limits, plot_state_labels, 
                        ind_gap_stable=None, ind_true_roa=roa_true, 
                        full_path=os.path.join(results_dir, 'nested_roas_after_training_roa_iter_{}_policyiter_{}.{}'.format(k, g, args.image_save_format)))
        training_info["roa_info"]["estimated_roa_sizes"][-1].append(copy.deepcopy(sum(lyapunov_nn.safe_set)))
        training_info["roa_info"]["level_values"][-1].append(copy.deepcopy(lyapunov_nn.c_max.detach().numpy()))
    save_dict(training_info, os.path.join(results_dir, "training_info_policyiter_{}.npy".format(g)))
    # Train controller
    policy_next = train_controller(grid, policy_current, lyapunov_nn, dynamics, args.controller_batchsize, horizon, args.roa_level_multiplier, 
                args.controller_inner_iters, args.controller_train_lr, fullpath_to_save_objectives=None, verbose=False)
    policy_current = policy_next