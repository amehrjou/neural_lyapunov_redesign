import numpy as np

class system_properties(object):
    def __init__(self, pendulum_dict):
        for k in pendulum_dict.keys():
            exec("self.{} = pendulum_dict[k]".format(k, k) )



# Evey dynamical system is defined as a dictionary
all_systems = {}
# Pendulum 1
state_dim = 2
action_dim = 1
theta_max = np.deg2rad(180 * 0.5)                     # angular position [rad]
omega_max = np.deg2rad(360 * 1.0)                     # angular velocity [rad/s]
state_norm = (theta_max, omega_max)
system_dict = { "type": "pendulum",
                "state_dim": 2,
                "action_dim": 1,
                "g":9.81, 
                "m":0.25,
                "L":0.5, 
                "b":0.0,
                "state_norm":state_norm,
                }
u_max = system_dict["g"] * system_dict["m"] * system_dict["L"] * np.sin(np.deg2rad(60))  # torque [N.m], control action
system_dict["action_norm"]  = (u_max,)
all_systems["pendulum1"] = system_properties(system_dict)
# Pendulum 2
state_dim = 2
action_dim = 1
theta_max = np.deg2rad(180)                     # angular position [rad]
omega_max = np.deg2rad(360)                     # angular velocity [rad/s]
state_norm = (theta_max, omega_max)
system_dict = { "type": "pendulum",
                "state_dim": 2,
                "action_dim": 1,
                "g":9.81, 
                "m":0.25,
                "L":0.5, 
                "b":0.1,
                "state_norm":state_norm, 
                }
u_max = system_dict["g"] * system_dict["m"] * system_dict["L"] * np.sin(np.deg2rad(60))  # torque [N.m], control action
system_dict["action_norm"]  = (u_max,)
all_systems["pendulum2"] = system_properties(system_dict)
# Vanderpol
state_dim = 2
action_dim = 1
x_max     = 10                   # linear position [m]
y_max     = 10                  # angular position [rad]
state_norm = (x_max, y_max)
action_norm = None
system_dict = { "type": "vanderpol",
                "state_dim": 2,
                "action_dim": None,
                "damping":3.0,
                "state_norm":state_norm, 
                "action_norm":action_norm,
                }
all_systems["vanderpol"] = system_properties(system_dict)






    