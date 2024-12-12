import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import biorbd

# Define the variable bounds
bounds = [
    Real(20, 50, name='R_frequency'),  # Hz
    Real(8, 20, name='R_intensity'),  #mA
    Real(200, 500, name='R_width'),  # micros
    Real(20, 50, name='L_frequency'),  # Hz
    Real(8, 20, name='L_intensity'),  # mA
    Real(200, 500, name='L_width'),  # micros
]

def objective(model, params):

    # Weights to tune depending on the user's preferences ?
    weight_comddot = 1
    weight_angular_momentum = 1
    weight_enegy = 1

    # Optimized FES parameters
    R_frequency = params[0]
    R_intensity = params[1]
    R_width = params[2]
    L_frequency = params[3]
    L_intensity = params[4]
    L_width = params[5]

    # Send stim... TODO

    # Receive data... TODO
    nb_frames = 100  # Let's say 100 frames
    q = np.zeros((16*3, nb_frames))
    qdot = np.zeros((16*3, nb_frames))
    qddot = np.zeros((16*3, nb_frames))
    tau = np.zeros((16*3, nb_frames))
    time_vector = np.linspace(0, 1, nb_frames)  # Let's say a step of 1s

    # Compute objective value
    comddot = np.zeros((nb_frames, ))
    angular_momentum = np.zeros((nb_frames, ))
    for i_frame in range(nb_frames):
        comddot[i_frame] = np.linalg.norm(model.CoMddot(q[:, i_frame], qdot[:, i_frame], qddot[:, i_frame]).to_array())
        angular_momentum[i_frame] = np.linalg.norm(model.angularMomentum(q[:, i_frame], qdot[:, i_frame]).to_array())

    # since the time is the same, min energy and power gives the same thing (same min)
    voltage = 0  # TODO
    power_stim = np.abs(R_intensity * voltage) + np.abs(L_intensity * voltage)
    power_total = np.sum(np.abs(tau * qdot), axis=0)
    power_human = power_total - power_stim
    energy_human = np.trapezoid(power_human, x=time_vector)

    return weight_comddot * comddot + weight_angular_momentum * angular_momentum + weight_enegy * energy_human

# Define the model, which will not change
model_path = "TODO.bioMod"
model = biorbd.Model(model_path)

# Perform Bayesian optimization using Gaussian Processes.
# gp_minimize will try to find the minimal value of the objective function.
result = gp_minimize(
    lambda params: objective(model, params),
    dimensions=bounds,
    n_calls=100,         # number of evaluations of f
    acq_func="LCB",      # "LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"
    kappa=5,  # *
    random_state=0,  #*
    n_jobs=1,
)  #x0, y0, kappa[exploitation, exploration], xi [minimal improvement default 0.01]
# TODO: stop when the same point has been hit t time (t=5 in general)

print("Best found minimum:")
print("X = %.4f, Y = %.4f" % (result.x[0], result.x[1]))
print("f(x,y) = %.4f" % result.fun)

# Optionally, plot convergence
fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122, projection="3d")

# Convergence plot
ax0.plot(result.func_vals, marker='o')
ax0.set_title('Convergence Plot')
ax0.set_xlabel('Number of calls')
ax0.set_ylabel('Objective function value')

# Plot the function sampling
x_iters_array = np.array(result.x_iters)
func_vals_array = np.array(result.func_vals)
colors_min = np.min(func_vals_array)
colors_max = np.max(func_vals_array)
normalized_cmap = (func_vals_array - colors_min) / (colors_max - colors_min)
colors = cm["viridis"](normalized_cmap)
p = ax1.scatter(x_iters_array[:, 0], x_iters_array[:, 1], result.func_vals, c=colors, marker='.')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Function Sampling')
cbar = fig.colorbar(p)
cbar.set_label('Objective function value')
plt.show()
