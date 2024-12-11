import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm


# Define the variable bounds
bounds = [
    Real(-5.12, 5.12, name='x'),
    Real(-5.12, 5.12, name='y')
]

def objective(params):
    x = params[0]
    y = params[1]
    return 20 + (x**2) + (y**2) - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

# Perform Bayesian optimization using Gaussian Processes.
# gp_minimize will try to find the minimal value of the objective function.
result = gp_minimize(
    objective,
    dimensions=bounds,
    n_calls=30,          # number of evaluations of f
    acq_func="EI",       # use Expected Improvement
    random_state=0,
    n_jobs=1,
)  #x0, y0, kappa[exploitation, exploration], xi [minimal improvement default 0.01]

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
