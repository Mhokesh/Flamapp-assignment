import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Load the data given in the assignment
df = pd.read_csv("xy_data.csv")
x_data = df["x"].values
y_data = df["y"].values
print(f"loaded {len(df)} points from xy_data.csv")

# instead of treating t as 1500 extra unknowns, we can solve
# for it directly from x, y, theta and X.
#
# from starting equations:
#   x - X = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta)
#   y - 42 = t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta)
#
# multiply eq1 by cos(theta), eq2 by sin(theta), add them.
# the wave terms cancel and cos^2+sin^2 = 1, leaving:
#
#   t = (x-X)*cos(theta) + (y-42)*sin(theta)
#
# multiply eq1 by -sin(theta), eq2 by cos(theta), add them:
#
#   exp(M*|t|)*sin(0.3t) = -(x-X)*sin(theta) + (y-42)*cos(theta)
#
# so for any guess of theta and X we get t for free, and then
# we just need M to make the second equation hold too.

def loss(params):
    theta, M, X = params

    t = (x_data - X) * np.cos(theta) + (y_data - 42) * np.sin(theta)

    # t must stay inside the range (6 < t < 60)
    penalty = 0.0
    below = np.maximum(6 - t, 0)
    above = np.maximum(t - 60, 0)
    if below.any() or above.any():
        penalty = 500 * np.sum(below + above)

    lhs = -(x_data - X) * np.sin(theta) + (y_data - 42) * np.cos(theta)
    rhs = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    return np.mean(np.abs(lhs - rhs)) + penalty


bounds = [
    (0.0, np.radians(50.0)),  # theta
    (-0.05, 0.05),             # M
    (0.0, 100.0),              # X
]

result = differential_evolution(loss, bounds, seed=42)
theta_r, M_opt, X_opt = result.x
theta_deg = np.degrees(theta_r)

print("\nfinal values:")
print(f"theta = {theta_deg:.4f} deg  ({theta_r:.6f} rad)")
print(f"M     = {M_opt:.2f}")
print(f"X     = {X_opt:.4f}")

# L1 distance: reconstruct x,y from the t we derived for each original point and compare to the original x,y directly.
t_recovered = (x_data - X_opt) * np.cos(theta_r) + (y_data - 42) * np.sin(theta_r)

x_pred = t_recovered * np.cos(theta_r) - np.exp(M_opt * np.abs(t_recovered)) * np.sin(0.3 * t_recovered) * np.sin(theta_r) + X_opt
y_pred = 42 + t_recovered * np.sin(theta_r) + np.exp(M_opt * np.abs(t_recovered)) * np.sin(0.3 * t_recovered) * np.cos(theta_r)

l1_x = np.mean(np.abs(x_data - x_pred))
l1_y = np.mean(np.abs(y_data - y_pred))
print(f"\nL1 distance (original vs predicted), x: {l1_x:.6f}")
print(f"L1 distance (original vs predicted), y: {l1_y:.6f}")
print(f"L1 distance overall (average of x and y): {(l1_x+l1_y)/2:.6f}")

# plot - original points as dots, predicted curve as a line
# the predicted curve(Blue) doesn't visible much in graph because it is very close to the original points(Red).
t_dense = np.linspace(6, 60, 2000)
x_dense = t_dense * np.cos(theta_r) - np.exp(M_opt * np.abs(t_dense)) * np.sin(0.3 * t_dense) * np.sin(theta_r) + X_opt
y_dense = 42 + t_dense * np.sin(theta_r) + np.exp(M_opt * np.abs(t_dense)) * np.sin(0.3 * t_dense) * np.cos(theta_r)

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_dense, color="#510cf3", linewidth=2.2, zorder=1,
         label=f"predicted curve (θ={theta_deg:.1f}°, M={M_opt:.3f}, X={X_opt:.1f})")
plt.scatter(x_data, y_data, s=14, color="#FDFBFB", edgecolors="red", linewidths=0.4,
            alpha=0.85, zorder=2, label="original points (xy_data.csv)")

plt.title("Original data vs. fitted parametric curve")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("curve_fit_result.png", dpi=300, facecolor="white")
print("\nsaved plot to curve_fit_result.png")