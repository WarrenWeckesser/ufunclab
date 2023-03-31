import numpy as np
import matplotlib.pyplot as plt
from ufunclab import linear_interp1d


# The first row is the x coordinates, and the second row is the y coordinates.
points = np.array([[3.0, 7.0,  1.0, -3.0, -0.5, 3.0],
                   [8.0, 3.0, -2.0,  1.0,  3.0, 1.8]])
n = points.shape[1]

# Use the cumulative distance `r` along the connecting segments of the points
# as the independent variable.
d = np.zeros(n)
d[1:] = np.linalg.norm(np.diff(points, axis=1), axis=0)
r = d.cumsum()

t = np.linspace(0, r[-1], 80)
p = linear_interp1d(t, r, points[:, None, :])

plt.plot(points[0], points[1], 'go', label='given points')
plt.plot(p[0], p[1], 'k.', markersize=4, alpha=0.5,
         label='linear interpolation\n(arclength parameterization)')
plt.axis('equal')
plt.grid(True, alpha=0.4)
plt.legend(framealpha=1, shadow=True)
plt.title('Demonstration of ufunclab.linear_interp1d')
# plt.show()
plt.savefig('linear_interp1d_demo.png')
