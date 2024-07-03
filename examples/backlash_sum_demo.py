import numpy as np
import matplotlib.pyplot as plt

from ufunclab import backlash_sum


t = np.linspace(0, 4*np.pi, 1000)
x = 2*np.sin(t)

w = np.array([0.125, 0.25, 0.25])
deadband = np.array([0.5, 1.0, 1.5])
initial = np.zeros(3)

y, final = backlash_sum(x, w, deadband, initial)

plt.figure(figsize=(6, 4))
plt.plot(t, x, linewidth=2, label='x(t)')
plt.plot(t, y, '--', linewidth=2, label='backlash_sum(x(t))')
plt.xlabel('t')
plt.title('backlash_sum')
plt.legend(shadow=True, loc='upper right')
plt.xlim(-0.99, 14.5)
plt.grid(alpha=0.6)
plt.savefig('backlash_sum_demo_x_vs_t.png')

plt.figure(figsize=(6, 4))
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('backlash_sum')
plt.grid(alpha=0.6)
plt.savefig('backlash_sum_demo_y_vs_x.png')

# plt.show()
