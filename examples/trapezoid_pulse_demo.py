import numpy as np
import matplotlib.pyplot as plt

from ufunclab import trapezoid_pulse


a = 1.0
b = 3.0
c = 4.0
d = 5.0
amp = 2.0
x = np.linspace(0, 6, 301)
y = trapezoid_pulse(x, a, b, c, d, amp)

plt.figure(figsize=(6, 4))
plt.plot(x, y)

plt.title(f'trapezoid_pulse(x, {a:g}, {b:g}, {c:g}, {d:g}, {amp:g})')
#plt.plot([-0.25, 0.5], [0, 0], 'k.')
#plt.annotate("low=-0.25", (-0.25, 0.04),
#             horizontalalignment='center')
#plt.annotate("high=0.5", (0.5, -0.04),
#             horizontalalignment='center',
#             verticalalignment='top')
plt.xlabel('x')
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('trapezoid_pulse_demo.png')
