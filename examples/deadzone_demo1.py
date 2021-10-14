import numpy as np
import matplotlib.pyplot as plt

from ufunclab import deadzone


x = np.linspace(-1.25, 1.25, 500)
y = deadzone(x, -0.25, 0.5)

plt.figure(figsize=(6, 4))
plt.plot(x, y, label='deadzone(x, -0.25, 0.5)')

plt.title('deadzone(x, -0.25, 0.5)')
plt.plot([-0.25, 0.5], [0, 0], 'k.')
plt.annotate("low=-0.25", (-0.25, 0.04),
             horizontalalignment='center')
plt.annotate("high=0.5", (0.5, -0.04),
             horizontalalignment='center',
             verticalalignment='top')
plt.xlabel('x')
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('deadzone_demo1.png')
