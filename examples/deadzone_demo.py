import numpy as np
import matplotlib.pyplot as plt

from ufunclab import deadzone


t = np.linspace(0, 10.0, 1000)
x = np.sin(t)
y = deadzone(x, -0.5, 0.5)


plt.figure(figsize=(6, 4))
plt.plot(t, x, linewidth=2, label='x')
plt.plot(t, y, '--', linewidth=2, label='deadzone(x, -0.5, 0.5)')

plt.title('Deadzone applied to a sine wave')
plt.xlabel('t')
plt.legend(shadow=True)
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('deadzone_demo.png')
