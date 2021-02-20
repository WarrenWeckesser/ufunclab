import numpy as np
import matplotlib.pyplot as plt

from ufunclab import hysteresis_relay


t = np.linspace(0, 10.0, 1000)
x = 0.4*t*np.sin(t)
low_threshold = -0.5
high_threshold = 0.5
low_value = -1.0
high_value = 1.0
init = low_value
y = hysteresis_relay(x, low_threshold, high_threshold,
                     low_value, high_value, init)


plt.figure(figsize=(6, 4))
plt.plot(t, x, linewidth=2, label='x')
plt.plot(t, y, '--', linewidth=2,
         label='hysteresis_relay(x, -0.5, 0.5, -1, 1, -1)')
plt.axhline(low_threshold, color='k', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(high_threshold, color='k', linestyle='--', linewidth=1, alpha=0.5)
plt.title('hysteresis_relay example')
plt.xlabel('t')
plt.legend(shadow=True)
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('hysteresis_relay_demo.png')
