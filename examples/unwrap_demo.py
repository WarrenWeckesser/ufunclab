import numpy as np
import matplotlib.pyplot as plt
from ufunclab import unwrap


def generate_demo_data():
    x = np.linspace(0, 25, 801)
    return np.mod(1.5*np.sin(1.1*x + 0.26)*(1 - x/6 + (x/23)**3), 2.0) - 1


y = generate_demo_data()
u = unwrap(y, 2.0)

plt.figure(figsize=(6, 4))

plt.plot(y, label='input (a signal wrapped to [-1, 1])')
plt.plot(u, linewidth=2.5, alpha=0.5, label='unwrapped')

plt.grid(alpha=0.6)
plt.legend(framealpha=1, shadow=True)
plt.title('unwrap demo')
# plt.show()
plt.savefig('unwrap_demo.png')
