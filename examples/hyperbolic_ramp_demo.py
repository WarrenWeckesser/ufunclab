import numpy as np
import matplotlib.pyplot as plt

from ufunclab import hyperbolic_ramp


x = np.linspace(-5, 5, 250)

plt.figure(figsize=(6, 3.2))

for a, ls in [(0.0, '-'), (0.5, '--'), (1.0, ':')]:
    plt.plot(x, hyperbolic_ramp(x, a), ls, label=f'{a=}')

plt.title('hyperbolic_ramp(x, a)')
plt.xlabel('x')
plt.grid(alpha=0.6)
plt.legend(framealpha=1, shadow=True)
plt.tight_layout()
# plt.show()
plt.savefig('hyperbolic_ramp_demo.png')
