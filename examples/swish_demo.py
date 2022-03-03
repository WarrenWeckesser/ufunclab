import numpy as np
import matplotlib.pyplot as plt

from ufunclab import swish


beta = 1.5
x = np.linspace(-4.5, 3.5, 250)
y = swish(x, beta)

plt.figure(figsize=(6, 3.2))
plt.plot(x, y)

plt.title(f'swish(x, {beta})')

plt.xlabel('x')
plt.grid(alpha=0.6)
plt.tight_layout()
# plt.show()
plt.savefig('swish_demo.png')
