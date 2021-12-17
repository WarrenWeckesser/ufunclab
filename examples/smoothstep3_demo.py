import numpy as np
import matplotlib.pyplot as plt

from ufunclab import smoothstep3


a = 1.0
b = 2.5
fa = -1
fb = 3
x = np.linspace(0, 3.5, 201)
y = smoothstep3(x, a, b, fa, fb)

plt.figure(figsize=(6, 4))
plt.plot(x, y)

plt.title(f'smoothstep3(x, {a:g}, {b:g}, {fa:g}, {fb:g})')

plt.xlabel('x')
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('smoothstep3_demo.png')
