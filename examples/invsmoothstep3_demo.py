import numpy as np
import matplotlib.pyplot as plt

from ufunclab import invsmoothstep3


a = 1.0
b = 2.5
fa = -1
fb = 3
y = np.linspace(fa, fb, 201)
x = invsmoothstep3(y, a, b, fa, fb)

plt.figure(figsize=(6, 4))
plt.plot(y, x)

plt.title(f'invsmoothstep3(y, {a:g}, {b:g}, {fa:g}, {fb:g})')

plt.xlabel('y')
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('invsmoothstep3_demo.png')
