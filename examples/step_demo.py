import numpy as np
import matplotlib.pyplot as plt

from ufunclab import step


a = 3.0
flow = -1.0
fa = 0.75
fhigh = 2.5
x = np.linspace(0, 6, 61)
y = step(x, a, flow, fa, fhigh)

plt.figure(figsize=(6, 4))
plt.plot(x, y, '.-')

plt.title(f'step(x, {a:g}, {flow:g}, {fa:g}, {fhigh:g})')

plt.xlabel('x')
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('step_demo.png')
