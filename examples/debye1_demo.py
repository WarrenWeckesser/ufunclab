import numpy as np
import matplotlib.pyplot as plt

from ufunclab import debye1


x = np.linspace(-5, 25, 301)
y = debye1(x)

plt.figure(figsize=(6, 4))
plt.plot(x, y)

plt.title('The Debye function D₁(x)')

plt.xlabel('x')
plt.grid(alpha=0.6)
# plt.show()
plt.savefig('debye1_demo.png')
