import numpy as np
import matplotlib.pyplot as plt
from ufunclab import logistic, logistic_deriv


x = np.linspace(-5.25, 5.25, 401)
y = logistic(x)
dy = logistic_deriv(x)

plt.figure(figsize=(6, 4))
plt.subplot(2, 1, 1)
plt.plot(x, y, label='logistic(x)')
plt.grid(alpha=0.6)
plt.legend(framealpha=1, shadow=True)
plt.subplot(2, 1, 2)
plt.plot(x, dy, '--', label='logistic_deriv(x)')
plt.xlabel('x')
plt.grid(alpha=0.6)
plt.legend(framealpha=1, shadow=True)
# plt.show()
plt.savefig('logistic_demo.png')
