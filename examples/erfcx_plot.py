import numpy as np
import matplotlib.pyplot as plt
from ufunclab import erfcx


plt.figure(figsize=(6, 3.2))

x = np.linspace(-0.5, 7.5, 500)
plt.plot(x, erfcx(x))

plt.title('erfcx(x)')
plt.xlabel('x')
plt.xlim(-0.5, 7.5)
plt.ylim(0.0, 2.0)
plt.grid(alpha=0.6)
plt.tight_layout()
# plt.show()
plt.savefig('erfcx_plot.png')
