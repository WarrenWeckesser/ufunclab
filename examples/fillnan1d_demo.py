import numpy as np
import matplotlib.pyplot as plt

from ufunclab import fillnan1d


k = np.arange(50)
x = np.sin(0.06*k) + 0.2*np.sin(0.03*k) + 0.05*np.sin(31*k)
x[5:12] = np.nan
x[20:22] = np.nan
x[28:29] = np.nan
x[39:47] = np.nan

y = fillnan1d(x)

plt.figure(figsize=(6, 3.2))

plt.plot(k, x, 'o-', markersize=5, alpha=0.75, markerfacecolor='#FFFFFF00',
         label='x (gaps are nan)')
plt.plot(fillnan1d(x), 'o', alpha=0.75, markersize=2,
         label='fillnan1d(x)')

plt.xlabel('k')
plt.grid(alpha=0.6)
plt.legend(framealpha=1, shadow=True)
plt.tight_layout()
# plt.show()
plt.savefig('fillnan1d_demo.png')
