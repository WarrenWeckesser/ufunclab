import numpy as np
import matplotlib.pyplot as plt

from ufunclab import backlash


t = np.linspace(0, 20, 1000)
x = (21-t)*np.sin(t*t/20)/8

deadband = 1.0
y0 = 0.0

y = backlash(x, deadband, y0)


plt.figure(figsize=(6, 4))
plt.plot(t, x, linewidth=2, label='x(t)')
plt.plot(t, y, '--', linewidth=2, label='backlash(x(t))')
plt.plot(t, x + deadband/2, 'k:', alpha=0.5, linewidth=1,
         label=f'x(t) ± {deadband/2}')
plt.plot(t, x - deadband/2, 'k:', alpha=0.5, linewidth=1)

plt.title(f'Backlash, deadband={deadband}')
plt.legend(shadow=True)
plt.grid(alpha=0.6)
#plt.show()
plt.savefig('backlash_demo.png')
