
import numpy as np
import matplotlib.pyplot as plt
from ufunclab import semivar


h = np.linspace(0, 4, 500)
nugget = 0.6
sill = 2.0
rng = 2.5

plt.plot(h, semivar.exponential(h, nugget, sill, rng),
         alpha=0.75, label='exponential')
plt.plot(h, semivar.linear(h, nugget, sill, rng), '--',
         alpha=0.75, label='linear')
plt.plot(h, semivar.spherical(h, nugget, sill, rng), '-.',
         alpha=0.75, label='spherical')
plt.plot(h, semivar.parabolic(h, nugget, sill, rng), ':',
         alpha=0.75, label='parabolic')

plt.title(f'Semivariograms\n{nugget = }  {sill = }  {rng = }')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.5)
plt.xlabel('h')
# plt.show()
plt.savefig('semivar_demo.png')
