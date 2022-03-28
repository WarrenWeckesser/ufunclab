# This example requires scipy and matplotlib.

import numpy as np
from scipy.signal import ellip
import matplotlib.pyplot as plt
from ufunclab import sosfilter


# Sample rate, samples per second.
fs = 50.0

rng = np.random.default_rng(122333444455555)
T = 6.0  # Duration of demonstration signal, seconds.
n = int(fs * T)
t = np.arange(n)/fs
# Frequencies of the "signal" in channel 0 and channel 1.
f = np.array([[2.5], [3.0]])
x = (np.sin(2*np.pi*f*t) + 0.05*rng.normal(size=(2, n)).cumsum(axis=1)
     + 0.05*rng.normal(size=(2, n)) + np.array([[-0.08], [-0.1]])*(t-3.5)**2)


# Edges of the passband.  In the interval [passband_left, passpand_right],
# the maximum ripple should be given by pass_ripple_db.  Frequencies are
# in the same units as fs.
passband_left = 2.0
passband_right = 6.0
pass_ripple_db = -20*np.log10(0.9995)
stop_atten_db = -20*np.log10(0.0001)
filter_order = 6

# Design an elliptic bandpass digital filter.
sos = ellip(filter_order, rp=pass_ripple_db, rs=stop_atten_db,
            btype='bandpass',
            Wn=[passband_left, passband_right], fs=fs,
            output='sos')

y = sosfilter(sos, x)

plt.figure(figsize=(6, 4))
for k in range(2):
    plt.subplot(2, 1, k + 1)
    plt.plot(t, x[k], alpha=0.75, label=f'channel {k}')
    plt.plot(t, y[k], alpha=0.75, linestyle=(0, [7, 1]), label='filtered')
    plt.ylim(-3.75, 2.5)
    plt.legend(framealpha=1, shadow=True, loc='best')
    plt.grid()

plt.xlabel('t')
# plt.show()
plt.savefig('sosfilter_demo.png')
