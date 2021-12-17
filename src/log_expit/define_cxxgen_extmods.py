
from ufunc_config_types import ExtMod, Func


log_expit_docstring = """\
log_expit(x, /, ...)

Compute the logarithm of the logistic sigmoid function.

The name `expit` is taken from SciPy, where `scipy.special.expit`
implements the logistic sigmoid function.

Parameters
----------
x : array_like
    Input values

Returns
-------
out : ndarray
    The computed values.

Examples
--------
>>> import numpy as np
>>> from ufunclab import log_expit

>>> log_expit([-3.0, 0.25, 2.5, 5.0])
array([-3.04858735, -0.57593942, -0.07888973, -0.00671535])

Large negative values:

>>> log_expit([-100, -500, -1000])
array([ -100.,  -500., -1000.])

Large positive values:

>>> log_expit([25, 100, 400])
array([-1.38879439e-011, -3.72007598e-044, -1.91516960e-174])
"""


funcs = [
    Func(cxxname='log_expit',
         ufuncname='log_expit',
         types=['f->f', 'd->d', 'g->g'],
         docstring=log_expit_docstring),
]

extmods = [ExtMod(modulename='_log_expit',
                  funcs={'log_expit.h': funcs})]
