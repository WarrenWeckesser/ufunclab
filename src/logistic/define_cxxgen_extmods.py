
from ufunc_config_types import ExtMod, Func

logistic_docstring = """\
logistic(x, /, ...)

Compute the standard logistic sigmoid function

    logistic(x) = 1/(1 + exp(-x))

Parameters
----------
x : array_like
    Input values

Returns
-------
out : ndarray
    The computed values of the logistic sigmoid function.

Examples
--------
>>> import numpy as np
>>> from ufunclab import logistic

>>> logistic([-3, -0.25, 0, 0.25, 3])
array([0.04742587, 0.4378235 , 0.5       , 0.5621765 , 0.95257413])

"""

logistic_deriv_docstring = """\
logistic_deriv(x, /, ...)

Compute the derivative of the standard logistic sigmoid function.

Parameters
----------
x : array_like
    Input values

Returns
-------
out : ndarray
    The computed derivatives of the logistic sigmoid function.

Examples
--------
>>> import numpy as np
>>> from ufunclab import logistic_deriv

>>> logistic_deriv([-3, -0.25, 0, 0.25, 3])
array([...])

"""

log_logistic_docstring = """\
log_logistic(x, /, ...)

Compute the logarithm of the standard logistic sigmoid function.

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
    Func(cxxname='logistic',
         ufuncname='logistic',
         types=['f->f', 'd->d', 'g->g'],
         docstring=logistic_docstring),
    Func(cxxname='logistic_deriv',
         ufuncname='logistic_deriv',
         types=['f->f', 'd->d', 'g->g'],
         docstring=logistic_deriv_docstring),
    Func(cxxname='log_logistic',
         ufuncname='log_logistic',
         types=['f->f', 'd->d', 'g->g'],
         docstring=log_logistic_docstring),
]

extmods = [ExtMod(modulename='_logistic',
                  funcs={'logistic.h': funcs})]
