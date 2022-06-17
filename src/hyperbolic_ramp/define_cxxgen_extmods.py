
from ufunc_config_types import ExtMod, Func


hyperbolic_ramp_docstring = """\
hyperbolic_ramp(x, a, /, ...)

Compute the *hyperbolic ramp* function (a smoothed ramp):

    hyperbolic_ramp(x, a) = (x + sqrt(x*x + 4*a*a))/2

As x -> inf, the function approaches the line y = x.
As x -> -inf, the function approaches 0.

Parameters
----------
x : array_like
    Input values
a : array_like
    Determines the sharpness of the ramp transition.
    Note that `hyperbolic_ramp(0, a) = abs(a)`.

Returns
-------
out : ndarray
    The computed values.

Examples
--------
>>> import numpy as np
>>> from ufunclab import hyperbolic_ramp

>>> x = np.array([-4, -1, 0, 1.5, 5])
>>> a = np.array([[1.0], [0.5], [0.1]])
>>> hyperbolic_ramp(x, a)
array([[-0.07194484, -0.26894142,  0.        ,  1.22636171,  4.96653575],
       [-0.00989049, -0.18242552,  0.        ,  1.3569758 ,  4.99723611]])

"""

funcs = [
    Func(cxxname='hyperbolic_ramp',
         ufuncname='hyperbolic_ramp',
         types=['ff->f', 'dd->d', 'gg->g'],
         docstring=hyperbolic_ramp_docstring),
]

extmods = [ExtMod(modulename='_hyperbolic_ramp',
                  funcs={'hyperbolic_ramp.h': funcs})]
