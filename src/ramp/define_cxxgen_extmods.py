
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
    `a` must not be negative.
    Note that `hyperbolic_ramp(0, a) = a`.

Returns
-------
out : ndarray
    The computed values.

Examples
--------
>>> import numpy as np
>>> from ufunclab import hyperbolic_ramp

>>> x = np.arange(-5.0, 6.0)
>>> a = 0.5
>>> y = hyperbolic_ramp(x, a)
>>> print(np.column_stack((x, y)))
[[-5.          0.04950976]
 [-4.          0.06155281]
 [-3.          0.08113883]
 [-2.          0.11803399]
 [-1.          0.20710678]
 [ 0.          0.5       ]
 [ 1.          1.20710678]
 [ 2.          2.11803399]
 [ 3.          3.08113883]
 [ 4.          4.06155281]
 [ 5.          5.04950976]]

"""


exponential_ramp_docstring = """\
exponential_ramp(x, a, /, ...)

Compute the *exponential ramp* function (a smoothed ramp):

    exponential_ramp(x, a) = a*log_2(1 + 2**(x/a))

This function is also known as the *softplus* function.

Parameters
----------
x : array_like
    Input values
a : array_like
    Determines the sharpness of the ramp transition.
    `a` must not be negative.
    Note that `exponential_ramp(0, a) = a`.

Returns
-------
out : ndarray
    The computed values.

Examples
--------
>>> import numpy as np
>>> from ufunclab import exponential_ramp

>>> x = np.linspace(-3.0, 3.0, 13)
>>> a = 0.5
>>> y = exponential_ramp(x, a)
>>> print(np.column_stack((x, y)))
[[-3.          0.01118391]
 [-2.5         0.02219706]
 [-2.          0.04373142]
 [-1.5         0.0849625 ]
 [-1.          0.16096405]
 [-0.5         0.29248125]
 [ 0.          0.5       ]
 [ 0.5         0.79248125]
 [ 1.          1.16096405]
 [ 1.5         1.5849625 ]
 [ 2.          2.04373142]
 [ 2.5         2.52219706]
 [ 3.          3.01118391]]

"""

funcs = [
    Func(cxxname='hyperbolic_ramp',
         ufuncname='hyperbolic_ramp',
         types=['ff->f', 'dd->d', 'gg->g'],
         docstring=hyperbolic_ramp_docstring),
    Func(cxxname='exponential_ramp',
         ufuncname='exponential_ramp',
         types=['ff->f', 'dd->d', 'gg->g'],
         docstring=exponential_ramp_docstring),
]

extmods = [ExtMod(modulename='_ramp',
                  funcs={'ramp.h': funcs})]
