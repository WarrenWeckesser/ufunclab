
from ufunc_config_types import ExtMod, Func


yeo_johnson_docstring = """\
yeo_johnson(x, lmbda, /, ...)

Compute the Yeo-Johnson transform of x.

Parameters
----------
x : array_like
    Input values
lmbda : array_like
    Input values

Returns
-------
out : ndarray
    The computed values.

Examples
--------
>>> import numpy as np
>>> from ufunclab import yeo_johnson

>>> yeo_johnson([-1.5, -1.2, -1, 0.25, 1.1, 3.5, 4, 4.5], 2.5)
array([-0.73508894, -0.65160028, -0.58578644,  0.29877124,  2.15627886,
       16.78269478, 21.96067977, 27.97701535])
"""

inv_yeo_johnson_docstring = """\
inv_yeo_johnson(x, lmbda, /, ...)

Compute the inverse of the Yeo-Johnson transform of x.

Parameters
----------
x : array_like
    Input values
lmbda : array_like
    Input values

Returns
-------
out : ndarray
    The computed values.

Examples
--------
>>> import numpy as np
>>> from ufunclab import inv_yeo_johnson, yeo_johnson

>>> x = inv_yeo_johnson([-1.5, -1.2, -1, 0.25, 1.1, 3.5, 4.5], 2.5)
>>> x
array([-15.        ,  -5.25      ,  -3.        ,   0.21434292,
         0.6967291 ,   1.48657662,   1.7242969 ])
>>> yeo_johnson(x, 2.5)
array([-1.5 , -1.2 , -1.  ,  0.25,  1.1 ,  3.5 ,  4.5 ])
"""

funcs = [
    Func(cxxname='yeo_johnson',
         ufuncname='yeo_johnson',
         types=['ff->f', 'dd->d', 'gg->g'],
         docstring=yeo_johnson_docstring),
    Func(cxxname='inv_yeo_johnson',
         ufuncname='inv_yeo_johnson',
         types=['ff->f', 'dd->d', 'gg->g'],
         docstring=inv_yeo_johnson_docstring),
]

extmods = [ExtMod(modulename='_yeo_johnson',
                  funcs={'yeo_johnson.h': funcs})]
