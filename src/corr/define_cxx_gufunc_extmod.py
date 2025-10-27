from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


PEARSON_CORR_DOCSTRING = """\
pearson_corr(x, y, /, *)

Pearson's product-moment correlation coefficient.

Parameters
---------
x, y : array_like (1-d)
    Arrays for which the correlation coefficient is computed.

Returns
-------
r : scalar
    The Pearson product-moment correlation coefficient.
    The return value is `nan` if:

    * Either `x` or `y` contains `nan` or `inf`.
    * The lengths of `x` and `y` are 1.
    * The values in `x` or in `y` are all the same.

Example
-------
>>> import numpy as np
>>> from ufunclab import pearson_corr

>>> x = np.array([1.0, 2.0, 3.5, 7.0, 8.5, 10.0, 11.0])
>>> y = np.array([10, 11.5, 11.4, 13.6, 15.1, 16.7, 15.0])
>>> pearson_corr(x, y)
0.9506381287828244

`pearson_corr` is a gufunc with shape signature `(n),(n)->()`.  In
the following example, a trivial dimension is added to the array `a`
before passing it to `pearson_corr`, so the inputs are compatible for
broadcasting.  The correlation coefficient of each row of `a` with
each row of `b` is computed, giving a result with shape (3, 2).

>>> a = np.array([[2, 3, 1, 3, 5, 8, 8, 9],
                  [3, 3, 1, 2, 2, 4, 4, 5],
                  [2, 5, 1, 2, 2, 3, 3, 8]])
>>> b = np.array([[9, 8, 8, 7, 4, 4, 1, 2],
                  [8, 9, 9, 6, 5, 7, 3, 4]])
>>> pearson_corr(np.expand_dims(a, 1), b)
array([[-0.92758645, -0.76815464],
       [-0.65015428, -0.53015896],
       [-0.43575108, -0.32925148]])

"""


pearson_corr_src = UFuncSource(
    funcname='pearson_corr_core',
    typesignatures=['ff->f', 'dd->d', 'gg->g'],
)

pearson_corr_int_src = UFuncSource(
    funcname='pearson_corr_int_core',
    typesignatures=['bb->d', 'BB->d', 'hh->d', 'HH->d', 'ii->d', 'II->d',
                    'll->d', 'LL->d'],
)

pearson_corr_ufunc = UFunc(
    name='pearson_corr',
    header='corr_gufunc.h',
    docstring=PEARSON_CORR_DOCSTRING,
    signature='(n),(n)->()',
    sources=[pearson_corr_int_src, pearson_corr_src],
    process_core_dims_func="process_core_dims",
)

extmod = UFuncExtMod(
    module='_corr',
    docstring="This extension module defines the gufunc 'pearson_corr'.",
    ufuncs=[pearson_corr_ufunc],
)
