"""
Wrappers of gufuncs that provide a nicer API.
"""

import operator
import numpy as np
try:
    from numpy.exceptions import AxisError
except ImportError:
    from numpy import AxisError
from ufunclab._convert_to_base import convert_to_base as _convert_to_base
from ufunclab._nextn import (nextn_less as _nextn_less,
                             nextn_greater as _nextn_greater)
from ufunclab._one_hot import one_hot as _one_hot


# XXX Except for `out` and `axis`, this function does not expose any of the
# gufunc keyword parameters.
#
def convert_to_base(k, base, ndigits, out=None, axis=-1):
    """
    Convert the integer `k` to the given `base`, using `ndigits` digits.
    The "digits" are the integer coefficients of the expansion of `k` as
    the sum of powers of `base`.  The digits are given from lowest order
    to highest.

    Only the lowest order `ndigits` are output.  If `k` is too big for
    the given number of digits, the result can be interpreted as being
    modulo `base**ndigits`.

    Notes
    -----
    This function is a Python wrapper of a gufunc with shape
    signature `(),()->(n)`.  The gufunc can be accessed as
    `convert_to_base.gufunc`.  The gufunc does not provide the `ndigits`
    parameter.  To use the gufunc directly, you must provide the `out`
    parameter; its shape determines `ndigits`.

    Examples
    --------
    >>> import numpy as np
    >>> from ufunclab import convert_to_base

    >>> convert_to_base(1249, 8, ndigits=4)
    array([1, 4, 3, 2])

    That result follows from 1249 = 1*8**0 + 4*8**1 + 3*8**2 + 2*8**3.

    Broadcasting applies to `k` and `base`:

    >>> x = np.array([10, 24, 85])    # shape is (3,)
    >>> base = np.array([[8], [16]])  # shape is (2, 1)
    >>> convert_to_base(x, base, ndigits=4)  # output shape is (2, 3, 4)
    array([[[ 2,  1,  0,  0],
            [ 0,  3,  0,  0],
            [ 5,  2,  1,  0]],
           [[10,  0,  0,  0],
            [ 8,  1,  0,  0],
            [ 5,  5,  0,  0]]])

    """
    k = np.asarray(k)
    base = np.asarray(base)
    try:
        ndigits = operator.index(ndigits)
    except TypeError:
        raise ValueError(f'ndigits must be an integer; got {ndigits!r}')
    param_bcast_shape = np.broadcast_shapes(k.shape, base.shape)
    adjusted_axis = axis
    if adjusted_axis < 0:
        adjusted_axis += 1 + len(param_bcast_shape)
    if adjusted_axis < 0 or adjusted_axis > 1 + len(param_bcast_shape):
        raise AxisError(f'invalid axis {axis}')
    out_shape = (param_bcast_shape[:adjusted_axis] + (ndigits,)
                 + param_bcast_shape[adjusted_axis:])
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(f'out.shape must be {out_shape}; '
                             f'got {out.shape}.')
    else:
        out = np.empty(out_shape, dtype=int)
    return _convert_to_base(k, base, out=out, axis=axis)


convert_to_base.gufunc = _convert_to_base


def nextn_greater(x, n, out=None, axis=-1):
    """
    Return the next n floating point values greater than x.

    x must be one of the real floating point types np.float32,
    np.float64 or np.longdouble.
    """
    x = np.asarray(x)
    if x.dtype.char not in 'fdg':
        raise ValueError('x must be an array of np.float32, np.float64 or '
                         'np.longdouble.')
    try:
        n = operator.index(n)
    except TypeError:
        raise ValueError(f'n must be an integer; got {n!r}')
    x_shape = x.shape
    adjusted_axis = axis
    if adjusted_axis < 0:
        adjusted_axis += 1 + len(x_shape)
    if adjusted_axis < 0 or adjusted_axis > 1 + len(x_shape):
        raise AxisError(f'invalid axis {axis}')
    out_shape = (x_shape[:adjusted_axis] + (n,)
                 + x_shape[adjusted_axis:])
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(f'out.shape must be {out_shape}; '
                             f'got {out.shape}.')
    else:
        out = np.empty(out_shape, dtype=x.dtype)
    return _nextn_greater(x, out=out, axis=axis)


nextn_greater.gufunc = _nextn_greater


def nextn_less(x, n, out=None, axis=-1):
    """
    Return the next n floating point values less than x.

    x must be one of the real floating point types np.float32,
    np.float64 or np.longdouble.
    """
    x = np.asarray(x)
    if x.dtype.char not in 'fdg':
        raise ValueError('x must be a scalar or array of np.float32, '
                         'np.float64 or np.longdouble.')
    try:
        n = operator.index(n)
    except TypeError:
        raise ValueError(f'n must be an integer; got {n!r}')
    x_shape = x.shape
    adjusted_axis = axis
    if adjusted_axis < 0:
        adjusted_axis += 1 + len(x_shape)
    if adjusted_axis < 0 or adjusted_axis > 1 + len(x_shape):
        raise AxisError(f'invalid axis {axis}')
    out_shape = (x_shape[:adjusted_axis] + (n,)
                 + x_shape[adjusted_axis:])
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(f'out.shape must be {out_shape}; '
                             f'got {out.shape}.')
    else:
        out = np.empty(out_shape, dtype=x.dtype)
    return _nextn_less(x, out=out, axis=axis)


nextn_less.gufunc = _nextn_less


def one_hot(k, n, out=None, axis=-1):
    """
    Create a 1-d integer array of length n, all zero except for 1 at index k.
    """
    k = np.asarray(k)
    if k.dtype.char not in np.typecodes['AllInteger']:
        raise ValueError('k must be an integer scalar or array.')
    try:
        n = operator.index(n)
    except TypeError:
        raise ValueError(f'n must be an integer; got {n!r}')
    k_shape = k.shape
    adjusted_axis = axis
    if adjusted_axis < 0:
        adjusted_axis += 1 + len(k_shape)
    if adjusted_axis < 0 or adjusted_axis > 1 + len(k_shape):
        raise AxisError(f'invalid axis {axis}')
    out_shape = (k_shape[:adjusted_axis] + (n,)
                 + k_shape[adjusted_axis:])
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(f'out.shape must be {out_shape}; '
                             f'got {out.shape}.')
    else:
        out = np.empty(out_shape, dtype=k.dtype)
    return _one_hot(k, out=out, axis=axis)


one_hot.gufunc = _one_hot
