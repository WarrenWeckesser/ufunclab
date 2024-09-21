"""
Wrappers of gufuncs that provide a nicer API.
"""

import operator
import numpy as np
try:
    from numpy.exceptions import AxisError
except ImportError:
    from numpy import AxisError
from ufunclab._bincount import bincount as _bincount, bincountw as _bincountw
from ufunclab._convert_to_base import convert_to_base as _convert_to_base
from ufunclab._nextn import (nextn_less as _nextn_less,
                             nextn_greater as _nextn_greater)
from ufunclab._one_hot import one_hot as _one_hot


def _normalize_axis(ndim, axis):
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        raise AxisError(msg)
    if axis < 0:
        axis = axis + ndim
    return axis


def _shape_remove_axis(shape, axis):
    ndim = len(shape)
    axis = _normalize_axis(ndim, axis)
    return shape[:axis] + shape[axis + 1:]


def _shape_insert_axis(shape, axis, n):
    ndim = len(shape)
    axis = _normalize_axis(ndim + 1, axis)
    return shape[:axis] + (n,) + shape[axis:]


def bincount(x, m=None, weights=None, out=None, axis=-1):
    """
    Count the number of occurrences of the positive integers in the 1-d
    array `x` that are less than `m`, or accumulate values from `weights`
    based in indices found in `x`.

    If `m` is not given, the default value is `max(np.max(x) + 1, 0)`.

    If `x` or `weights` are n-dimensional arrays, `axis` selects which
    axis of the arrays the operation is applied to.

    Notes
    -----
    This function is a Python wrapper of two gufuncs, one with shape
    signature `(n)->(m)` (no weights) and one with shape signature
    `(n),(n)->(m)`.

    Examples
    --------
    >>> import numpy as np
    >>> from ufunclab import bincount

    Create an array to work with.  `x` is an array with shape `(3, 12)`.
    The first set of examples do not use the `weights` parameter.

    >>> rng = np.random.default_rng(121263137472525314065)
    >>> x = rng.integers(0, 8, size=(3, 12))
    >>> x
    array([[7, 0, 5, 0, 2, 7, 7, 3, 0, 3, 4, 5],
           [2, 6, 7, 1, 3, 0, 6, 1, 2, 0, 0, 6],
           [0, 6, 1, 5, 2, 1, 4, 2, 6, 4, 2, 6]])

    By default, `bincount` operates along the last axis.  The default
    value of `m` is one more than maximum value in `x`, so in this case
    the output length of the counts will be 8.  That is, the output
    array will have shape `(3, 8)`.

    >>> bincount(x)
    array([[3, 0, 1, 2, 1, 2, 0, 3],
           [3, 2, 2, 1, 0, 0, 3, 1],
           [1, 2, 3, 0, 2, 1, 3, 0]], dtype=uint64)

    If we given a value for `m` that is larger than 8, the final values
    will be 0.

    >>> bincount(x, 10)
    array([[3, 0, 1, 2, 1, 2, 0, 3, 0, 0],
           [3, 2, 2, 1, 0, 0, 3, 1, 0, 0],
           [1, 2, 3, 0, 2, 1, 3, 0, 0, 0]], dtype=uint64)

    If the given value of `m` is smaller than `np.max(x) + 1`, the values
    greater than or equal to `m` are ignored.

    >>> bincount(x, 4)
    array([[3, 0, 1, 2],
           [3, 2, 2, 1],
           [1, 2, 3, 0]], dtype=uint64)

    The `axis` parameter selects the axis of `x` along which `bincount`
    is applied.  In the following example, since `x` has shape `(3, 12)`,
    the output has shape `(8, 12)` when `axis=0` is given.

    >>> bincount(x, axis=0)
    array([[1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
           [1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2],
           [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=uint64)

    Some examples with weights.

    >>> x = np.array([3, 4, 5, 1, 1, 0, 4])
    >>> w = np.array([1.0, 0.25, 1.5, 0.5, 0.75, 1.0, 1.5])
    >>> bincount(x, weights=w)
    array([1.  , 1.25, 0.  , 1.  , 1.75, 1.5 ])

    >>> x = np.array([[1, 0, 2, 2],
    ...               [0, 0, 0, 2]])
    >>> w = np.array([0.25, 0.75, 0.75, 0.5])
    >>> bincount(x, weights=w)
    array([[0.75, 0.25, 1.25],
           [1.75, 0.  , 0.5 ]])

    """
    x = np.asarray(x)
    if x.dtype.char not in np.typecodes['AllInteger']:
        raise ValueError('x must be an integer array.')
    if m is None:
        m = max(np.max(x) + 1, 0)
    else:
        try:
            m = operator.index(m)
        except TypeError:
            raise ValueError(f'm must be a nonnegative integer; got {m!r}')
        if m < 0:
            raise ValueError(f'm must be a nonnegative integer; got {m!r}')

    if weights is not None:
        weights = np.asarray(weights)
        x_bc_shape = _shape_remove_axis(x.shape, axis)
        w_bc_shape = _shape_remove_axis(weights.shape, axis)
        bc_shape = np.broadcast_shapes(x_bc_shape, w_bc_shape)
        out_shape = _shape_insert_axis(bc_shape, axis, m)

        if out is not None:
            if out.shape != out_shape:
                msg = ('For the given inputs, out.shape must be '
                       f'{out_shape}; got {out.shape}.')
                raise ValueError(msg)
        else:
            out = np.zeros(out_shape, dtype=weights.dtype)

        return _bincountw(x, weights, out=out, axes=[axis, axis, axis])
    else:
        x_shape = x.shape
        adjusted_axis = _normalize_axis(x.ndim, axis)
        out_shape = list(x_shape)
        out_shape[adjusted_axis] = m
        out_shape = tuple(out_shape)
        if out is not None:
            if out.shape != out_shape:
                raise ValueError(f'out.shape must be {out_shape}; '
                                 f'got {out.shape}.')
        else:
            out = np.zeros(out_shape, dtype=np.intp)

        return _bincount(x, out=out, axes=[axis, axis])


bincount._bincount = _bincount
bincount._bincountw = _bincountw


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
