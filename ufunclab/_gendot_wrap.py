
import numpy as np
from ._gendot import _gendot


longdouble_code, longcomplex_code = (
    (12, 15) if np.dtype('g') == np.dtype('d') else (13, 16)
)

reverse_typedict = {
    np.bool_: 0,
    np.int8: 1,
    np.uint8: 2,
    np.int16: 3,
    np.uint16: 4,
    np.int32: 5,
    np.uint32: 6,
    np.int64: 7,
    np.uint64: 8,
    np.longlong: 9,
    np.ulonglong: 10,
    np.float16: 23,
    np.float32: 11,
    np.float64: 12,
    np.longdouble: longdouble_code,
    np.complex64: 14,
    np.complex128: 15,
    np.longcomplex: longcomplex_code,
    np.object_: 17,
    np.bytes_: 18,
    np.str_: 19,
    np.void: 20,
    np.datetime64: 21,
    np.timedelta64: 22,
}


def _check_ufunc1(func, argname):
    if not (isinstance(func, np.ufunc)
            and func.signature is None
            and func.nin == 2 and func.nout == 1):
        raise ValueError(f"{argname} must be an element-wise ufunc with 2 "
                         "inputs and 1 output.")
    if len(func.types) > 256:
        raise ValueError(f"{argname} has {len(func.types)} ufunc inner loops! "
                         "(Sorry, can't handle that many; max is 256.)")


def _check_n_to_1(gufunc):
    if not isinstance(gufunc, np.ufunc):
        return False
    if gufunc.signature is None:
        return False
    if gufunc.nin != 1 or gufunc.nout != 1:
        return False
    signature = gufunc.signature.replace(' ', '')
    sigin, sigout = signature.split('->')
    if sigout != '()':
        return False
    return sigin[1:-1].isidentifier()


def _check_ufunc2(func, argname):
    ok = (isinstance(func, np.ufunc) and
          ((func.signature is None and func.nin == 2 and func.nout == 1) or
           _check_n_to_1(func)))
    if not ok:
        raise ValueError(f"{argname} must be an element-wise ufunc with 2 "
                         "inputs and 1 output, or a gufunc with signature "
                         "'(i)->()'.")
    if len(func.types) > 256:
        raise ValueError(f"{argname} has {len(func.types)} ufunc inner loops! "
                         "(Sorry, can't handle that many; max is 256.)")


def gendot(prodfunc, sumfunc, name=None, doc=None):
    """
    Compose two ufuncs to create a gufunc that generalizes the dot product.

    Parameters
    ----------
    prodfunc : ufunc
        Must be a scalar ufunc (i.e. `ufunc.signature` is None), with two
        inputs and one output.
    sumfunc : ufunc
        Must be either a scalar ufunc (i.e. `ufunc.signature` is None),
        with two inputs and one output, or a gufunc with signature
        `(i)->()`.
    name : str, optional
        Name to assign to the `__name__` attribute of the gufunc.
    doc : str, optional
        Docstring for the gufunc.

    Returns
    -------
    dotfunc : gufunc
        A gufunc with signature `(i),(i)->()` that is computed by applying
        prodfunc element-wise to the inputs, and then reducing the result
        with sumfunc.

    Examples
    --------
    >>> import numpy as np
    >>> from ufunclab import gendot

    Create a dot-like composition of `logical_and` and `logical_or`:

    >>> logical_dot = gendot(np.logical_and, np.logical_or)

    Take a look at a couple of the gufunc attributes:

    >>> logical_dot.signature
    '(i),(i)->()'
    >>> print(logical_dot.types)
    ['??->?', 'bb->?', 'BB->?', 'hh->?', 'HH->?', 'ii->?', 'II->?', 'll->?',
    'LL->?', 'qq->?', 'QQ->?', 'ee->?', 'ff->?', 'dd->?', 'gg->?', 'FF->?',
    'DD->?', 'GG->?']

    Apply `logical_dot` to the arrays `p` and `q`:

    >>> p = np.array([[False, True,  True, True],
    ...               [False, True, False, True]])
    >>> q = np.array([True, False, True, False])
    >>> logical_dot(p, q)
    array([ True, False])

    Create a dot-like composition of `minimum` and `maximum`:

    >>> minmaxdot = gendot(np.minimum, np.maximum)
    >>> a = np.array([1.0, 2.5, 0.3, 1.9, 3.0, 1.8])
    >>> b = np.array([0.5, 1.1, 0.9, 2.1, 0.3, 3.0])
    >>> minmaxdot(a, b)
    1.9

    Here's the same calculation with `np.minimum` and `np.maximum`:

    >>> np.maximum.reduce(np.minimum(a, b))
    1.9

    """
    _check_ufunc1(prodfunc, 'prodfunc')
    _check_ufunc2(sumfunc, 'sumfunc')
    if name is None:
        name = '_'.join(['gendot', prodfunc.__name__, sumfunc.__name__])
    if doc is None:
        # FIXME: Should allow None to be handled in the C code.
        #        OR, provide a simple default docstring?
        doc = ""
    dot_types = []
    loop_indices = []
    for i, prod_type in enumerate(prodfunc.types):
        if 'O' in prod_type:
            # Skip object types.
            continue
        prod_outtype = prod_type.replace(' ', '')[-1]
        for j, sum_type in enumerate(sumfunc.types):
            sum_intype = sum_type.replace(' ', '').split('->')[0]
            if sum_intype == sumfunc.nin*prod_outtype:
                loop_indices.append((i, j))
                dot_types.append((prod_type, sum_type))
    typechars = [prod_type[:2] + sum_type[-1]
                 for prod_type, sum_type in dot_types]
    itemsizes = np.array([np.dtype(c[-1]).itemsize for c in typechars],
                         dtype=np.intp)  # Actually, np.uint8 would be OK.
    typecodes = np.array([[reverse_typedict[np.dtype(c).type] for c in chars]
                          for chars in typechars],
                         dtype=np.uint8)
    loop_indices = np.array(loop_indices, dtype=np.uint8)
    sumfunc_identity_array = np.zeros((len(typechars), np.dtype('G').itemsize),
                                      dtype=np.uint8)
    sumfunc_has_identity = False
    if sumfunc.identity is not None:
        sumfunc_has_identity = True
        for k, tc in enumerate(typechars):
            identity = np.dtype(tc[-1]).type(sumfunc.identity)
            size = identity.itemsize
            identity_bytes = identity.view(np.dtype(('B', (size,))))
            sumfunc_identity_array[k, :size] = identity_bytes

    return _gendot(name, doc, prodfunc,
                   sumfunc, sumfunc_has_identity, sumfunc_identity_array,
                   loop_indices, typecodes, itemsizes)
