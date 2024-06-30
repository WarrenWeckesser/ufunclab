
from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


FIRST_DOCSTRING = """\
first(x, op, target, otherwise, /, ...)

Find the first value that matches the given comparison.
"""

ARGFIRST_DOCSTRING = """\
argfirst(x, op, target, /, ...)

Find the index in `x` of the first value where `x op target`
is true, where `op` is one of the basic comparison operators.

Parameters
----------
x : array_like, size (..., n)
    Array to be searched.
op : int, one of {0, 1, 2, 3, 4, 5}
    Defines the comparison operation to be used. Attributes of
    the class `ufunclab.op` may be used as symbolic names of
    the operators.

        Comparison  op  ufunclab.op attribute
        ----------  --  ---------------------
            <        0    ufunclab.op.LT
            <=       1    ufunclab.op.LE
            ==       2    ufunclab.op.EQ
            !=       3    ufunclab.op.NE
            >        4    ufunclab.op.GT
            >=       5    ufunclab.op.GE

    An error is not raised if `op` is not in {0, 1, 2, 3, 4, 5},
    but the return value will be -1.

target : value to be searched for
    For best efficiency, this value should have the same
    type as the elements of `x`.

Returns
-------
index : integer
    The index of the first element where the comparison
    is true.  If no value is found, -1 is returned.

Examples
--------
>>> import numpy as np
>>> from ufunclab import argfirst, op

Find the index of the first occurrence of 0 in `x`:

>>> x = np.array([10, 35, 19, 0, -1, 24, 0])
>>> argfirst(x, op.EQ, 0)
3

Find the index of the first nonzero value in `a`:

>>> a = np.array([0, 0, 0, 0, 0, -0.5, 0, 1, 0.1])
>>> argfirst(a, op.NE, 0.0)
5

`argfirst` is a gufunc, so it can handle higher-dimensional
array arguments, and among its gufunc-related parameters is
`axis`.  By default, the gufunc operates along the last axis.
For example, here we find the location of the first nonzero
element in each row of `b`:

>>> b = np.array([[0, 8, 0, 0], [0, 0, 0, 0], [0, 0, 9, 2]],
...              dtype=np.uint8)
>>> b
array([[0, 8, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 9, 2]])
>>> argfirst(b, op.NE, np.uint8(0))
array([ 1, -1,  2])

If we give the argument `axis=0`, we tell `argfirst` to
operate along the first axis, which in this case is the
columns:

>>> argfirst(b, op.NE, np.uint8(0), axis=0)
array([-1,  0,  2,  2])

"""

first_src = UFuncSource(
    funcname='first_core_calc',
    typesignatures=[
        'bbbb->b',
        'BbBB->B',
        'hbhh->h',
        'HbHH->H',
        'ibii->i',
        'IbII->I',
        'lbll->l',
        'LbLL->L',
        'fbff->f',
        'dbdd->d',
        'gbgg->g',
    ]
)

first_src_object = UFuncSource(
    funcname='first_core_calc_object',
    typesignatures=['ObOO->O']
)

first = UFunc(
    name='first',
    header='first_gufunc.h',
    docstring=FIRST_DOCSTRING,
    signature='(n),(),(),()->()',
    sources=[first_src, first_src_object],
)

argfirst_src = UFuncSource(
    funcname='argfirst_core_calc',
    typesignatures=[
        'bbb->l',
        'BbB->l',
        'hbh->l',
        'HbH->l',
        'ibi->l',
        'IbI->l',
        'lbl->l',
        'LbL->l',
        'fbf->l',
        'dbd->l',
        'gbg->l',
    ]
)

argfirst_src_object = UFuncSource(
    funcname='argfirst_core_calc_object',
    typesignatures=['ObO->l']
)

argfirst = UFunc(
    name='argfirst',
    header='first_gufunc.h',
    docstring=ARGFIRST_DOCSTRING,
    signature='(n),(),()->()',
    sources=[argfirst_src, argfirst_src_object],
)

extmod = UFuncExtMod(
    module='_first',
    docstring=("This extension module defines the gufuncs 'first' and "
               "'argfirst'."),
    ufuncs=[first, argfirst],
    # The call `status = add_comparison_constants(module);` will be
    # added to the end of the module init function.
    extra_module_funcs=['add_comparison_constants']
)
