from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource

ORDERED_DOCSTRING = """\
ordered(x, op, /, ...)

Test that the input `x` is ordered. `op` selects the ordering
to test for, as follows:

    +----------+-------------------------+
    |   op     | Order                   |
    +----------+-------------------------+
    |  op.LT   | strictly increasing     |
    |  op.LE   | increasing              |
    |  op.EQ   | constant                |
    |  op.GE   | decreasing              |
    |  op.GT   | strictly decreasing     |
    +----------+-------------------------+

`ordered` is implemented for `x` having an integer type, a single or
double precision floating point type, or an object type.

Examples
--------
>>> import numpy as np
>>> from ufunclab import ordered, op

>>> x = np.array([1, 3, 3.5, 3.5, 10, 11, 11])
>>> ordered(x, op.LT)
np.False_

>>> ordered(x, op.LE)
np.True_

>>> ordered(x, [op.LT, op.LE, op.EQ, op.GE, op.GT])
array([False,  True, False, False, False])

>>> a = np.array([[3, 4, 10, 12],
...               [1, 1, 13, 15]])
>>> ordered(a, op.LT)
array([ True, False])

ordered(a, op.LE)
array([ True,  True])

`ordered` accepts object arrays.

>>> from fractions import Fraction
>>> y = np.array([10, Fraction(355, 113), Fraction(5, 2), Fraction(1, 3)])
>>> ordered(y, op.GT)
np.True_
"""

ordered_core = UFuncSource(
    funcname='ordered_core',
    typesignatures=['bb->?','Bb->?', 'hb->?', 'Hb->?', 'ib->?', 'Ib->?',
                    'lb->?', 'Lb->?', 'fb->?', 'db->?', 'gb->?'],
)

ordered_core_object = UFuncSource(
    funcname='ordered_core_object',
    typesignatures=['Ob->?'],
)

ufunc = UFunc(
    name="ordered",
    header='ordered_gufunc.h',
    docstring=ORDERED_DOCSTRING,
    signature='(n),()->()',
    sources=[ordered_core, ordered_core_object],
)

extmod = UFuncExtMod(
    module='_ordered',
    docstring="This extension module defines the gufunc 'ordered'.",
    ufuncs=[ufunc],
)
