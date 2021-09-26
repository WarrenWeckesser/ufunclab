"""
NumPy ufuncs and utilities.
"""

from ._logfact import logfactorial
from ._cross3 import cross3
from ._findfirst import findfirst
from ._peaktopeak import peaktopeak
from ._minmax import minmax, argminmax, min_argmin, max_argmax
from ._means import gmean, hmean
from ._mad import mad, mad1, rmad, rmad1
from ._vnorm import vnorm
from ._backlash import backlash
from ._deadzone import deadzone
from ._hysteresis_relay import hysteresis_relay
from ._all_same import all_same
from ._ufunc_inspector import ufunc_inspector

import numpy as _np


# XXX Maybe use enum.IntEnum for `op`?

class op:
    """
    Allowed values for the `op` argument of `findfirst`.
    """
    LT = _np.int8(0)
    LE = _np.int8(1)
    EQ = _np.int8(2)
    NE = _np.int8(3)
    GT = _np.int8(4)
    GE = _np.int8(5)
