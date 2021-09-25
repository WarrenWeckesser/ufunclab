"""
NumPy ufuncs and utilities.
"""

from ._logfact import logfactorial
from ._cross3 import cross3
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
