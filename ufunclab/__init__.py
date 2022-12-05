"""
NumPy ufuncs and utilities.
"""

from ._logfact import logfactorial
from ._issnan import issnan
from ._abs_squared import abs_squared
from ._cabssq import cabssq
from ._expint1 import expint1, logexpint1
from ._logistic import logistic, logistic_deriv, log_logistic, swish
from ._hyperbolic_ramp import hyperbolic_ramp
from ._normal import normal_cdf, normal_logcdf, normal_sf, normal_logsf, erfcx
from ._yeo_johnson import yeo_johnson, inv_yeo_johnson
from ._cross import cross3, cross2
from ._first import first, argfirst, _LT, _LE, _EQ, _NE, _GT, _GE
from ._searchsorted import searchsortedl, searchsortedr
from ._peaktopeak import peaktopeak
from ._minmax import argmin, argmax, minmax, argminmax, min_argmin, max_argmax
from ._means import gmean, hmean
from ._meanvar import meanvar
from ._corr import pearson_corr
from ._mad import mad, rmad, gini
from ._vnorm import vnorm, rms, vdot
from ._tri_area import tri_area
from ._backlash import backlash
from ._fillnan1d import fillnan1d
from ._semivar import semivar_exponential, semivar_linear, semivar_spherical
from ._deadzone import deadzone
from ._trapezoid_pulse import trapezoid_pulse
from ._hysteresis_relay import hysteresis_relay
from ._all_same import all_same
from ._sosfilter import sosfilter, sosfilter_ic, sosfilter_ic_contig

from ._step import step, linearstep, smoothstep3, invsmoothstep3, smoothstep5
from ._next import next_greater, next_less

from ._gendot_wrap import gendot
from ._ufunc_inspector import ufunc_inspector

import numpy as _np


__version__ = "0.0.7.dev7"


class op:
    """
    Allowed values for the `op` argument of `argfirst`.
    """
    LT = _np.int8(_LT)
    LE = _np.int8(_LE)
    EQ = _np.int8(_EQ)
    NE = _np.int8(_NE)
    GT = _np.int8(_GT)
    GE = _np.int8(_GE)
