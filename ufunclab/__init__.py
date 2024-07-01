"""
NumPy ufuncs and utilities.
"""
import importlib as _imp
# To allow moving 'erfcx' from the 'normal' submodule to the
# top-level namespace here in __init__.py, 'normal' is not
# lazy-loaded.
from . import normal
from .normal import erfcx
del normal.erfcx

# To allow `op` to be created here, ._first is not lazy-loaded.
# It is imported here to have access to the constants _LT, _LE, etc.
from ._first import first, argfirst, _LT, _LE, _EQ, _NE, _GT, _GE
import numpy as _np


# The keys of this dict are in modules that are lazy-loaded.
_name_to_module = {
    'logfactorial': '._logfact',
    'loggamma1p': '._loggamma1p',
    'issnan': '._issnan',
    'abs_squared': '._abs_squared',
    'cabssq': '._cabssq',
    'log1p_theorem4': '._log1p',
    'log1p_doubledouble': '._log1p',
    'debye1': '._debye1',
    'expint1': '._expint1',
    'logexpint1': '._expint1',
    'pow1pm1': '._pow1pm1',
    'logistic': '._logistic',
    'logistic_deriv': '._logistic',
    'log_logistic': '._logistic',
    'swish': '._logistic',
    'hyperbolic_ramp': '._ramp',
    'exponential_ramp': '._ramp',
    'yeo_johnson': '._yeo_johnson',
    'inv_yeo_johnson': '._yeo_johnson',
    'cross3': '._cross',
    'cross2': '._cross',
    'searchsortedl': '._searchsorted',
    'searchsortedr': '._searchsorted',
    'peaktopeak': '._peaktopeak',
    'argmin': '._minmax',
    'argmax': '._minmax',
    'minmax': '._minmax',
    'argminmax': '._minmax',
    'min_argmin': '._minmax',
    'max_argmax': '._minmax',
    'multivariate_logbeta': '._multivariate_logbeta',
    'gmean': '._means',
    'hmean': '._means',
    'meanvar': '._meanvar',
    'pearson_corr': '._corr',
    'wjaccard': '._wjaccard',
    'mad': '._mad',
    'rmad': '._mad',
    'gini': '._mad',
    'vnorm': '._vnorm',
    'rms': '._vnorm',
    'vdot': '._vnorm',
    'tri_area': '._tri_area',
    'tri_area_indexed': '._tri_area',
    'backlash': '._backlash',
    'fillnan1d': '._fillnan1d',
    'linear_interp1d': '._linear_interp1d',
    'deadzone': '._deadzone',
    'trapezoid_pulse': '._trapezoid_pulse',
    'hysteresis_relay': '._hysteresis_relay',
    'all_same': '._all_same',
    'sosfilter': '._sosfilter',
    'sosfilter_ic': '._sosfilter',
    'sosfilter_ic_contig': '._sosfilter',
    'step': '._step',
    'linearstep': '._step',
    'smoothstep3': '._step',
    'invsmoothstep3': '._step',
    'smoothstep5': '._step',
    'next_greater': '._next',
    'next_less': '._next',
    'gendot': '._gendot_wrap',
    'ufunc_inspector': '._ufunc_inspector',
    '__version__': '._version',
}


def __getattr__(name):
    try:
        module_name = _name_to_module[name]
    except Exception:
        raise AttributeError
    module = _imp.import_module(module_name, __name__)
    return getattr(module, name)


class op:
    """
    Allowed values for the `op` argument of `first` and `argfirst`.
    """
    LT = _np.int8(_LT)
    LE = _np.int8(_LE)
    EQ = _np.int8(_EQ)
    NE = _np.int8(_NE)
    GT = _np.int8(_GT)
    GE = _np.int8(_GE)


del _np, _LT, _LE, _EQ, _NE, _GT, _GE

__all__ = sorted(list(_name_to_module.keys()) +
                 ['first', 'argfirst', 'op'])


def __dir__():
    return __all__
