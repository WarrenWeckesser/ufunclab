
from ufunc_config_types import ExtMod, Func


normal_cdf_docstring = """\
normal_cdf(x, /, ...)

CDF of the standard normal distribution.

"""

normal_logcdf_docstring = """\
normal_logcdf(x, /, ...)

Logarithm of the CDF of the standard normal distribution.

"""

normal_sf_docstring = """\
normal_sf(x, /, ...)

Survival function (complementary CDF) of the standard normal distribution.

"""

normal_logsf_docstring = """\
normal_logsf(x, /, ...)

Logarithm of the survival function of the standard normal distribution.

"""

erfcx_docstring = """\
erfcx(x, /, ...)

Scaled version of erfc(x).
"""

normal_funcs = [
    Func(cxxname='normal_cdf',
         ufuncname='normal_cdf',
         types=['f->f', 'd->d', 'g->g'],
         docstring=normal_cdf_docstring),
    Func(cxxname='normal_logcdf',
         ufuncname='normal_logcdf',
         types=['f->f', 'd->d'],
         docstring=normal_cdf_docstring),
    Func(cxxname='normal_sf',
         ufuncname='normal_sf',
         types=['f->f', 'd->d', 'g->g'],
         docstring=normal_sf_docstring),
    Func(cxxname='normal_logsf',
         ufuncname='normal_logsf',
         types=['f->f', 'd->d'],
         docstring=normal_sf_docstring),
]

erfcx_funcs = [
    Func(cxxname='erfcx',
         ufuncname='erfcx',
         types=['f->f', 'd->d', 'g->g'],
         docstring=erfcx_docstring),
]

extmods = [ExtMod(modulename='_normal',
                  funcs={'normal.h': normal_funcs,
                         'erfcx_funcs.h': erfcx_funcs})]
