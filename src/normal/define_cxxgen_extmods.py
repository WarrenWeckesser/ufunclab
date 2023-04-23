
from ufunc_config_types import ExtMod, Func


normal_cdf_docstring = """\
cdf(x, /, ...)

CDF of the standard normal distribution.

"""

normal_logcdf_docstring = """\
logcdf(x, /, ...)

Logarithm of the CDF of the standard normal distribution.

"""

normal_sf_docstring = """\
sf(x, /, ...)

Survival function (complementary CDF) of the standard normal distribution.

"""

normal_logsf_docstring = """\
logsf(x, /, ...)

Logarithm of the survival function of the standard normal distribution.

"""

erfcx_docstring = """\
erfcx(x, /, ...)

Scaled version of erfc(x).
"""

normal_funcs = [
    Func(cxxname='normal_cdf',
         ufuncname='cdf',
         types=['f->f', 'd->d', 'g->g'],
         docstring=normal_cdf_docstring),
    Func(cxxname='normal_logcdf',
         ufuncname='logcdf',
         types=['f->f', 'd->d', 'g->g'],
         docstring=normal_cdf_docstring),
    Func(cxxname='normal_sf',
         ufuncname='sf',
         types=['f->f', 'd->d', 'g->g'],
         docstring=normal_sf_docstring),
    Func(cxxname='normal_logsf',
         ufuncname='logsf',
         types=['f->f', 'd->d', 'g->g'],
         docstring=normal_sf_docstring),
]

erfcx_funcs = [
    Func(cxxname='erfcx',
         ufuncname='_erfcx',
         types=['f->f', 'd->d', 'g->g'],
         docstring=erfcx_docstring),
]

extmods = [ExtMod(modulename='normal',
                  funcs={'normal.h': normal_funcs,
                         'erfcx_funcs.h': erfcx_funcs})]
