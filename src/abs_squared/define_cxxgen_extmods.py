
from ufunc_config_types import ExtMod, Func


normal_cdf_docstring = """\
abs_squared(z, /, ...)

Squared absolute value.

"""

funcs = [
    Func(cxxname='abs_squared',
         ufuncname='abs_squared',
         types=['f->f', 'd->d', 'g->g', 'F->f', 'D->d', 'G->g'],
         docstring=normal_cdf_docstring),
]


extmods = [ExtMod(modulename='_abs_squared',
                  funcs={'abs_squared.h': funcs})]
