
from ufunc_config_types import ExtMod, Func


abs_squared_docstring = """\
abs_squared(z, /, ...)

Squared absolute value.

"""

funcs = [
    Func(cxxname='abs_squared',
         ufuncname='abs_squared',
         types=['f->f', 'd->d', 'g->g', 'F->f', 'D->d', 'G->g'],
         docstring=abs_squared_docstring),
]


extmods = [ExtMod(modulename='_abs_squared',
                  funcs={'abs_squared.h': funcs})]
