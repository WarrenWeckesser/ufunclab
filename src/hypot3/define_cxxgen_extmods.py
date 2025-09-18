
from ufunc_config_types import ExtMod, Func


hypot3_docstring = """\
hypot3(x, y, z, /, ...)

Compute sqrt(x*x + y*y + z*z).

This is a wrapper of the C++ function std::hypot(x, y, z).

"""

hypot3_funcs = [
    Func(cxxname='hypot3',
         ufuncname='hypot3',
         types=['fff->f', 'ddd->d', 'ggg->g'],
         docstring=hypot3_docstring),
]

extmods = [ExtMod(modulename='_hypot3',
                  funcs={'hypot3.h': hypot3_funcs})]
