
from ufunc_config_types import ExtMod, Func


next_greater_docstring = """\
next_greater(x, /, ...)

Return the smallest floating point value with the same type as x
that is greater than x.
"""

next_less_docstring = """\
next_less(x, /, ...)

Return the largest floating point value with the same type as x
that is less than x.
"""

next_funcs = [
    Func(cxxname='NextFunctions::next_greater',
         ufuncname='next_greater',
         types=['f->f', 'd->d'], # 'g->g'],
         docstring=next_greater_docstring),
    Func(cxxname='NextFunctions::next_less',
         ufuncname='next_less',
         types=['f->f', 'd->d'], #, 'g->g'],
         docstring=next_less_docstring),
]

extmods = [ExtMod(modulename='_next',
                  funcs={'next.h': next_funcs})]
