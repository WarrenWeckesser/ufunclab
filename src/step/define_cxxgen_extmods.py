
from ufunc_config_types import ExtMod, Func


step_docstring = """\
step(x, a, flow, fa, fhigh, /, ...)

The function `step(x, a, flow, fa, fhigh)` returns `flow` for
`x < a`, `fhigh` for `x > a`, and `fa` for `x = a`.
"""

linearstep_docstring = """\
linearstep(x, a, b, fa, fb, /, ...)

The function `linearstep(x, a, b, fa, fb)` returns `fa` for
`x <= a`, `fb` for `x >= b`, and uses linear interpolation
from `fa` to `fb` in the interval `a < x < b`.
"""

smoothstep3_docstring = """\
smoothstep3(x, a, b, fa, fb, /, ...)

The function `smoothstep3(x, a, b, fa, fb)` returns `fa` for
`x <= a`, `fb` for `x >= b`, and uses a cubic polynomial in
the interval `a < x < b` to smoothly transition from `fa` to `fb`.
"""

invsmoothstep3_docstring = """\
invsmoothstep3(y, a, b, fa, fb, /, ...)

Inverse of `smoothstep3(x, a, b, fa, fb)`.

The function `invsmoothstep3(y, a, b, fa, fb)` returns `nan` for
values outside the range of `smoothstep3`.

The inverse is multivalued at `y == fa` and `y == fb`, so it would not be
inappropriate for the function to return `nan` at these points.  Instead,
the function returns `a` and `b`, respectively.
"""

smoothstep5_docstring = """\
smoothstep5(x, a, b, fa, fb, /, ...)

The function `smoothstep5(x, a, b, fa, fb)` returns `fa` for
`x <= a`, `fb` for `x >= b`, and uses a degree 5 polynomial in
the interval `a < x < b` to smoothly transition from `fa` to `fb`.
"""

step_funcs = [
    Func(cxxname='StepFunctions::step',
         ufuncname='step',
         types=['fffff->f', 'ddddd->d', 'ggggg->g'],
         docstring=step_docstring),
    Func(cxxname='StepFunctions::linearstep',
         ufuncname='linearstep',
         types=['fffff->f', 'ddddd->d', 'ggggg->g'],
         docstring=linearstep_docstring),
    Func(cxxname='StepFunctions::smoothstep3',
         ufuncname='smoothstep3',
         types=['fffff->f', 'ddddd->d', 'ggggg->g'],
         docstring=smoothstep3_docstring),
    Func(cxxname='StepFunctions::invsmoothstep3',
         ufuncname='invsmoothstep3',
         types=['fffff->f', 'ddddd->d', 'ggggg->g'],
         docstring=invsmoothstep3_docstring),
    Func(cxxname='StepFunctions::smoothstep5',
         ufuncname='smoothstep5',
         types=['fffff->f', 'ddddd->d', 'ggggg->g'],
         docstring=smoothstep5_docstring),
]

extmods = [ExtMod(modulename='_step',
                  funcs={'step.h': step_funcs})]
