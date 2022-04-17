
from ufunc_config_types import ExtMod, Func


semivar_exponential_docstring = """\
semivar_exponential(h, nugget, sill, rng, /, ...)

Exonential semivariogram.

    γ(h) = N + (S - N)*exp(1 - 3*h/R)

where N is the nugget, S is the sill and R is the range.

Parameters
----------
h : array_like
    The distance from a point.
nugget : array_like
    The base local variance.
sill : array_like
    The far-field variance.
rng : array_like
    The distance at which the variance becomes the constant `sill`
    and the data are no longer correlated.  For the exponential
    variogram, this is not a sharp boundary. `rng` controls the
    exponential decay towards the sill value.

Returns
-------
v : ndarray or scalar
    The variance at distance `h`.

"""

semivar_linear_docstring = """\
semivar_linear(h, nugget, sill, rng, /, ...)

Linear semivariogram.

    γ(h) = N + ((S - N)/R)*h   for h < R
    γ(h) = S                   for h >= R

where N is the nugget, S is the sill and R is the range.


Parameters
----------
h : array_like
    The distance from a point.
nugget : array_like
    The base local variance.
sill : array_like
    The far-field variance.
rng : array_like
    The distance at which the variance becomes the constant `sill`
    and the data are no longer correlated.

Returns
-------
v : ndarray or scalar
    The variance at distance `h`.

"""

semivar_spherical_docstring = """\
semivar_spherical(h, nugget, sill, rng, /, ...)

Spherical semivariogram.

    γ(h) = N + (S - N)*(ρ/2)*(3 - ρ**2)  for h < R
    γ(h) = S                             for h >= R

where ρ = h/range, N is the nugget, S is the sill and R is the range.

Parameters
----------
h : array_like
    The distance from a point.
nugget : array_like
    The base local variance.
sill : array_like
    The far-field variance.
rng : array_like
    The distance at which the variance becomes the constant `sill`
    and the data are no longer correlated.

Returns
-------
v : ndarray or scalar
    The variance at distance `h`.

"""


semivar_funcs = [
    Func(cxxname='semivar_exponential',
         ufuncname='semivar_exponential',
         types=['ffff->f', 'dddd->d', 'gggg->g'],
         docstring=semivar_exponential_docstring),
    Func(cxxname='semivar_linear',
         ufuncname='semivar_linear',
         types=['ffff->f', 'dddd->d', 'gggg->g'],
         docstring=semivar_linear_docstring),
    Func(cxxname='semivar_spherical',
         ufuncname='semivar_spherical',
         types=['ffff->f', 'dddd->d', 'gggg->g'],
         docstring=semivar_spherical_docstring),
]

extmods = [ExtMod(modulename='_semivar',
                  funcs={'semivar.h': semivar_funcs})]
