from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource

MULTIVARIATE_LOGBETA_DOCSTRING = """\
multivariate_logebta(x, /, ...)

Compute the logarithm of the multivariate beta function B(x).

For the core calculation, ``x`` is a 1-d array.
"""

multivariate_logbeta_core = UFuncSource(
    funcname='multivariate_logbeta_core',
    typesignatures=['f->f', 'd->d', 'g->g'],
)

multivariate_logbeta_ufunc = UFunc(
    name="multivariate_logbeta",
    header="multivariate_logbeta_gufunc.h",
    docstring=MULTIVARIATE_LOGBETA_DOCSTRING,
    signature='(n)->()',
    sources=[multivariate_logbeta_core],
)


extmod = UFuncExtMod(
    module='_multivariate_logbeta',
    docstring="This extension module defines the gufunc multivariate_logbeta.",
    ufuncs=[multivariate_logbeta_ufunc],
)
