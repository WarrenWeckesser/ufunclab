

from ufunc_config_types import UFuncExtMod, UFunc, UFuncSource


SOSFILTER_DOCSTRING = """\
sosfilter(sos, x, /, ...)

Discrete time linear filter using second order sections.

This function is like `scipy.signal.sosfilt`, but it does
not accept the `zi` parameter that provided initial values
for the internal state of the files.

See `scipy.signal.sosfilt` for more information about creating
and using an SOS filter.

Parameters
----------
sos : array_like, shape (m, 6)
    Array of *second order sections* that represents the linear
    filter.
x : array_like
    Signal to be filtered.

Returns
-------
y : ndarray
    Filtered signal.

"""

SOSFILTER_IC_DOCSTRING = """\
sosfilter_ic(sos, x, zi, /, ...)

Discrete time linear filter using second order sections.

This function is like `scipy.signal.sosfilt`, but in this version,
zi (the initial state of the filter) is *required*.  Also, because
this function is a gufunc, it follows the standard gufunc rules
for broadcasting.  `scipy.signal.sosfilt` broadcasts its `zi`
parameter differently.

See `scipy.signal.sosfilt` for more information about creating
and using an SOS filter.

Parameters
----------
sos : array_like, shape (m, 6)
    Array of *second order sections* that represents the linear
    filter.
x : array_like
    Signal to be filtered.
zi : array_like, shape (m, 2)
    Inital state of the filter.

Returns
-------
y : ndarray
    Filtered signal.

"""

SOSFILTER_IC_CONTIG_DOCSTRING = """\
sosfilter_ic_contig(sos, x, zi, /, ...)

Discrete time linear filter using second order sections.

This function has the same inputs and outputs as `sosfilter_ic`;
see that function for more information.

All input arrays to this version of the function *must* be
C-contiguous.  If they are not, the results will be incorrect and
the program might crash.

"""

sosfilter_core_source = UFuncSource(
    funcname='sosfilter_core',
    typesignatures=['ff->f', 'dd->d', 'gg->g'],
)

sosfilter_gufunc = UFunc(
    name='sosfilter',
    docstring=SOSFILTER_DOCSTRING,
    header='sosfilter_gufunc.h',
    signature='(m,6), (n) -> (n)',
    sources=[sosfilter_core_source],
)

sosfilter_ic_core_source = UFuncSource(
    funcname='sosfilter_ic_core',
    typesignatures=['fff->ff', 'ddd->dd', 'ggg->gg'],
)

sosfilter_ic_gufunc = UFunc(
    name='sosfilter_ic',
    docstring=SOSFILTER_IC_DOCSTRING,
    header='sosfilter_gufunc.h',
    signature='(m,6), (n), (m,2) -> (n), (m,2)',
    sources=[sosfilter_ic_core_source],
)

sosfilter_ic_contig_source = UFuncSource(
    funcname='sosfilter_ic_contig_core',
    typesignatures=['fff->ff', 'ddd->dd', 'ggg->gg'],
)

sosfilter_ic_contig_gufunc = UFunc(
    name='sosfilter_ic_contig',
    docstring=SOSFILTER_IC_CONTIG_DOCSTRING,
    header='sosfilter_gufunc.h',
    signature='(m,6), (n), (m,2) -> (n), (m,2)',
    sources=[sosfilter_ic_contig_source],
)

MODULE_DOCSTRING = """\
SOS filter implementations.

This module defines the gufuncs 'sosfilter', 'sosfilter_ic' and
'sosfilter_ic_contig.
"""

extmod = UFuncExtMod(
    module='_sosfilter',
    docstring=MODULE_DOCSTRING,
    ufuncs=[sosfilter_gufunc, sosfilter_ic_gufunc, sosfilter_ic_contig_gufunc],
)
