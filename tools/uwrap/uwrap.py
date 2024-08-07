# Experimental code for generating wrappers of ufuncs that have a
# nicer API.  Specifically, in the wrappers, give the core parameters
# nicer names, and make them keyword parameters.  For example, instead
# of
#     vnorm(x1, x2, ...)
#     backlash(x1, x2, x2, ...)
#
# the signatures of the wrappers of `vnorm` and `backlash` are
#
#     vnorm(x, p, ...)
#     backlash(x, deadband, initial, ...)
#
# In fact, vnorm is wrapped so that a default value of p is provided:
#
#     vnorm(x, p=2, ...)


# These are the functions that act like reductions on an unordered set.
# In the wrappers, the processing of the `axis` parameter will be
# extended to allow `axis` to accept `None` or a tuple.  In each case,
# the first parameter is the only parameter with a nontrival dimension
# so for now, we don't have to consider a signature such as `(n),(n)->()`.
set_reduction_funcs = ['all_same', 'gmean', 'hmean', 'mad', 'mad1',
                       'peaktopeak', 'rmad', 'rmad1']


def get_input_sig_strings(ufunc):
    """
    Return the input items of a gufunc signature, with the
    parentheses removed.  If ufunc is not a gufunc, `['']*ufunc.nin`
    is returned.

    For example, if the signature of ufunc is "(m,n),(),(n)->(n),()",
    the return value is ['m,n', '', 'n'].
    """
    if ufunc.signature is not None:
        return ufunc.signature.split('->')[0][1:-1].split('),(')
    else:
        return ['']*ufunc.nin


def make_self_assignment(s):
    """
    `s` must be a string that contains one '='.

    Converts a string of the form "plate=shrimp" to "plate=plate".
    """
    left, right = s.split('=')
    return '='.join([left, left])


def check_params(params, ufunc):
    if ufunc.nout > 1:
        multi_out_params = [f'out{k}' for k in range(1, ufunc.nout + 1)]
    else:
        multi_out_params = []
    # Note:
    #  * `where` is a parameter of plain ufuncs only.
    #  * `axis`, `axes` and `keepdims` are parameters of gufuncs only.
    all_ufunc_params = ['out', 'casting', 'order', 'dtype', 'subok',
                        'signature', 'extobj', 'where',
                        'axis', 'axes', 'keepdims'] + multi_out_params
    args = []
    kwargs = []
    for item in params:
        if isinstance(item, str):
            name = item
            args.append(name)
            if len(kwargs) > 0:
                raise ValueError(f"{ufunc.__name__}: parameter {name} must "
                                 "have a default value")
        else:
            name = item[0]
            kwargs.append(item)
        if name in all_ufunc_params:
            raise ValueError(f"{ufunc.__name__}: parameter '{name}' is also "
                             "a ufunc parameter.")
    return args, kwargs


def to_lines(line, offset):
    parts = line.split(', ')
    lines = []
    pad = 0
    while len(parts) > 0:
        k = 0
        n = 0
        while k != len(parts) and n + pad + len(parts[k]) + 2 < 80:
            n += len(parts[k]) + 2
            k += 1
        if k == len(parts):
            final = ''
        else:
            final = ','
        lines.append(' '*pad + ', '.join(parts[:k]) + final)
        pad = offset
        parts = parts[k:]

    return lines


def get_def_and_call_sig(ufunc, params):
    if ufunc.nout > 1:
        out_arg = f"out={(None,)*ufunc.nout}"
    else:
        out_arg = "out=None"
    common_args = [out_arg, "casting='same_kind'", "order='K'", "subok=True"]
    # Common keyword parameters not explicitly included in the
    # def signature: signature, extobj
    ufunc_only_args = ['where=True']
    if ufunc.__name__ in set_reduction_funcs:
        axis_arg = ['axis=None']
    else:
        axis_arg = []
    # gufunc_only_args = ["axis=None", "axes=None", "keepdims=None"]
    if ufunc.signature is None:
        control_args = common_args + ufunc_only_args
    else:
        # control_args = common_args + gufunc_only_args
        control_args = common_args + axis_arg
    args, kwargs = check_params(params, ufunc)
    kwdstrs = ([f'{name}={value!r}' for name, value in kwargs]
               + control_args)
    def_sig = ', '.join(args + ['*'] + kwdstrs + ['dtype=None']) + ', **kwargs'
    call_sig = ', '.join(args + [kw[0] for kw in kwargs]
                         + [make_self_assignment(arg) for arg in control_args]
                         + ['**kwargs'])
    return def_sig, call_sig


def uwrap(ufunc, namespace, params=None, name=None):
    """
    ufunc : NumPy ufunc or gufunc
        The function to be given a wrapper.
    namespace : str
        The namespace in which ufunc is defined.
    params : list of str
        The parameter names to be used in the signature of
        the wrapper for the core parameters.  I.e. these are
        the names that replace the generic names that are
        generated by NumPy.
    name : str
        The name of the wrapper function.  If not given,
        ``ufunc.__name__`` is used.
    """
    if name is None:
        name = ufunc.__name__

    if params is None:
        # Get the third line of the docstring.  This skips the two lines
        # added to the docstring by the NumPy code, and gets the line that
        # has the form
        #     name(arg1, args2, ..., /, ...)
        # This is a convention used in ufunclab when creating the docstrings.
        # It is not automatically generated by NumPy, and it is not required
        # by NumPy, so this will not work with ufuncs that do not follow the
        # convention (e.g. np.sin, or any other NumPy ufunc).
        sigline = ufunc.__doc__.splitlines()[2].strip()
        k = sigline.find('(')
        params = sigline[k+1:-6].split(', ')[:-1]

    nin = ufunc.nin
    if len(params) != nin:
        raise ValueError(f"ufunc '{name}' has {nin} input arguments, but "
                         f"len(params) is {len(params)}")

    def_sig, call_sig = get_def_and_call_sig(ufunc, params)

    source = ['', '']

    # First line is def statement of the wrapper.
    line = f"def {name}({def_sig}):"
    source.extend(to_lines(line, offset=5 + len(name)))

    # Copy the docstring.  The first four lines are skipped; the first line
    # was generated by numpy for the ufunc, and the third line is the line
    # that follows the conventions `funcname(param1, param2, /, ...)`, which
    # isn't needed in the wrapped function.  The second and fourth lines are
    # expected to be blank.
    source.append('    """')
    doc_lines = [('    ' + line) if line != '' else ''
                 for line in ufunc.__doc__.splitlines()[4:]]
    source.extend(doc_lines)
    source.append('    """')

    if name in set_reduction_funcs:
        source.append('')
        source.append('    # Handle axis is None or a tuple.')
        source.append(f'    {params[0]}, axis = '
                      f'_process_axis_for_set_reduction_func({params[0]}, '
                      'axis)')
        source.append('')

    source.append("    if dtype is not None:")
    source.append("        kwargs['dtype'] = dtype")

    # Generate the return statement with the call of the unwrapped ufunc.
    ufunc_name = namespace + ('.' if namespace else '') + ufunc.__name__
    call_ufunc = f"    return {ufunc_name}({call_sig})"
    source.extend(to_lines(call_ufunc, offset=12+len(ufunc_name)))
    source.append('')
    return '\n'.join(source)


wrapper_preamble = '''
"""
Wrappers for the functions in ufunclab.

This module defines wrapper functions for the functions in ufunclab.
The wrapper functions have descriptive parameter names for the core
parameters instead of the geneeric names that are generated by NumPy
(e.g. (`x`, or `x1`, `x2`, ...) .  These parameters are keyword parameters,
so the functions may be called with keyword assignment rather than with
purely positional arguments.
"""

import numpy as np
import ufunclab


def _process_axis_for_set_reduction_func(x, axis):
    """
    This enables the wrapper to handle `axis=None` or axis being a
    tuple. This function is intended to be applied *only* to functions
    with the signature '(n)->()', and really only for those gufuncs for
    with the input can be considered to be an unordered set.
    """
    if axis is None:
        x = x.ravel()
        axis = -1
    elif isinstance(axis, tuple):
        # Move the dimensions in axis to the end.
        x = np.moveaxis(x, axis, tuple(range(-1, -1-len(axis), -1)))
        # Flatten those final dimensions into a single dimension.
        x = x.reshape(x.shape[:x.ndim - len(axis)] + (-1,))
        axis = -1
    return x, axis
'''


if __name__ == "__main__":
    # Run this after ufunclab has been installed.
    from datetime import datetime
    from os.path import basename
    import numpy as np
    import ufunclab as ul

    # Get all the ufuncs (plain ufuncs and gufuncs) in the `ufunclab`
    # namespace.
    ufuncs = [getattr(ul, name) for name in dir(ul)
              if isinstance(getattr(ul, name), np.ufunc)]

    with open('wrappers.py', 'w') as f:
        f.write(f"# Do not edit this file!\n"
                f"# This file was generated by '{basename(__file__)}'"
                f" on {datetime.now()}.\n")
        f.write(wrapper_preamble)
        for ufunc in ufuncs:
            # Give default values to some of the parameters of a few ufuncs.
            # TODO: Make this a configuration setting somewhere.
            if ufunc.__name__ == 'vnorm':
                params = ['x', ('p', 2)]
            elif ufunc.__name__ == 'step':
                params = ['x',
                          ('a', 0), ('flow', 0), ('fa', 0.5), ('fhigh', 1)]
            elif ufunc.__name__ in ['linearstep',
                                    'smoothstep3', 'smoothstep5']:
                params = ['x',
                          ('a', 0), ('b', 1), ('fa', 0), ('fb', 1)]
            else:
                # Get the params from the docstring.
                params = None
            s = uwrap(ufunc, namespace="ufunclab", params=params)
            f.write(s)
