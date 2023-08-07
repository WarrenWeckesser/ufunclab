import numpy as np

from generate_utils import typechar_to_npy_type as npy_types
from generate_utils import typechar_to_npy_ctype


def classify_typenames(types):
    types = np.array([list(t.replace('->', '')) for t in types])
    w = [''.join(c) for c in types.T]
    uts = []
    result = []
    for i in range(len(w)):
        if w[i].replace(w[i][0], '') == '':
            result.append(w[i][0])
        else:
            if w[i] in uts:
                k = uts.index(w[i])
                result.append(k)
            else:
                result.append(len(uts))
                uts.append(w[i])
    return uts, result


def ordered_unique(w):
    u = []
    for item in w:
        if item not in u:
            u.append(item)
    return u


def toseq(s):
    s = s.split(',')
    t = []
    for item in s:
        try:
            item = int(item)
        except ValueError:
            pass
        t.append(item)
    if len(t) == 1:
        t = t[0]
    return t


def split_parts(s):
    parts = [toseq(p.lstrip('(').rstrip(')')) for p in s.split('),(')]
    return parts


def flatten(parts):
    return sum([[p] if isinstance(p, (str, int)) else p
                for p in parts], start=[])


def parse_gufunc_signature(sig):
    """
    Parameters
    ----------
    sig : gufunc shape signature (i.e. the string stored in the
          `signature` attribute of the Python gufunc object)

    Returns
    -------
    core_dims : List[Union[str,int]]
        unique set of core dimension symbols or constants used in `sig`
    shapes_in : List[{int, str, List[{int, str}]]
        "Shape"-portion of the items on the left of '->' in `sig`
    shapes_out : List[{int, str, List[{int, str}]]
        "Shape"-portion of the items on the right of '->' in sig.

    Examples
    --------
    >>> parse_gufunc_signature('(n)->()')
    (['n'], ['n'], [''])

    >>> parse_gufunc_signature('(m, n), (2), () -> (m, 2), (n)')
    (['m', 'n', 2], [['m', 'n'], 2, ''], [['m', 2], 'n'])

    """
    sig = sig.replace(' ', '')
    left, right = sig.split('->')
    shapes_in = split_parts(left)
    shapes_out = split_parts(right)
    core_dims = []
    for name in flatten(shapes_in) + flatten(shapes_out):
        try:
            name = int(name)
        except ValueError:
            pass
        if name != '' and name not in core_dims:
            core_dims.append(name)
    return core_dims, shapes_in, shapes_out


def shape_name_str(shape_name):
    return ', '.join([str(t) for t in shape_name])


def suffix(k, n):
    if n == 1:
        return ''
    else:
        return str(k)


_include = """
#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <assert.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayscalars.h"
#include "numpy/ufuncobject.h"

#include "../util/ufunc_tools.h"

"""


def get_varnames_from_docstring_and_sig(docstring, signature):
    docfirstline = docstring.split('\n', maxsplit=1)[0]
    start = docfirstline.find('(') + 1
    end = docfirstline.find(', /')
    params = docfirstline[start:end]
    varnames = params.replace(' ', '').split(',')
    nout = len(signature.replace(' ', '').split('->')[1].split('),('))
    if nout > 1:
        varnames.extend([f'out{k+1}' for k in range(nout)])
    else:
        varnames.append('out')
    return varnames


def generate_declaration(name, signature,
                         template_types, var_types,
                         varnames, core_dims, shapes,
                         corename, coretypes):
    nct = len(coretypes)
    if nct == 0:
        return ''

    text = []

    text.append('')
    text.append('//')
    text.append(f'// Prototype for `{corename}`, the C++ core function')
    text.append(f"// for the gufunc `{name}` with signature '{signature}'")
    text.append(f'// for types {coretypes}.')
    text.append('//')

    text.append('/*')
    if len(template_types) > 0:
        template_type_names = [chr(ord('T') + j)
                               for j in range(len(template_types))]
        if nct > 1:
            s = ', '.join(['typename ' + tn for tn in template_type_names])
            text.append(f"template<{s}>")
    text.append(f"static void {corename}(")

    vardecls = []

    for k, core_dim in enumerate(core_dims):
        if not isinstance(core_dim, int):
            vardecls.append((f'npy_intp {core_dim}',
                             f'// core dimension {core_dim}'))

    vt = [template_type_names[j] if isinstance(j, int)
          else typechar_to_npy_ctype(j) for j in var_types]
    for k, (shape_name, varname, vct) in enumerate(zip(shapes,
                                                       varnames,
                                                       vt)):
        if isinstance(shape_name, str):
            if shape_name == '':
                insert1 = ' '
                insert2 = ''
            else:
                insert1 = ' first element of '
                insert2 = (f', a strided 1-d array with {shape_name}'
                           ' elements')
            vardecls.append((f'{vct} *p_{varname}',
                             f'// pointer to{insert1}{varname}{insert2}'))
            if shape_name != '':
                vardecls.append((f'npy_intp {varname}_stride',
                                 ('// stride (in bytes) for elements of '
                                  f'{varname}')))
        elif isinstance(shape_name, int):
            if shape_name == 1:
                insert1 = ' '
                insert2 = ''
            else:
                insert1 = ' first element of '
                insert2 = (f', a strided 1-d array with {shape_name}'
                           ' elements')
            vardecls.append((f'{vct} *p_{varname}',
                             f'// pointer to{insert1}{varname}{insert2}'))
            if shape_name != 1:
                vardecls.append((f'npy_intp {varname}_stride',
                                 ('// stride (in bytes) for elements of '
                                  f'{varname}')))
        else:
            vardecls.append((f'{vct} *p_{varname}',
                             f'// pointer to first element of {varname}, '
                             f'a strided {len(shape_name)}-d array with '
                             f'shape ({shape_name_str(shape_name)})'))
            vardecls.append(((f'const npy_intp {varname}_strides'
                              f'[{len(shape_name)}]'),
                             (f'// array of length {len(shape_name)}'
                              ' of strides (in bytes) of '
                              f'{varname}')))

    dlen = max([len(d) for d, c in vardecls])
    for j, (d, c) in enumerate(vardecls):
        trail = ',' if j < len(vardecls) - 1 else ' '
        s = f"        {d}{trail}" + ' '*(dlen - len(d) + 2) + c
        text.append(s)
    text.append(');')
    text.append('*/')
    return '\n'.join(text)


def concrete_loop_function_name(corename, typecodes):
    ctypes = [typechar_to_npy_ctype(c) for c in typecodes]
    ctypes_id = [r.replace(' ', '').replace('npy_', '')
                 for r in ctypes]
    loop_func_name = '_'.join(['loop', corename] + ctypes_id)
    return loop_func_name


def generate_concrete_loop(name, corename, varnames,
                           template_typecodes, var_types,
                           core_dims, nonzero_coredims, shapes):
    loop_func_name = concrete_loop_function_name(corename, template_typecodes)
    text = []
    text.append('')
    text.append(f'static void {loop_func_name}(')
    text.append('        char **args, const npy_intp *dimensions,')
    text.append('        const npy_intp* steps, void* data)')
    text.append('{')

    ctypes = [typechar_to_npy_ctype(c) for c in template_typecodes]

    var_ctypes = []
    for vt in var_types:
        if isinstance(vt, int):
            c = template_typecodes[vt]
        else:
            c = vt
        var_ctypes.append(typechar_to_npy_ctype(c))

    for k, (vct, varname) in enumerate(zip(var_ctypes, varnames)):
        text.append(f'    char *p_{varname} = args[{k}];')

    text.append('    npy_intp nloops = dimensions[0];')

    dim_asserts = []
    for k, core_dim in enumerate(core_dims):
        if isinstance(core_dim, int):
            dim_asserts.append('    '
                               f'assert(dimensions[{k+1}] == {core_dim});')
        else:
            text.append('    '
                        f'npy_intp {core_dim} = dimensions[{k+1}];'
                        '  // core dimension')

    if len(dim_asserts) > 0:
        text.append('')
        text.append('\n'.join(dim_asserts))

    if nonzero_coredims is not None:
        for nonzero_coredim in nonzero_coredims:
            k = core_dims.index(nonzero_coredim)
            code = f"""
    if ({nonzero_coredim} == 0) {{
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
        PyErr_SetString(PyExc_ValueError,
                        "{name} core dimension {nonzero_coredim} must be at least 1.");
        NPY_DISABLE_C_API
        return;
    }}"""
            text.append(code)

    text.append('')
    text.append('    for (int j = 0; j < nloops; ++j, ')
    for k, varname in enumerate(varnames):
        line = []
        line.append(' '*32)
        line.append(f'p_{varname} += steps[{k}]')
        if k == len(varnames) - 1:
            line.append(') {')
        else:
            line.append(',')
        text.append(''.join(line))
    if len(ctypes) > 0:
        template_params = '<' + ', '.join(ctypes) + '>'
    else:
        template_params = ''
    text.append(f'        {corename}{template_params}(')
    core_dim_symbols = [dim for dim in core_dims if not isinstance(dim, int)]
    if len(core_dim_symbols) > 0:
        sp = ' '*16
        text.append(sp + f'{", ".join(core_dim_symbols)},  '
                         '// core dimension')
    istep = len(varnames)

    for k, (shape_name, varname, vct) in enumerate(zip(shapes,
                                                       varnames,
                                                       var_ctypes)):
        line = [' '*16]
        line.append(f'({vct} *) p_{varname}')
        if isinstance(shape_name, str):
            if shape_name != '':
                line.append(f', steps[{istep}]')
                istep += 1
        elif isinstance(shape_name, int):
            line.append(f', steps[{istep}]')
            istep += 1
        else:
            # shape_name is a sequence
            line.append(f', &steps[{istep}]')
            istep += len(shape_name)
        if k == len(varnames) - 1:
            line.append(');')
        else:
            line.append(', ')
        text.append(''.join(line))

    text.append('    }')

    text.append('}')

    return loop_func_name, '\n'.join(text)


def create_c_docstring_def(name, docstring):
    text = []
    NAME = name.upper()
    text.append(f'#define {NAME}_DOCSTRING \\')
    doclines = docstring.splitlines()
    llen = max([len(line) for line in doclines])
    for line in doclines:
        pad = llen - len(line) + 2
        text.append(f'"{line}\\n"' + ' '*pad + '\\')
    text.append('""')
    return '\n'.join(text)


def gen(extmod):
    """
    Generate C++ code to implement a gufunc.  The core operation of the
    gufunc must be implemented separately as C++ templated functions.
    `gen` will generate the function prototypes for the core functions
    that must be implemented.

    `extmod` must be an instance of UFuncExtMod.
    """
    header_files = []
    text = []
    text.append("// This file was generated automatically.  Do not edit!\n")
    text.append(_include)
    ufunc_nloops = []
    for ufunc in extmod.ufuncs:
        varnames = get_varnames_from_docstring_and_sig(ufunc.docstring,
                                                       ufunc.signature)
        c_docstring_def = create_c_docstring_def(ufunc.name, ufunc.docstring)
        core_dims, shapes_in, shapes_out = parse_gufunc_signature(ufunc.signature)

        nin = len(shapes_in)
        nout = len(shapes_out)
        if nin + nout != len(varnames):
            raise ValueError('len(varnames) does not match the given signature')
        shapes = shapes_in + shapes_out

        loop_func_names = []

        text.append('//')
        text.append(f"// code for ufunc '{ufunc.name}'")
        text.append('//')
        text.append('')
        if ufunc.header not in header_files:
            header_files.append(ufunc.header)
            text.append(f'#include "{ufunc.header}"')
            text.append('')
        text.append(c_docstring_def)

        for ufunc_source in ufunc.sources:
            corename = ufunc_source.funcname
            coretypes = ufunc_source.typesignatures
            if np.dtype('g') == np.dtype('d'):
                # long double is the same as double, so drop any
                # type signatures containing 'g' or 'G'.
                coretypes = [t for t in coretypes if 'g' not in t.lower()]

            template_types, var_types = classify_typenames(coretypes)

            dec = generate_declaration(ufunc.name, ufunc.signature,
                                       template_types, var_types,
                                       varnames, core_dims,
                                       shapes,
                                       corename, coretypes)
            text.append(dec)

            text.append('')
            text.append('//')
            text.append('// Instantiated loop functions with C calling convention')
            text.append('//')
            text.append('')
            text.append('extern "C" {')

            if len(template_types) == 0:
                loopname, code = generate_concrete_loop(ufunc.name,
                                                        corename, varnames,
                                                        [], var_types,
                                                        core_dims,
                                                        ufunc.nonzero_coredims,
                                                        shapes)
                loop_func_names.append(loopname)
                text.append(code)
            else:
                for template_typecodes in zip(*template_types):
                    loopname, code = generate_concrete_loop(ufunc.name,
                                                            corename, varnames,
                                                            template_typecodes,
                                                            var_types,
                                                            core_dims,
                                                            ufunc.nonzero_coredims,
                                                            shapes)
                    loop_func_names.append(loopname)
                    text.append(code)

            text.append('')
            text.append('}  // extern "C"')

        # Generate the array of typecodes for the ufunc.
        text.append('')
        text.append(f"// Typecodes for the ufunc '{ufunc.name}'")
        text.append(f'static char {ufunc.name}_typecodes[] = {{')
        lines = []
        for ufunc_source in ufunc.sources:
            corename = ufunc_source.funcname
            coretypes = ufunc_source.typesignatures
            if np.dtype('g') == np.dtype('d'):
                # long double is the same as double, so drop any
                # type signatures containing 'g' or 'G'.
                coretypes = [t for t in coretypes if 'g' not in t.lower()]
            for typesig in coretypes:
                charcodes = list(typesig.replace('->', ''))
                npy_typecodes = [npy_types[c] for c in charcodes]
                lines.append('    ' + ', '.join(npy_typecodes))
        text.append(',\n'.join(lines))
        text.append('};')

        # Generate the array loop function pointer for the ufunc.
        text.append('')
        text.append(f'static PyUFuncGenericFunction {ufunc.name}_funcs[] = {{')
        lines = []
        for loop_func_name in loop_func_names:
            lines.append(f'    (PyUFuncGenericFunction) &{loop_func_name}')
        text.append(',\n'.join(lines))
        text.append('};')

        ufunc_nloops.append(len(loop_func_names))

        # Generate the array of auxiliary data for the ufunc.
        text.append('')
        text.append(f'static void *{ufunc.name}_data[{len(loop_func_names)}];')
        text.append('')

    c_module_docstring_def = create_c_docstring_def("MODULE", extmod.docstring)
    text.append(c_module_docstring_def)
    text.append('')

    text.append(f"""
static PyMethodDef {extmod.module}_methods[] = {{
        {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef moduledef = {{
    PyModuleDef_HEAD_INIT,
    .m_name = "{extmod.module}",
    .m_doc = MODULE_DOCSTRING,
    .m_size = -1,
    .m_methods = {extmod.module}_methods
}};
""")

    text.append(f"""
PyMODINIT_FUNC PyInit_{extmod.module}(void)
{{
    PyObject *module;
    PyUFuncObject *gufunc;

    module = PyModule_Create(&moduledef);
    if (!module) {{
        return NULL;
    }}

    import_array();
    import_umath();
""")

    for nloops, ufunc in zip(ufunc_nloops, extmod.ufuncs):
        text.append(f"""
    // Create the {ufunc.name} ufunc.  The gufunc object is returned on
    // success, but it is borrowed reference. The object is owned by the
    // module.
    gufunc = ul_define_gufunc(module, "{ufunc.name}",
                         {ufunc.name.upper()}_DOCSTRING,
                         "{ufunc.signature}",
                         {nloops},
                         {ufunc.name}_funcs,
                         {ufunc.name}_data,
                         {ufunc.name}_typecodes);
    if (gufunc == NULL) {{
        Py_DECREF(module);
        return NULL;
    }}
""")
    text.append('    return module;')
    text.append('}')

    return '\n'.join(text)
