import numpy as np

from generate_utils import typechar_to_npy_type as npy_types


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
    sig = sig.replace(' ', '')
    left, right = sig.split('->')
    leftparts = split_parts(left)
    rightparts = split_parts(right)
    core_dim_names = []
    for name in flatten(leftparts) + flatten(rightparts):
        try:
            name = int(name)
        except ValueError:
            pass
        if isinstance(name, str) and name != '' and name not in core_dim_names:
            core_dim_names.append(name)
    return core_dim_names, leftparts, rightparts


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


def gen(name, signature, corefuncs, docstring, header):
    """
    Generate C++ code to implement a gufunc.  The core operation of the
    gufunc must be implemented separately as C++ templated functions.
    `gen` will generate the function prototypes for the core functions
    that must be implemented.

    name : str
        gufunc name; this will be the name exposed in Python
    signature : str
        gufunc signature, e.g. '(m),()->()'
    corefuncs : Dict[str, List[str]]
        Dictionary that contains the core function names and the list
        of types that the core function handles.  The list of types
        corresponds to the types that are stored in the `types`
        attribute of a gufunc.  `corefuncs` is a dictionary of functions
        because in general, different core implementations of a gufunc
        might be necessary to handle different classes of types, even
        when templating is taken into account.  For example, the vector
        norm gufunc `vnorm` has two core implementations, one to handle
        real vectors and the other to handle complex vectors.  `corefuncs`
        for `vnorm` might be

            {'realvnorm': ['ff->f', 'dd->d', 'gg->g'],
             'complexvnorm': ['Ff->f', 'Dd->D', 'Gg->G']}
    docstring : str
        Docstring for the gufunc.
    header : str
        Filename of C++ .h file that defines the core functions.
    """
    varnames = get_varnames_from_docstring_and_sig(docstring, signature)

    core_dim_names, leftparts, rightparts = parse_gufunc_signature(signature)
    nin = len(leftparts)
    nout = len(rightparts)
    if nin + nout != len(varnames):
        raise ValueError('len(varnames) does not match the given signature')

    text = []
    text.append('// This file was generated automatically.  Do not edit!')
    text.append('')
    text.append('//')
    text.append(f'// Python extension module to implement the {name} gufunc.')
    text.append('//')
    text.append(_include)

    text.append('')
    text.append(f'#include "{header}"')

    loop_func_names = []

    for corename, coretypes in corefuncs.items():
        template_types, var_types = classify_typenames(coretypes)

        text.append('')
        text.append('//')
        text.append(f'// Prototype for `{corename}`, the C++ core function')
        text.append(f"// for the gufunc `{name}` with signature '{signature}'")
        text.append(f'// for types {coretypes}.')
        text.append('//')

        text.append('/*')
        template_type_names = [chr(ord('T') + j)
                               for j in range(len(template_types))]
        s = ', '.join(['typename ' + tn for tn in template_type_names])
        text.append(f"template<{s}>")
        text.append(f"static void {corename}(")

        vardecls = []

        for k, core_dim_name in enumerate(core_dim_names):
            vardecls.append((f'npy_intp {core_dim_name}',
                             f'// core dimension {core_dim_name}'))

        vt = [template_type_names[j] if isinstance(j, int)
              else npy_types[j].lower() for j in var_types]
        parts = leftparts + rightparts
        for k, (shape_name, varname, vct) in enumerate(zip(parts,
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
                vardecls.append(((f'npy_intp {varname}_strides'
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

        text.append('//')
        text.append('// Instantiated loop functions with C calling convention')
        text.append('//')
        text.append('')
        text.append('extern "C" {')

        for template_typecodes in zip(*template_types):
            print('*** template_typecodes:', template_typecodes)
            ctypes = [npy_types[c].lower() for c in template_typecodes]
            text.append('')
            ctypes_id = '_'.join([r.replace(' ', '').replace('npy_', '')
                                  for r in ctypes])
            loop_func_name = f'loop_{ctypes_id}'
            loop_func_names.append(loop_func_name)
            text.append(f'static void {loop_func_name}(')
            text.append('        char **args, const npy_intp *dimensions,')
            text.append('        const npy_intp* steps, void* data)')
            text.append('{')

            var_ctypes = []
            for vt in var_types:
                if isinstance(vt, int):
                    c = template_typecodes[vt]
                else:
                    c = vt
                var_ctypes.append(npy_types[c].lower())

            for k, (vct, varname) in enumerate(zip(var_ctypes, varnames)):
                text.append(f'    char *p_{varname} = args[{k}];')

            text.append('    npy_intp nloops = dimensions[0];')
            for k, core_dim_name in enumerate(core_dim_names):
                text.append('    '
                            f'npy_intp {core_dim_name} = dimensions[{k+1}];'
                            '  // core dimension')
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
            ttypes = ', '.join(ctypes)
            text.append(f'        {corename}<{ttypes}>(')
            if len(core_dim_names) > 0:
                sp = ' '*16
                text.append(sp + f'{", ".join(core_dim_names)},  '
                                 '// core dimension')
            istep = len(varnames)

            parts = leftparts + rightparts
            for k, (shape_name, varname, vct) in enumerate(zip(parts,
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
        text.append('')

        text.append('}  // extern "C"')

    text.append('')
    text.append(f'static char {name}_typecodes[] = {{')
    lines = []
    for corename, coretypes in corefuncs.items():
        for typesig in coretypes:
            charcodes = list(typesig.replace('->', ''))
            npy_typecodes = [npy_types[c] for c in charcodes]
            lines.append('    ' + ', '.join(npy_typecodes))
    text.append(',\n'.join(lines))
    text.append('};')

    text.append('')
    text.append(f'static PyUFuncGenericFunction {name}_funcs[] = {{')
    lines = []
    for loop_func_name in loop_func_names:
        lines.append(f'    (PyUFuncGenericFunction) &{loop_func_name}')
    text.append(',\n'.join(lines))
    text.append('};')
    text.append('')
    text.append(f'static void *{name}_data[{len(loop_func_names)}];')
    text.append('')

    NAME = name.upper()

    text.append(f"""
static PyMethodDef {name}_methods[] = {{
        {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef moduledef = {{
    PyModuleDef_HEAD_INIT,
    .m_name = "_{name}",
    .m_doc = "Module that defines the {name} function.",
    .m_size = -1,
    .m_methods = {name}_methods
}};
""")
    text.append(f'#define {NAME}_DOCSTRING \\')
    doclines = docstring.splitlines()
    llen = max([len(line) for line in doclines])
    for line in doclines:
        pad = llen - len(line) + 2
        text.append(f'"{line}\\n"' + ' '*pad + '\\')
    text.append('""')

    text.append(f"""
PyMODINIT_FUNC PyInit__{name}(void)
{{
    PyObject *module;
    int num_loop_funcs = {len(loop_func_names)};

    module = PyModule_Create(&moduledef);
    if (!module) {{
        return NULL;
    }}

    import_array();
    import_umath();

    // Create the {name} ufunc.
    if (ul_define_gufunc(module, "{name}", {NAME}_DOCSTRING, "{signature}",
                         num_loop_funcs,
                         {name}_funcs, {name}_data, {name}_typecodes) < 0) {{
        Py_DECREF(module);
        return NULL;
    }}

    return module;
}}
""")

    return '\n'.join(text)


if __name__ == "__main__":
    """
    # vnorm
    signature = '(n),()->()'
    varnames = ['x', 'order', 'out']
    corefuncs = dict(vnorm_core_calc=['ff->f', 'dd->d', 'gg->g'],
                     cvnorm_core_calc=['Ff->f', 'Dd->d', 'Gg->g'])

    text = gen('vnorm', varnames, signature, corefuncs,
               "This is the docstring. Testing 1 2 3.\nLorem ipsit.",
               "vnorm_gufunc.h")
    print(text)
    """

    signature = '(n),(),()->(),(n)'
    varnames = ['x', 'order', 'code', 'out1', 'out2']
    corefuncs = dict(func=['ffi->ff', 'ddi->dd', 'ggi->gg'])
    text = gen('vnorm', varnames, signature, corefuncs,
               "This is the docstring. Testing 1 2 3.\nLorem ipsit.",
               "vnorm_gufunc.h")
    print(text)
