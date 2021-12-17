import os
from os import path
from textwrap import dedent
from generate_utils import (typechar_to_ctype, typesig_to_ext,
                            typechar_to_npy_type,
                            header_to_concrete_filenames)


def cdef_docstring(funcname, docstring):
    cdef = (f'#define {funcname.upper()}_DOCSTRING \\\n"'
            + docstring.replace('\n', '\\n"\\\n"') + '"')
    return cdef


def cap(s):
    return s.replace('_', ' ').title().replace(' ', '')


def print_extmod_start(modulename, funcs, file):
    tmpl = \
f"""
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef {cap(modulename)}Methods[] = {{
        {{NULL, NULL, 0, NULL}}
}};

extern "C" {{

static struct PyModuleDef moduledef = {{
    PyModuleDef_HEAD_INIT,
    .m_name = "{modulename}",
    .m_doc = "Module that defines the ufuncs {', '.join([f.ufuncname for f in funcs])}.",
    .m_size = -1,
    .m_methods = {cap(modulename)}Methods
}};


PyMODINIT_FUNC PyInit_{modulename}(void)
{{
    PyObject *module;
    PyUFuncObject *ufunc;
    size_t ntypes;
    int nin;
    int nout = 1;
    int status;

    import_array();
    import_umath();

    module = PyModule_Create(&moduledef);
    if (!module) {{
        return NULL;
    }}

"""
    print(tmpl, file=file)


def print_ufunc_create(func, file):
    nin = len(func.types[0].split('->')[0])
    ntypes = len(func.types)
    name = func.ufuncname
    print(file=file)
    print(f'    // Create the {name} ufunc.', file=file)
    print(f'    ntypes = {ntypes};', file=file)
    print(f'    nin = {nin};', file=file)
    print('    ufunc = (PyUFuncObject *) PyUFunc_FromFuncAndData(', file=file)
    print(f'            {name}_funcs, {name}_data, {name}_types,', file=file)
    print('            ntypes, nin, nout,', file=file)
    print(f'            PyUFunc_None, "{name}",', file=file)
    print(f'            {name.upper()}_DOCSTRING, 0);', file=file)
    print('    if (ufunc == NULL) {', file=file)
    print('        Py_DECREF(module);', file=file)
    print('        return NULL;', file=file)
    print('    }', file=file)
    print(f'    status = PyModule_AddObject(module, "{name}", (PyObject *) ufunc);',
          file=file)
    print(
"""    if (status == -1) {
        Py_DECREF(ufunc);
        Py_DECREF(module);
        return NULL;
    }""", file=file)


preamble = """
    #define PY_SSIZE_T_CLEAN
    #include "Python.h"

    #include <stddef.h>
    #include <stdint.h>

    #define NPY_NO_DEPRECATED_API NPY_API_VERSION
    #include "numpy/ndarraytypes.h"
    #include "numpy/arrayscalars.h"
    #include "numpy/ufuncobject.h"
    """


def generate_ufunc_extmod(modulename, cxxgenpath, header, funcs):
    """
    modulename: str
        Name of the extension module.
    funcs: sequence of Func instances
        Each instance describes a templated C++ function and the type
        signature for the ufunc loops to be generated.

    The C implementation of the extension module is written
    to the file `modulename + 'module.cxx'`.
    """
    extmod_filename = modulename + 'module.cxx'
    if path.exists(extmod_filename):
        raise RuntimeError(f"file '{extmod_filename} already exists.")

    concrete_header, concrete_filename = header_to_concrete_filenames(header)

    gendir = path.join(cxxgenpath, 'generated')
    if not path.exists(gendir):
        os.mkdir(gendir)
    extmod_fullpath = path.join(gendir, extmod_filename)

    with open(extmod_fullpath, 'w') as f:
        print('// This file was generated automatically.  Do not edit!',
              file=f)
        print(file=f)
        print('//', file=f)
        print(f'// {modulename} extension module', file=f)
        print('//', file=f)
        print(f'// This module creates the ufuncs: {", ".join([f.ufuncname for f in funcs])}',
              file=f)
        print('//', file=f)
        print(dedent(preamble), file=f)

        print("""
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc loop functions and data definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
""", file=f)

        print(f'#include "{concrete_header}"', file=f)
        print(file=f)
        for func in funcs:
            print('//', file=f)
            print(f"// {func.ufuncname} (ufunc wrapper of the C++ function {func.cxxname})",
                  file=f)
            print("//", file=f)
            print(file=f)
            print('extern "C" {', file=f)
            print(file=f)
            for typesig in func.types:
                in_types, out_type = typesig.split('->')
                assert len(out_type) == 1
                nin = len(in_types)
                in_ctypes = [typechar_to_ctype[c] for c in in_types]
                out_ctype = typechar_to_ctype[out_type]
                #
                # Print the loop functions.
                #
                print('static void', file=f)
                loopfuncname = f'{func.ufuncname}_{typesig_to_ext(typesig)}_loop'
                print(f'{loopfuncname}(char **args, const npy_intp *dimensions,',
                      file=f)
                print(' '*len(loopfuncname), 'const npy_intp* steps, void* data)',
                      file=f)
                print('{', file=f)
                for k in range(nin):
                    print(f'    char *px{k} = args[{k}];', file=f)
                print(f'    char *pout = args[{k+1}];', file=f)
                print('    npy_intp n = dimensions[0];', file=f)
                print('    for (int j = 0; j < n; ++j,', file=f)
                for k in range(nin):
                    print(' '*26, f'px{k} += steps[{k}],', file=f)
                print(' '*26, f'pout += steps[{k+1}]) {{', file=f)
                for k in range(nin):
                    print(f'        {in_ctypes[k]} x{k} = *(({in_ctypes[k]} *) px{k});',
                          file=f)
                args = ', '.join([f'x{k}' for k in range(nin)])
                print(f'        *(({out_ctype} *) pout) = {func.ufuncname}_{typesig_to_ext(typesig)}({args});',
                      file=f)
                print('    }', file=f)
                print('}', file=f)
                print(file=f)
            print('}  // extern "C"', file=f)
            print(file=f)

            #
            # Print the type array
            #
            print(f'static char {func.ufuncname}_types[] = {{', file=f)
            for k, typesig in enumerate(func.types):
                in_types, out_type = typesig.split('->')
                end = ',\n' if k < len(func.types) - 1 else '\n'
                print('  ', ', '.join([typechar_to_npy_type[c] for c in in_types]
                                      + [typechar_to_npy_type[out_type]]),
                      end=end, file=f)
            print('};', file=f)
            print(file=f)
            #
            # Print the array of loop function pointers.
            #
            print(f'static PyUFuncGenericFunction {func.ufuncname}_funcs[] = {{',
                  file=f)
            for k, typesig in enumerate(func.types):
                end = ',\n' if k < len(func.types) - 1 else '\n'
                print(f'   (PyUFuncGenericFunction) &{func.ufuncname}_{typesig_to_ext(typesig)}_loop',
                      end=end, file=f)
            print('};', file=f)
            print(file=f)
            #
            # Print the extra data array.
            #
            print(f'static void *{func.ufuncname}_data[{len(func.types)}];',
                  file=f)
            print(file=f)
            #
            # Print the #define for the docstring.
            cdef = cdef_docstring(func.ufuncname, func.docstring)
            print(cdef, file=f)
            print(file=f)

        print_extmod_start(modulename, funcs, file=f)
        for func in funcs:
            print_ufunc_create(func, file=f)
        print('    return module;', file=f)
        print('}', file=f)
        print(file=f)
        print('}  // extern "C"', file=f)
