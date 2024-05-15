//
// debye1_ufunc.c
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "debye1_generated.h"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loops".
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void debye1_double_loop(char **args, const npy_intp *dimensions,
                               const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double x = *(double *)in;
        *((double *)out) = debye1(x);
    }
}

static void debye1_float_loop(char **args, const npy_intp *dimensions,
                                     const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double x = (double) *(float *)in;
        *((float *)out) = debye1(x);
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays debye1_typecodes and debye1_data.
PyUFuncGenericFunction debye1_funcs[] = {
    (PyUFuncGenericFunction) &debye1_float_loop,
    (PyUFuncGenericFunction) &debye1_double_loop
};

#define DEBYE1_NLOOPS \
    (sizeof(debye1_funcs) / sizeof(debye1_funcs[0]))

// These are the input and return type codes for the two loops.  It is
// created as a 1-d C array, but it can be interpreted as a 2-d array with
// shape (num_loops, num_args).  (num_args is the sum of the number of
// input args and output args.  In this case num_args is 2.)
static char debye1_typecodes[] = {
    NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE
};

// The ufunc will not use the 'data' array.
static void *debye1_data[] = {NULL, NULL};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef Debye1Methods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_debye1",
    .m_doc = "Module that defines the debye1 function.",
    .m_size = -1,
    .m_methods = Debye1Methods
};


#define DEBYE1_DOCSTRING \
"debye1(x, /, ...)\n"                                 \
"\n"                                                  \
"The Debye function D1(x).\n"                         \
"\n"                                                  \
"See https://en.wikipedia.org/wiki/Debye_function\n"  \
"\n"


PyMODINIT_FUNC PyInit__debye1(void)
{
    PyObject *module;
    PyObject *debye1_ufunc;
    int nin, nout;
    int status;

    module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    import_array();
    import_umath();

    // Create the debye1 ufunc object.
    nin = 1;
    nout = 1;
    debye1_ufunc = PyUFunc_FromFuncAndData(debye1_funcs,
                                           debye1_data,
                                           debye1_typecodes,
                                           DEBYE1_NLOOPS, nin, nout,
                                           PyUFunc_None,
                                           "debye1",
                                           DEBYE1_DOCSTRING, 0);
    if (debye1_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "debye1",
                                (PyObject *) debye1_ufunc);
    if (status == -1) {
        Py_DECREF(debye1_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
