//
// loggamma1p_ufunc.c
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "loggamma1p.h"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loops".
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void loggamma1p_double_loop(char **args, const npy_intp *dimensions,
                                   const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double x = *(double *)in;
        *((double *)out) = loggamma1p(x);
    }
}

static void loggamma1p_float_loop(char **args, const npy_intp *dimensions,
                                     const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double x = (double) *(float *)in;
        *((float *)out) = loggamma1p(x);
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays loggamma1p_typecodes and loggamma1p_data.
PyUFuncGenericFunction loggamma1p_funcs[] = {
    (PyUFuncGenericFunction) &loggamma1p_float_loop,
    (PyUFuncGenericFunction) &loggamma1p_double_loop
};

#define LOGGAMMA1P_NLOOPS \
    (sizeof(loggamma1p_funcs) / sizeof(loggamma1p_funcs[0]))

// These are the input and return type codes for the two loops.  It is
// created as a 1-d C array, but it can be interpreted as a 2-d array with
// shape (num_loops, num_args).  (num_args is the sum of the number of
// input args and output args.  In this case num_args is 2.)
static char loggamma1p_typecodes[] = {
    NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE
};

// The ufunc will not use the 'data' array.
static void *loggamma1p_data[] = {NULL, NULL};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef LogGamma1PMethods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_loggamma1p",
    .m_doc = "Module that defines the loggamma1p function.",
    .m_size = -1,
    .m_methods = LogGamma1PMethods
};


#define LOGGAMMA1P_DOCSTRING \
"loggamma1p(x, /, ...)\n"                             \
"\n"                                                  \
"Natural logarithm of gamma(1+x) for real x > -1.\n"  \
"\n"                                                  \
"nan is returned if x <= -1.\n"                       \
"\n"


PyMODINIT_FUNC PyInit__loggamma1p(void)
{
    PyObject *module;
    PyObject *loggamma1p_ufunc;
    int nin, nout;
    int status;

    module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    import_array();
    import_umath();

    // Create the loggamma1p ufunc object.
    nin = 1;
    nout = 1;
    loggamma1p_ufunc = PyUFunc_FromFuncAndData(loggamma1p_funcs,
                                               loggamma1p_data,
                                               loggamma1p_typecodes,
                                               LOGGAMMA1P_NLOOPS, nin, nout,
                                               PyUFunc_None,
                                               "loggamma1p",
                                               LOGGAMMA1P_DOCSTRING, 0);
    if (loggamma1p_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "loggamma1p",
                                (PyObject *) loggamma1p_ufunc);
    if (status == -1) {
        Py_DECREF(loggamma1p_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
