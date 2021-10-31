//
// logfactorial_ufunc.c
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "logfactorial.h"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loops".  For this example, we implement just two inner
// loop, for 32 and 64 bit signed integers.  These are the 'i->d' and 'l->d'
// loops.
//
// Other types will be cast to 32 or 64 bit integers by the ufunc machinery.
// Given the nature of this function, there is no point in creating loops
// for floating point inputs.  We *could* create loops for all the other
// integers, but to do that we should use the template processing provided
// by numpy.distutils.  As this ufunc is intended to be a tutorial, we'll
// keep it simple and provide just the two loops, coded up explicitly here.
// That is also why some of the convenience macros provided in the NumPy
// header files are not used.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void logfactorial_int32_loop(char **args, const npy_intp *dimensions,
                                    const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        // Get the int32_t value and cast it to int64_t, to be passed to
        // the C function logfactorial(x).
        int64_t x = (int64_t) *(int32_t *)in;
        if (x < 0) {
            *((double *)out) = NAN;
        }
        else {
            *((double *)out) = logfactorial(x);
        }
    }
}

static void logfactorial_int64_loop(char **args, const npy_intp *dimensions,
                                    const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        int64_t x = *(int64_t *)in;
        if (x < 0) {
            *((double *)out) = NAN;
        }
        else {
            *((double *)out) = logfactorial(x);
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays logfactorial_typecodes and logfactorial_data.
PyUFuncGenericFunction logfactorial_funcs[] = {
    (PyUFuncGenericFunction) &logfactorial_int32_loop,
    (PyUFuncGenericFunction) &logfactorial_int64_loop
};

#define LOGFACTORIAL_NLOOPS \
    (sizeof(logfactorial_funcs) / sizeof(logfactorial_funcs[0]))

// These are the input and return type codes for the two loops.  It is
// created as a 1-d C array, but it can be interpreted as a 2-d array with
// shape (num_loops, num_args).  (num_args is the sum of the number of
// input args and output args.  In this case num_args is 2.)
static char logfactorial_typecodes[] = {
    NPY_INT32, NPY_DOUBLE,
    NPY_INT64, NPY_DOUBLE
};

// The ufunc will not use the 'data' array.
static void *logfactorial_data[] = {NULL, NULL};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef LogFactorialMethods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_logfactorial",
    .m_doc = "Module that defines the logfactorial function.",
    .m_size = -1,
    .m_methods = LogFactorialMethods
};


#define LOGFACTORIAL_DOCSTRING \
"logfactorial(x, /, ...)\n"                                \
"\n"                                                       \
"Natural logarithm of the factorial of the integer x."


PyMODINIT_FUNC PyInit__logfact(void)
{
    PyObject *module;
    PyObject *logfactorial_ufunc;
    int nin, nout;
    int status;

    module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    import_array();
    import_umath();

    // Create the logfactorial ufunc object.
    nin = 1;
    nout = 1;
    logfactorial_ufunc = PyUFunc_FromFuncAndData(logfactorial_funcs,
                                                 logfactorial_data,
                                                 logfactorial_typecodes,
                                                 LOGFACTORIAL_NLOOPS, nin, nout,
                                                 PyUFunc_None,
                                                 "logfactorial",
                                                 LOGFACTORIAL_DOCSTRING, 0);
    if (logfactorial_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "logfactorial",
                                (PyObject *) logfactorial_ufunc);
    if (status == -1) {
        Py_DECREF(logfactorial_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
