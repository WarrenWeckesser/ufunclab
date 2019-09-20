//
// logfactorial_ufunc.c
//

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "logfactorial.h"


static PyMethodDef LogFactorialMethods[] = {
        {NULL, NULL, 0, NULL}
};


static void logfactorial_loop(char **args, npy_intp *dimensions,
                              npy_intp* steps, void* data)
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

PyUFuncGenericFunction funcs[1] = {&logfactorial_loop};

// These are the input and return dtypes of logfactorial.
static char types[2] = {NPY_INT64, NPY_DOUBLE};

static void *data[1] = {NULL};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_logfactorial",
    .m_doc = "Module that defines the logfactorial function.",
    .m_size = -1,
    .m_methods = LogFactorialMethods
};


#define LOGFACTORIAL_DOCSTRING \
    "Natural logarithm of the factorial of the integer x."


PyMODINIT_FUNC PyInit__logfact(void)
{
    PyObject *m;
    PyObject *logfactorial_ufunc;
    PyObject *d;
    size_t ntypes;
    int nin, nout;

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    ntypes = sizeof(funcs) / sizeof(funcs[0]);
    nin = 1;
    nout = 1;
    logfactorial_ufunc = PyUFunc_FromFuncAndData(funcs, data,
                                                 types, ntypes, nin, nout,
                                                 PyUFunc_None, "logfactorial",
                                                 LOGFACTORIAL_DOCSTRING, 0);

    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "logfactorial", logfactorial_ufunc);
    Py_DECREF(logfactorial_ufunc);

    return m;
}
