//
// log1p_ufunc.c
//
// This C extension module defines the `log1p` ufunc.  `log1p`
// is implemented for the type np.complex128 only (type signature
// is `D->D`).
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <complex.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loop". The type signatures of the log1p ufunc are
//     >>> log1p.types
//     ['D->D']
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void
log1p_D_D_loop(char **args, const npy_intp *dimensions,
               const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double _Complex w;
        double _Complex z = *(double _Complex *) in;

        if (isnan(creal(z)) || isnan(cimag(z))) {
            w = CMPLX(NAN, NAN);
        }
        else {
            double _Complex u = z + 1.0;
            if (creal(u) == 1.0 && cimag(u) == 0.0) {
                // z + 1 == 1
                w = z;
            }
            else {
                if (creal(u) - 1.0 == creal(z)) {
                    // u - 1 == z
                    w = clog(u);
                }
                else {
                    w = clog(u) * (z / (u - 1.0));
                }
            }
        }
        *((double _Complex *) out) = w;
    }
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays log1p_typecodes and log1p_data.
PyUFuncGenericFunction log1p_funcs[] = {
    (PyUFuncGenericFunction) &log1p_D_D_loop,
};

#define LOG1P_NLOOPS \
    (sizeof(log1p_funcs) / sizeof(log1p_funcs[0]))

// These are the input and return type codes for the inner loops.
static char log1p_typecodes[] = {
    NPY_CDOUBLE, NPY_CDOUBLE,
};

// The ufunc will not use the 'data' array.
static void *log1p_data[] = {NULL, NULL, NULL, NULL, NULL, NULL};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef Log1pMethods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_log1p",
    .m_doc = "Module that defines the log1p function.",
    .m_size = -1,
    .m_methods = Log1pMethods
};


#define LOG1P_DOCSTRING                     \
"log1p(z, /, ...)\n"                        \
"\n"                                        \
"log1p(z) for complex z."


PyMODINIT_FUNC PyInit__log1p(void)
{
    PyObject *module;
    PyObject *log1p_ufunc;
    int nin, nout;
    int status;

    module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    import_array();
    import_umath();

    // Create the log1p ufunc object.
    nin = 1;
    nout = 1;
    log1p_ufunc = PyUFunc_FromFuncAndData(log1p_funcs, log1p_data,
                                          log1p_typecodes,
                                          LOG1P_NLOOPS, nin, nout,
                                          PyUFunc_None,
                                          "log1p", LOG1P_DOCSTRING, 0);
    if (log1p_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "log1p",
                                (PyObject *) log1p_ufunc);
    if (status == -1) {
        Py_DECREF(log1p_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
