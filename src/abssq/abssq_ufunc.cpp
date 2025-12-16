//
// abssq_ufunc.cpp
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loops". The type signatures of the abssq ufunc are
//     >>> abssq.types
//     ['f->f', 'd->d', 'g->g', 'F->f', 'D->d', 'G->g']
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <typename Real>
inline static void
abssq_real_contig(npy_intp n, Real *in, Real *out)
{
    for (npy_intp i = 0; i < n; ++i) {
        out[i] = in[i]*in[i];
    }
}

template <typename Real>
static void abssq_real_loop(char **args, const npy_intp *dimensions,
                            const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    if (in_step == sizeof(Real) && out_step == sizeof(Real)) {
        abssq_real_contig(dimensions[0], reinterpret_cast<Real *>(in), reinterpret_cast<Real *>(out));
        return;
    }
    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        Real x = *reinterpret_cast<Real *>(in);
        *(reinterpret_cast<Real *>(out)) = x*x;
    }
}

template <typename Real>
inline static void
abssq_complex_contig(npy_intp n, Real *in, Real *out)
{
    for (npy_intp i = 0; i < n; ++i) {
        out[i] = in[2*i]*in[2*i] + in[2*i+1]*in[2*i+1];
    }
}

template <typename Real>
static void abssq_complex_loop(char **args, const npy_intp *dimensions,
                               const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    if (in_step == 2*sizeof(Real) && out_step == sizeof(Real)) {
        abssq_complex_contig(dimensions[0], reinterpret_cast<Real *>(in), reinterpret_cast<Real *>(out));
        return;
    }
    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        Real x = *reinterpret_cast<Real *>(in);
        Real y = *reinterpret_cast<Real *>(in + sizeof(Real));
        *(reinterpret_cast<Real *>(out)) = x*x + y*y;
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays abssq_typecodes and abssq_data.
PyUFuncGenericFunction abssq_funcs[] = {
    (PyUFuncGenericFunction) &abssq_real_loop<float>,
    (PyUFuncGenericFunction) &abssq_real_loop<double>,
    (PyUFuncGenericFunction) &abssq_real_loop<long double>,
    (PyUFuncGenericFunction) &abssq_complex_loop<float>,
    (PyUFuncGenericFunction) &abssq_complex_loop<double>,
    (PyUFuncGenericFunction) &abssq_complex_loop<long double>
};

#define ABSSQ_NLOOPS \
    (sizeof(abssq_funcs) / sizeof(abssq_funcs[0]))

// These are the input and return type codes for the inner loops.  It is
// created as a 1-d C array, but it can be interpreted as a 2-d array with
// shape (num_loops, num_args).  num_args is the sum of the number of
// input args and output args.  In this case num_args is 2.
static char abssq_typecodes[] = {
    NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE,
    NPY_CFLOAT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_DOUBLE,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE
};

// The ufunc will not use the 'data' array.
static void *abssq_data[] = {NULL, NULL, NULL, NULL, NULL, NULL};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef AbsSqMethods[] = {
        {NULL, NULL, 0, NULL}
};

/*
static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_abssq",
    .m_doc = "Module that defines the abssq function.",
    .m_size = -1,
    .m_methods = AbsSqMethods
};
*/

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,                      // m_base
    "_abssq",                                   // m_name
    "Module that defines the abssq function.",  // m_doc
    -1,                                         // m_size
    AbsSqMethods,                               // m_methods
    NULL,                                       // m_slots
    NULL,                                       // m_traverse
    NULL,                                       // m_clear
    NULL,                                       // m_free
};


#define ABSSQ_DOCSTRING                     \
"abssq(z, /, ...)\n"                        \
"\n"                                        \
"Squared absolute value for inexact input."


PyMODINIT_FUNC PyInit__abssq(void)
{
    PyObject *module;
    PyObject *abssq_ufunc;
    int nin, nout;
    int status;

    module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    if (PyArray_ImportNumPyAPI() < 0) {
        return NULL;
    }
    import_umath();

    // Create the abssq ufunc object.
    nin = 1;
    nout = 1;
    abssq_ufunc = PyUFunc_FromFuncAndData(abssq_funcs, abssq_data,
                                          abssq_typecodes,
                                          ABSSQ_NLOOPS, nin, nout,
                                          PyUFunc_None,
                                          "abssq", ABSSQ_DOCSTRING, 0);
    if (abssq_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "abssq",
                                (PyObject *) abssq_ufunc);
    if (status == -1) {
        Py_DECREF(abssq_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
