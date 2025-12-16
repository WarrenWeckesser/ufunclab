//
// cabssq_ufunc.c
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loops". The type signatures of the cabssq ufunc are
//     >>> cabssq.types
//     ['F->f', 'D->d', 'G->g']
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

template <typename Real>
static void cabssq_complex_contig(npy_intp n, Real *in, Real *out)
{
    for (npy_intp i = 0; i < n; ++i) {
        out[i] = in[2*i]*in[2*i] + in[2*i+1]*in[2*i+1];
    }
}

template <typename Real>
static void cabssq_complex_loop(char **args, const npy_intp *dimensions,
                                const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    if (in_step == 2*sizeof(Real) && out_step == sizeof(Real)) {
        cabssq_complex_contig(dimensions[0], reinterpret_cast<Real *>(in), reinterpret_cast<Real *>(out));
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
// along with the arrays cabssq_typecodes and cabssq_data.
PyUFuncGenericFunction cabssq_funcs[] = {
    (PyUFuncGenericFunction) &cabssq_complex_loop<float>,
    (PyUFuncGenericFunction) &cabssq_complex_loop<double>,
    (PyUFuncGenericFunction) &cabssq_complex_loop<long double>
};

#define CABSSQ_NLOOPS \
    (sizeof(cabssq_funcs) / sizeof(cabssq_funcs[0]))

// These are the input and return type codes for the inner loops.  It is
// created as a 1-d C array, but it can be interpreted as a 2-d array with
// shape (num_loops, num_args).  num_args is the sum of the number of
// input args and output args.  In this case num_args is 2.
static char cabssq_typecodes[] = {
    NPY_CFLOAT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_DOUBLE,
    NPY_CLONGDOUBLE, NPY_LONGDOUBLE
};

// The ufunc will not use the 'data' array.
static void *cabssq_data[] = {NULL, NULL, NULL, NULL, NULL, NULL};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Python extension module definitions.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static PyMethodDef CAbsSqMethods[] = {
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_cabssq",
    .m_doc = "Module that defines the cabssq function.",
    .m_size = -1,
    .m_methods = CAbsSqMethods
};


#define CABSSQ_DOCSTRING                    \
"cabssq(z, /, ...)\n"                       \
"\n"                                        \
"Squared absolute value for complex input."


PyMODINIT_FUNC PyInit__cabssq(void)
{
    PyObject *module;
    PyObject *cabssq_ufunc;
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

    // Create the cabssq ufunc object.
    nin = 1;
    nout = 1;
    cabssq_ufunc = PyUFunc_FromFuncAndData(cabssq_funcs, cabssq_data,
                                           cabssq_typecodes,
                                           CABSSQ_NLOOPS, nin, nout,
                                           PyUFunc_None,
                                           "cabssq", CABSSQ_DOCSTRING, 0);
    if (cabssq_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "cabssq",
                                (PyObject *) cabssq_ufunc);
    if (status == -1) {
        Py_DECREF(cabssq_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
