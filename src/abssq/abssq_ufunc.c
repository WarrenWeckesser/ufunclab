//
// abssq_ufunc.c
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

static void abssq_f_f_loop(char **args, const npy_intp *dimensions,
                           const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        float x = (float) *(float *)in;
        *((float *) out) = x*x;
    }
}

static void abssq_d_d_loop(char **args, const npy_intp *dimensions,
                           const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double x = (double) *(double *)in;
        *((double *) out) = x*x;
    }
}

static void abssq_g_g_loop(char **args, const npy_intp *dimensions,
                           const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        long double x = (long double) *(long double *)in;
        *((long double *) out) = x*x;
    }
}

static void abssq_F_f_loop(char **args, const npy_intp *dimensions,
                           const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        float x = (float) *(float *)in;
        float y = (float) *(float *)(in + sizeof(float));
        *((float *) out) = x*x + y*y;
    }
}

static void abssq_D_d_loop(char **args, const npy_intp *dimensions,
                           const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double x = (double) *(double *)in;
        double y = (double) *(double *)(in + sizeof(double));
        *((double *) out) = x*x + y*y;
    }
}

static void abssq_G_g_loop(char **args, const npy_intp *dimensions,
                           const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        long double x = (long double) *(long double *)in;
        long double y = (long double) *(long double *)(in + sizeof(long double));
        *((long double *) out) = x*x + y*y;
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays abssq_typecodes and abssq_data.
PyUFuncGenericFunction abssq_funcs[] = {
    (PyUFuncGenericFunction) &abssq_f_f_loop,
    (PyUFuncGenericFunction) &abssq_d_d_loop,
    (PyUFuncGenericFunction) &abssq_g_g_loop,
    (PyUFuncGenericFunction) &abssq_F_f_loop,
    (PyUFuncGenericFunction) &abssq_D_d_loop,
    (PyUFuncGenericFunction) &abssq_G_g_loop
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

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_abssq",
    .m_doc = "Module that defines the abssq function.",
    .m_size = -1,
    .m_methods = AbsSqMethods
};


#define ABSSQ_DOCSTRING     \
"abssq(z, /, ...)\n"        \
"\n"                        \
"Squared absolute value."


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

    import_array();
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
