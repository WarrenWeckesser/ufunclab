//
// log1p_ufunc.c
//
// This C extension module defines the `log1p` ufunc.  `log1p`
// is implemented for the type np.complex128 only (type signature
// is `D->D`).
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Double-double functions used by log1p to avoid loss of precision
// when the complex input z is close to the unit circle centered at -1+0j.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

struct _doubledouble_t {
    double upper;
    double lower;
};

typedef struct _doubledouble_t doubledouble_t;

static void
two_sum_quick(double x, double y, doubledouble_t *out)
{
    double r = x + y;
    double e = y - (r - x);
    out->upper = r;
    out->lower = e;
}

static void
two_sum(double x, double y, doubledouble_t *out)
{
    double s = x + y;
    double v = s - x;
    double e = (x - (s - v)) + (y - v);
    out->upper = s;
    out->lower = e;
}

static void
double_sum(const doubledouble_t x, const doubledouble_t y,
           doubledouble_t *out)
{
    two_sum(x.upper, y.upper, out);
    out->lower += x.lower + y.lower;
    two_sum_quick(out->upper, out->lower, out);
}

static void
split(double x, doubledouble_t *out)
{
    double t = ((1 << 27) + 1)*x;
    out->upper = t - (t - x);
    out->lower = x - out->upper;
}

static void
square(double x, doubledouble_t *out)
{
    doubledouble_t xsplit;
    out->upper = x*x;
    split(x, &xsplit);
    out->lower = (xsplit.upper*xsplit.upper - out->upper)
                  + 2*xsplit.upper*xsplit.lower
                  + xsplit.lower*xsplit.lower;
}

static double
foo(double x, double y)
{
    doubledouble_t x2, y2, twox, sum1, sum2;

    square(x, &x2);
    square(y, &y2);
    twox.upper = 2*x;
    twox.lower = 0.0;
    double_sum(x2, twox, &sum1);
    double_sum(sum1, y2, &sum2);
    return sum2.upper;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loop". The type signatures of the log1p ufunc are
//     >>> log1p.types
//     ['D->D']
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void log1p_D_D_loop(char **args, const npy_intp *dimensions,
                           const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        double lnr;
        double x = (double) *(double *)in;
        double y = (double) *(double *)(in + sizeof(double));

        if (isnan(x) || isnan(y)) {
            *((double *) out) = NAN;
            *((double *) (out + sizeof(double))) = NAN;            
        }
        else {
            if (x > -2.2 && x < 0.2 && y > -1.2 && y < 1.2
                    && fabs(x*(2.0 + x) + y*y) < 0.1) {
                // The input is close to the unit circle centered at -1+0j.
                // Use double-double to evaluate the real part of the result.
                lnr = 0.5*log1p(foo(x, y));
            }
            else {
                lnr = log(hypot(x + 1, y));
            }
            *((double *) out) = lnr;
            *((double *) (out + sizeof(double))) = atan2(y, x + 1.0);
        }
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
