//
// log1p_ufunc.c
//
// This C extension module defines the ufuncs `log1p_theorem4` and
// `log1p_doubledouble`.  They are implemented for the type np.complex128
// only (type signature is `D->D`).
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <complex.h>
#include <math.h>

#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

//
// We are using C99, so define our own version of CMPLX(x, y)
// if it is not provided by the compiler.
// After switching to C11, this macro must be removed.
//
#ifndef CMPLX
#define CMPLX(x, y) ((x) + (y)*_Complex_I)
#endif

//
// Compute log1p(z) using an implementation that is based on
// Theorem 4 of David Goldberg's paper "What every computer
// scientist should know about floating-point arithmetic".
// The quality of this implementation depends on the quality
// of the complex log function clog(z) provided in the math
// library.
//
// This function assumes that neither part of z is nan.
//
static double _Complex
log1p_theorem4(double _Complex z)
{
    double _Complex w;

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
    return w;
}

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

//
// Dekker splitting.  See, for example, Theorem 1 of
//
//   Seppa Linnainmaa. Software for Double-Precision Floating-Point
//   Computations, ACM Transactions on Mathematical Software, Vol 7, No 3,
//   September 1981, pages 272-283.
//
// See also
//
//   Claude-Pierre Jeannerod, Jean-Michel Muller, Paul Zimmermann.
//   On various ways to split a floating-point number. ARITH 2018 - 25th
//   IEEE Symposium on Computer Arithmetic, Jun 2018, Amherst (MA),
//   United States. pp.53-60, 10.1109/ARITH.2018.8464793. hal-01774587v2
//
static void
split(double x, doubledouble_t *out)
{
    double t = ((1 << 27) + 1)*x;
#ifdef FP_FAST_FMA
    out->upper = fma(-(1 << 27), x, t);
#else
    out->upper = t - (t - x);
#endif
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

//
// Implement log1p(z) using double-double near the unit circle |z + 1| = 1
// to avoid loss of precision.
//
// This function assumes that neither part of z is nan.
//

static double _Complex
log1p_doubledouble(double _Complex z)
{
    double lnr;

    double x = creal(z);
    double y = cimag(z);
    if (x > -2.2 && x < 0.2 && y > -1.2 && y < 1.2) {
        // This nested `if` condition *should* be part of the outer
        // `if` condition, but clang on Mac OS 13 doesn't seem to
        // short-circuit the evaluation correctly and generates an
        // overflow when x and y are sufficiently large.
        if (fabs(x*(2.0 + x) + y*y) < 0.4) {
            // The input is close to the unit circle centered at -1+0j.
            // Use double-double to evaluate the real part of the result.
            lnr = 0.5*log1p(foo(x, y));
        }
        else {
            lnr = log(hypot(x + 1, y));
        }
    }
    else {
        lnr = log(hypot(x + 1, y));
    }
    return CMPLX(lnr, atan2(y, x + 1.0));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loop". The type signatures of the log1p ufunc are
//     >>> log1p.types
//     ['D->D']
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void
log1p_theorem4_D_D_loop(char **args, const npy_intp *dimensions,
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
            w = log1p_theorem4(z);
        }
        *((double _Complex *) out) = w;
    }
}

static void
log1p_doubledouble_D_D_loop(char **args, const npy_intp *dimensions,
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
            w = log1p_doubledouble(z);
        }
        *((double _Complex *) out) = w;
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays log1p_typecodes and log1p_data.
PyUFuncGenericFunction log1p_theorem4_funcs[] = {
    (PyUFuncGenericFunction) &log1p_theorem4_D_D_loop,
};

PyUFuncGenericFunction log1p_doubledouble_funcs[] = {
    (PyUFuncGenericFunction) &log1p_doubledouble_D_D_loop,
};


// Same for theorem4 and doubledouble
#define LOG1P_NLOOPS \
    (sizeof(log1p_theorem4_funcs) / sizeof(log1p_theorem4_funcs[0]))

// These are the input and return type codes for the inner loops.
// Same for theorem4 and doubledouble
static char log1p_typecodes[] = {
    NPY_CDOUBLE, NPY_CDOUBLE,
};

// The ufunc will not use the 'data' array.
// Same for theorem4 and doubledouble
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


#define LOG1P_THEOREM4_DOCSTRING \
"log1p_theorem4(z, /, ...)\n"                                           \
"\n"                                                                    \
"log1p(z) for complex z.\n"                                             \
"\n"                                                                    \
"This implementation is based on Theorem 4 of Golberg's paper\n"        \
"'What every computer scientists needs to know about floating-point"    \
"arithmetic'.\n"                                                        \
"\n"

#define LOG1P_DOUBLEDOUBLE_DOCSTRING \
"log1p_doubledouble(z, /, ...)\n"                                       \
"\n"                                                                    \
"log1p(z) for complex z.\n"                                             \
"\n"                                                                    \
"This implementation uses double-double values for some intermediate\n" \
"calculations to avoid loss of precision near the unit circle\n"        \
"|z + 1| = 1.\n"                                                        \
"\n"


PyMODINIT_FUNC PyInit__log1p(void)
{
    PyObject *module;
    PyObject *log1p_theorem4_ufunc;
    PyObject *log1p_doubledouble_ufunc;
    int nin, nout;
    int status;

    module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    import_array();
    import_umath();

    // Create the log1p_theorem4 ufunc object.
    nin = 1;
    nout = 1;
    log1p_theorem4_ufunc =
        PyUFunc_FromFuncAndData(log1p_theorem4_funcs, log1p_data,
                                log1p_typecodes,
                                LOG1P_NLOOPS, nin, nout,
                                PyUFunc_None,
                                "log1p_theorem4", LOG1P_THEOREM4_DOCSTRING, 0);
    if (log1p_theorem4_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "log1p_theorem4",
                                (PyObject *) log1p_theorem4_ufunc);
    if (status == -1) {
        Py_DECREF(log1p_theorem4_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    // Create the log1p_doubledouble ufunc object.
    nin = 1;
    nout = 1;
    log1p_doubledouble_ufunc =
        PyUFunc_FromFuncAndData(log1p_doubledouble_funcs, log1p_data,
                                log1p_typecodes,
                                LOG1P_NLOOPS, nin, nout,
                                PyUFunc_None,
                                "log1p_doubledouble", LOG1P_DOUBLEDOUBLE_DOCSTRING, 0);
    if (log1p_doubledouble_ufunc == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    // Add the ufunc to the module.
    status = PyModule_AddObject(module, "log1p_doubledouble",
                                (PyObject *) log1p_doubledouble_ufunc);
    if (status == -1) {
        Py_DECREF(log1p_doubledouble_ufunc);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
