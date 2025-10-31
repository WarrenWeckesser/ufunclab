//
// log1p_ufunc.c
//
// This C extension module defines the ufuncs `log1p_theorem4` and
// `log1p_doubledouble`.  They are implemented for the types np.complex64
// and np.complex128 only.
//

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <complex.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

//
// We are using C99, so define our own version of CMPLX(x, y)
// if it is not provided by the compiler.
// After switching to C11, this macro must be removed.
//
#ifndef CMPLX
#ifdef _MSC_VER
#define CMPLX(x, y) _Cbuild(x, y)
#define CMPLXF(x, y) _FCbuild(x, y)
#else
#define CMPLX(x, y) ((x) + (y)*_Complex_I)
#define CMPLXF(x, y) ((x) + (y)*_Complex_I)
#endif
#endif

#ifdef _MSC_VER
#define complex_float _Fcomplex
#define complex_double _Dcomplex
#else
#define complex_float float _Complex
#define complex_double double _Complex
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
#ifdef _MSC_VER

#define TOPY(z) (*(Py_complex *) &(z))
#define TOCMPLX(z) (*(complex_double *) &(z))

static complex_float
log1pf_theorem4(complex_float z)
{
    complex_float w;

    complex_float u = CMPLXF(crealf(z) + 1.0f, cimagf(z));
    if (crealf(u) == 1.0f && cimagf(u) == 0.0f) {
        // z + 1 == 1
        w = z;
    }
    else {
        if (crealf(u) - 1.0f == crealf(z)) {
            // u - 1 == z
            w = clogf(u);
        }
        else {
            // w = clog(u) * (z / (u - 1.0));
            complex_float um1 = CMPLXF(crealf(u) - 1.0f, cimagf(u));
            complex_float logu = clogf(u);
            // Microsoft C doesn't implement complex division ($&@$*@!),
            // so use Python's functions.  We have to upcast to double.
            complex_double logu_d = CMPLX((double) crealf(logu), (double) cimagf(logu));
            complex_double z_d = CMPLX((double) crealf(z), (double) cimagf(z));
            complex_double um1_d = CMPLX((double) crealf(um1), (double) cimagf(um1));
            Py_complex v_d = _Py_c_prod(TOPY(logu_d), _Py_c_quot(TOPY(z_d), TOPY(um1_d)));
            w = CMPLXF((float) v_d.real, (float) v_d.imag);
        }
    }
    return w;
}

static complex_double
log1p_theorem4(complex_double z)
{
    complex_double w;

    complex_double u = CMPLX(creal(z) + 1.0, cimag(z));
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
            // w = clog(u) * (z / (u - 1.0));
            complex_double um1 = CMPLX(creal(u) - 1.0, cimag(u));
            complex_double logu = clog(u);
            // Microsoft C doesn't implement complex division ($&@$*@!),
            // so use Python's functions.
            Py_complex v = _Py_c_prod(TOPY(logu), _Py_c_quot(TOPY(z), TOPY(um1)));
            w = TOCMPLX(v);
        }
    }
    return w;
}

#else

static complex_float
log1pf_theorem4(complex_float z)
{
    complex_float w;

    complex_float u = z + 1.0f;
    if (crealf(u) == 1.0f && cimagf(u) == 0.0f) {
        // z + 1 == 1
        w = z;
    }
    else {
        if (crealf(u) - 1.0f == crealf(z)) {
            // u - 1 == z
            w = clogf(u);
        }
        else {
            w = clogf(u) * (z / (u - 1.0f));
        }
    }
    return w;
}

static complex_double
log1p_theorem4(complex_double z)
{
    complex_double w;

    complex_double u = z + 1.0;
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

#endif

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

//
// As the name makes clear, this function computes x**2 + 2*x + y**2.
// It uses doubledouble_t for the intermediate calculations.
//
// The function is used in log1p_doubledouble() to avoid the loss of
// precision that can occur in the expression when x**2 + y**2 ≈ -2*x.
//
static double
xsquared_plus_2x_plus_ysquared(double x, double y)
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
static complex_double
log1p_doubledouble(complex_double z)
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
            // This is equivalent to 0.5*log((1+x)**2 + y**2), since
            //   log((1 + x)**2 + y**2) = log(1 + 2*x + x**2 + y**2)
            //                          = log1p(x**2 + 2*x + y**2)
            lnr = 0.5*log1p(xsquared_plus_2x_plus_ysquared(x, y));
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

//
// Implement log1pf(z), upgrading to double precision near the unit
// circle |z + 1| = 1  to avoid loss of precision.
//
// This function assumes that neither part of z is nan.
//
static complex_float
log1pf_doubledouble(complex_float z)
{
    float lnr;

    float x = crealf(z);
    float y = cimagf(z);
    if (x > -2.2f && x < 0.2f && y > -1.2f && y < 1.2f) {
        // This nested `if` condition *should* be part of the outer
        // `if` condition, but clang on Mac OS 13 doesn't seem to
        // short-circuit the evaluation correctly and generates an
        // overflow when x and y are sufficiently large.
        if (fabsf(x*(2.0f + x) + y*y) < 0.4f) {
            // The input is close to the unit circle centered at -1+0j.
            // Use double-double to evaluate the real part of the result.
            // This is equivalent to 0.5*log((1+x)**2 + y**2), since
            //   log((1 + x)**2 + y**2) = log(1 + 2*x + x**2 + y**2)
            //                          = log1p(x**2 + 2*x + y**2)
            double x_d = (double) x;
            double y_d = (double) y;
            lnr = (float) (0.5*log1p(x_d*x_d + 2*x_d + y_d*y_d));
        }
        else {
            lnr = logf(hypotf(x + 1.0f, y));
        }
    }
    else {
        lnr = log(hypotf(x + 1.0f, y));
    }
    return CMPLXF(lnr, atan2f(y, x + 1.0f));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// The ufunc "inner loop". The type signatures of the log1p ufunc are
//     >>> log1p.types
//     ['F->F', 'D->D']
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

static void
log1p_theorem4_F_F_loop(char **args, const npy_intp *dimensions,
                        const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        complex_float w;
        complex_float z = *(complex_float *) in;

        if (isnan(crealf(z)) || isnan(cimagf(z))) {
            w = CMPLXF(NPY_NAN, NPY_NAN);
        }
        else {
            w = log1pf_theorem4(z);
        }
        *((complex_float *) out) = w;
    }
}

static void
log1p_theorem4_D_D_loop(char **args, const npy_intp *dimensions,
                        const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        complex_double w;
        complex_double z = *(complex_double *) in;

        if (isnan(creal(z)) || isnan(cimag(z))) {
            w = CMPLX(NPY_NAN, NPY_NAN);
        }
        else {
            w = log1p_theorem4(z);
        }
        *((complex_double *) out) = w;
    }
}

static void
log1p_doubledouble_F_F_loop(char **args, const npy_intp *dimensions,
                            const npy_intp* steps, void* data)
{
    char *in = args[0];
    char *out = args[1];
    npy_intp in_step = steps[0];
    npy_intp out_step = steps[1];

    for (npy_intp i = 0; i < dimensions[0]; ++i, in += in_step, out += out_step) {
        complex_float w;
        complex_float z = *(complex_float *) in;

        if (isnan(crealf(z)) || isnan(cimagf(z))) {
            w = CMPLXF(NPY_NAN, NPY_NAN);
        }
        else {
            w = log1pf_doubledouble(z);
        }
        *((complex_float *) out) = w;
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
        complex_double w;
        complex_double z = *(complex_double *) in;

        if (isnan(creal(z)) || isnan(cimag(z))) {
            w = CMPLX(NPY_NAN, NPY_NAN);
        }
        else {
            w = log1p_doubledouble(z);
        }
        *((complex_double *) out) = w;
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// ufunc configuration data.
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// This array of loop function pointers will be passed to PyUFunc_FromFuncAndData,
// along with the arrays log1p_typecodes and log1p_data.
PyUFuncGenericFunction log1p_theorem4_funcs[] = {
    (PyUFuncGenericFunction) &log1p_theorem4_F_F_loop,
    (PyUFuncGenericFunction) &log1p_theorem4_D_D_loop,
};

PyUFuncGenericFunction log1p_doubledouble_funcs[] = {
    (PyUFuncGenericFunction) &log1p_doubledouble_F_F_loop,
    (PyUFuncGenericFunction) &log1p_doubledouble_D_D_loop,
};


// Same for theorem4 and doubledouble
#define LOG1P_NLOOPS \
    (sizeof(log1p_theorem4_funcs) / sizeof(log1p_theorem4_funcs[0]))

// These are the input and return type codes for the inner loops.
// Same for theorem4 and doubledouble
static char log1p_typecodes[] = {
    NPY_CFLOAT, NPY_CFLOAT,
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
    .m_doc = "Module that defines the log1p_theorem4 and log1p_doubledouble functions.",
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

    if (PyArray_ImportNumPyAPI() < 0) {
        return NULL;
    }
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
