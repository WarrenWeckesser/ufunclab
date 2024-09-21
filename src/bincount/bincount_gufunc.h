
#ifndef BINCOUNT_GUFUNC_H
#define BINCOUNT_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <complex>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

#include "../src/util/strided.hpp"


//
// `bincount_core_calc` is the C++ core function
// for the gufunc `bincount` with signature '(n)->(m)'
//
template<typename T, typename U>
static void
bincount_core_calc(
    npy_intp n,         // core dimension n
    npy_intp m,         // core dimension m
    T *p_x,             // pointer to x
    npy_intp x_stride,
    U *p_out,           // pointer to out, a strided 1-d array
    npy_intp out_stride
)
{
    // Note that the output array is not initialized to 0.
    // This allows repeated calls to accumulate results.

    for (npy_intp i = 0; i < n; ++i) {
        T k = get(p_x, x_stride, i);
        if (k >= 0 && static_cast<npy_intp>(k) < m) {
            set(p_out, out_stride, k, get(p_out, out_stride, k) + 1);
        }
    }
}

//
// `bincountw_core_calc` is the C++ core function
// for the gufunc `bincountw` with signature '(n),(n)->(m)'
//
template<typename T, typename W>
static void
bincountw_core_calc(
    npy_intp n,         // core dimension n
    npy_intp m,         // core dimension m
    T *p_x,             // pointer to x
    npy_intp x_stride,
    W *p_w,             // pointer to w
    npy_intp w_stride,
    W *p_out,           // pointer to out, a strided 1-d array
    npy_intp out_stride
)
{
    // Note that the output array is not initialized to 0.
    // This allows repeated calls to accumulate results.

    for (npy_intp i = 0; i < n; ++i) {
        T k = get(p_x, x_stride, i);
        if (k >= 0 && static_cast<npy_intp>(k) < m) {
            W w = get(p_w, w_stride, i);
            set<W>(p_out, static_cast<ptrdiff_t>(out_stride),
                   static_cast<ptrdiff_t>(k),
                   get<W>(p_out, static_cast<ptrdiff_t>(out_stride),
                          static_cast<ptrdiff_t>(k)) + w);
        }
    }
}


void static inline
npy_complex_add_inplace(npy_cfloat *a, npy_cfloat b)
{
    *((float *) a) += npy_crealf(b);
    *((float *) a + 1) += npy_cimagf(b);
}

void static inline
npy_complex_add_inplace(npy_cdouble *a, npy_cdouble b)
{
    *((double *) a) += npy_creal(b);
    *((double *) a + 1) += npy_cimag(b);
}

//
// `bincountw_complex_core_calc` is the C++ core function
// for the gufunc `bincountw` with signature '(n),(n)->(m)'
// that handles complex weights.
//
template<typename T, typename W>
static void
bincountw_complex_core_calc(
    npy_intp n,         // core dimension n
    npy_intp m,         // core dimension m
    T *p_x,             // pointer to x
    npy_intp x_stride,
    W *p_w,             // pointer to w
    npy_intp w_stride,
    W *p_out,           // pointer to out, a strided 1-d array
    npy_intp out_stride
)
{
    // Note that the output array is not initialized to 0.
    // This allows repeated calls to accumulate results.

    for (npy_intp i = 0; i < n; ++i) {
        T k = get(p_x, x_stride, i);
        if (k >= 0 && static_cast<npy_intp>(k) < m) {
            W w = get(p_w, w_stride, i);
            W *p = (W *)((char *)p_out + k*out_stride);
            npy_complex_add_inplace(p, w);
        }
    }
}

#endif
