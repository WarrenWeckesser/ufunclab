
#ifndef BINCOUNT_GUFUNC_H
#define BINCOUNT_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

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
    for (npy_intp i = 0; i < m; ++i) {
        set(p_out, out_stride, i, static_cast<U>(0));
    }
    for (npy_intp i = 0; i < n; ++i) {
        T k = get(p_x, x_stride, i);
        if (k >= 0 && k < m) {        
            set(p_out, out_stride, k, get(p_out, out_stride, k) + 1);
        }
    }
}

#endif
