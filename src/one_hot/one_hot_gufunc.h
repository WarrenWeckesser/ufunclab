
#ifndef ONE_HOT_GUFUNC_H
#define ONE_HOT_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


//
// `one_hot_core_calc` is the C++ core function
// for the gufunc `one_hot` with signature '()->(n)'
//
template<typename T>
static void
one_hot_core_calc(
    npy_intp n,         // core dimension n
    T *p_x,             // pointer to x
    T *p_out,           // pointer to out, a strided 1-d array
    npy_intp out_stride
)
{
    if (out_stride == sizeof(T)) {
        memset(p_out, 0, n*sizeof(T));
    }
    else {
        for (npy_intp i = 0; i < n; ++i) {
            set(p_out, out_stride, i, static_cast<npy_intp>(0));
        }
    }
    if (0 <= *p_x && *p_x < n) {
        set(p_out, out_stride, *p_x, static_cast<npy_intp>(1));
    }
}

#endif
