#ifndef CONVERT_TO_BASE_GUFUNC_H
#define CONVERT_TO_BASE_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"

template<typename T>
static void convert_to_base_core_calc(
        npy_intp n,          // core dimension n
        T *p_k,              // pointer to k
        T *p_base,           // pointer to base
        T *p_out,            // pointer to first element of out
        npy_intp out_stride  // stride (in bytes) for elements of out
)
{
    T k = *p_k;
    T base = *p_base;
    for (npy_intp i = 0; i < n; ++i) {
        set(p_out, out_stride, i, static_cast<T>(k % base));
        k = k / base;
    }
}

#endif
