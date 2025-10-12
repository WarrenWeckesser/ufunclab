#ifndef UFUNCLAB_SOFTMAX_GUFUNC_H
#define UFUNCLAB_SOFTMAX_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"

template<typename T>
static void
fillnan1d_core(
    npy_intp n,                     // core dimension n
    T *p_x,                         // pointer to first element of x, a strided 1-d array with shape (n,)
    const npy_intp x_stride,        // stride (in bytes) of x
    T *p_out,                       // pointer to out, a strided 1-d array with shape (n,)
    const npy_intp out_stride       // stride (in bytes) of out
)
{
    // Find first non-nan.
    npy_intp k = 0;
    while (k < n) {
        T x = get(p_x, x_stride, k);
        if (!std::isnan(x)) {
            break;
        }
        ++k;
    }
    if (k == n) {
        // All values are nan.  Fill output with nan.
        for (npy_intp i = 0; i < n; ++i) {
            set(p_out, out_stride, i, static_cast<T>(NPY_NAN));
        }
        return;
    }

    // Replace any initial nan values with the first non-nan value.
    T left_value = get(p_x, x_stride, k);
    npy_intp left_index = k;
    for (npy_intp i = 0; i <= k; ++i) {
        set(p_out, out_stride, i, left_value);
    }
    for (npy_intp k = left_index + 1; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (!std::isnan(x)) {
            set(p_out, out_stride, k, x);
            if (left_index != k - 1) {
                long double slope = ((long double) x - (long double) left_value)/(k - left_index);
                for (npy_intp i = 1; i < k - left_index; ++i) {
                    set(p_out, out_stride, left_index + i, static_cast<T>(left_value + i*slope));
                }
            }
            left_value = x;
            left_index = k;
        }
    }
    if (left_index != n) {
        // Handle nan values at the right end.
        for (npy_intp i = left_index + 1; i < n; ++i) {
            set(p_out, out_stride, i, left_value);
        }
    }
}

#endif  // UFUNCLAB_SOFTMAX_GUFUNC_H
