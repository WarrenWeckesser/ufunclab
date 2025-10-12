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
softmax_core(
    npy_intp n,                     // core dimension n
    T *p_x,                         // pointer to first element of x, a strided 1-d array with shape (n,)
    const npy_intp x_stride,        // stride (in bytes) of x
    T *p_out,                       // pointer to out, a strided 1-d array with shape (n,)
    const npy_intp out_stride       // stride (in bytes) of out
)
{
    // Get the maximum, and count positive infs.
    npy_intp nposinf = 0;
    npy_intp posinf_index;
    bool has_nan = false;
    T xmax;
    for (npy_intp k = 0; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (std::isnan(x)) {
            has_nan = true;
            break;
        }
        if (std::isinf(x) && x > 0) {
            ++nposinf;
            posinf_index = k;
        }
        if (k == 0 || x > xmax) {
            xmax = x;
        }
    }
    if (has_nan || nposinf > 1) {
        // x contains nan, or there is more than one positive inf, so set the
        // result to all nan.
        for (npy_intp k = 0; k < n; ++k) {
            set(p_out, out_stride, k, static_cast<T>(NPY_NAN));
        }
    }
    else if (nposinf == 1) {
        for (npy_intp k = 0; k < n; ++k) {
            set(p_out, out_stride, k, static_cast<T>(0));
        }
        set(p_out, out_stride, posinf_index, static_cast<T>(1));
    }
    else {
        T expsum = 0.0;
        for (npy_intp k = 0; k < n; ++k) {
            T x = get(p_x, x_stride, k);
            expsum += std::exp(x - xmax);
        }
        for (npy_intp k = 0; k < n; ++k) {
            T x = get(p_x, x_stride, k);
            set(p_out, out_stride, k, std::exp(x - xmax)/expsum);
        }
    }
}

#endif  // UFUNCLAB_SOFTMAX_GUFUNC_H
