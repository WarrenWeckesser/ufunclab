#ifndef HYSTERESIS_RELAY_GUFUNC_H
#define HYSTERESIS_RELAY_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T>
static void
hysteresis_relay_core(
        npy_intp n,                     // core dimension n
        T *p_x,                         // pointer to first element of x, a strided 1-d array with shape (n,)
        const npy_intp x_stride,        // stride (in bytes) of x
        T *p_low_threshold,
        T *p_high_threshold,
        T *p_low_value,
        T *p_high_value,
        T *p_init,
        T *p_out,                       // pointer to out, a strided 1-d array with shape (n,)
        const npy_intp out_stride       // stride (in bytes) of out
)
{
    T low_threshold = *p_low_threshold;
    T high_threshold = *p_high_threshold;
    T low_value = *p_low_value;
    T high_value = *p_high_value;
    T init = *p_init;

    for (npy_intp k = 0; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (x < low_threshold) {
            set(p_out, out_stride, k, low_value);
        }
        else if (x > high_threshold) {
            set(p_out, out_stride, k, high_value);
        }
        else {
            if (k == 0) {
                set(p_out, out_stride, k, init);
            }
            else {
                set(p_out, out_stride, k, get(p_out, out_stride, k - 1));
            }
        }
    }
}

#endif
