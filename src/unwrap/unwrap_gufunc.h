#ifndef UNWRAP_GUFUNC_H
#define UNWRAP_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


//
// Compute the "unwrapped" value of x1 given x0 and period.
//
template<typename T>
T unwrap1(T x0, T x1, T period)
{
    T delta = x1 - x0;
    int q = (int)(delta/period);
    T rem = std::fmod(delta, period);
    if (rem < 0) {
        rem += period;
        q -= 1;
    }
    T frac = std::fabs(rem)/period;
    if (frac >= 0.5) {
        q += 1;
    }
    return x1 - q * period;
}

template<typename T>
static void
unwrap_core(
    npy_intp n,                     // core dimension n
    T *p_x,                         // pointer to first element of x, a strided 1-d array with shape (n,)
    const npy_intp x_stride,        // stride (in bytes) of x
    T *p_period,                    // pointer to the 'period' parameter
    T *p_out,                       // pointer to out, a strided 1-d array with shape (n,)
    const npy_intp out_stride       // stride (in bytes) of out
)
{
    if (n < 1) {
        // Nothing to do...
        return;
    }
    T x0 = *p_x;
    T period = *p_period;
    p_out[0] = x0;
    for (npy_intp k = 1; k < n; ++k) {
        T x1 = get(p_x, x_stride, k);
        if (!isfinite(x1)) {
            // inf or nan; output from here onwards is nan.
            for (npy_intp k1 = k; k1 < n; ++k1) {
                set(p_out, out_stride, k, static_cast<T>(NPY_NAN));
            }
            break;
        }
        T x1prime = unwrap1(x0, x1, period);
        set(p_out, out_stride, k, x1prime);
        x0 = x1prime;
    }
}

#endif
