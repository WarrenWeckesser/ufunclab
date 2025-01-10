#ifndef WJACCARD_GUFUNC_H
#define WJACCARD_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


//
// Core loop function for wjaccard.
//
// T is the input array type.
// U is the output type.
//
template<typename T, typename U>
static void wjaccard_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_y,             // pointer to first element of y, a strided 1-d array with n elements
        npy_intp y_stride,  // stride (in bytes) for elements of y
        U *p_out            // pointer to out
)
{
    // The gufunc is configured to require n > 0, so it is safe to
    // dereference p_x and p_y without first checking that n > 0.
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(*p_x) || std::isnan(*p_y)) {
            *p_out = NPY_NAN;
            return;
        }
    }
    U numer = std::min(*p_x, *p_y);
    U denom = std::max(*p_x, *p_y);
    for (npy_intp k = 1; k < n; ++k) {
        T xk = get(p_x, x_stride, k);
        T yk = get(p_y, y_stride, k);
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(xk) || std::isnan(yk)) {
                *p_out = NPY_NAN;
                return;
            }
        }
        numer += std::min(xk, yk);
        denom += std::max(xk, yk);
    }
    if (denom == 0) {
        if (numer == 0) {
            *p_out = NPY_NAN;
        }
        else {
            *p_out = -INFINITY;
        }
    }
    else {
        *p_out = (U) numer / denom;
    }
}

#endif
