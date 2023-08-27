#ifndef WJACCARD_GUFUNC_H
#define WJACCARD_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../util/strided.hpp"

//
// There are separate core functions for real types (float and double)
// and the integer types so the integer version can be implemented without
// checks for nan.
//

//
// Core loop function for wjaccard for float and double.
//
// T is the input array type.
// U is the output type.
//
template<typename T, typename U>
static void wjaccard_realtype_core(
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
    if (std::isnan(*p_x) || std::isnan(*p_y)) {
        *p_out = NAN;
        return;
    }
    U numer = std::min(*p_x, *p_y);
    U denom = std::max(*p_x, *p_y);
    for (npy_intp k = 1; k < n; ++k) {
        T xk = get(p_x, x_stride, k);
        T yk = get(p_y, y_stride, k);
        if (std::isnan(xk) || std::isnan(yk)) {
            *p_out = NAN;
            return;
        }
        numer += std::min(xk, yk);
        denom += std::max(xk, yk);
    }
    if (denom == 0) {
        if (numer == 0) {
            *p_out = NAN;
        }
        else {
            *p_out = -INFINITY;
        }
    }
    else {
        *p_out = (U) numer / denom;
    }
}

//
// Core loop function for wjaccard for integer types.
//
// T is the input array type.
// U is the output type.
//
template<typename T, typename U>
static void wjaccard_integer_core(
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
    U numer = std::min(*p_x, *p_y);
    U denom = std::max(*p_x, *p_y);
    for (npy_intp k = 1; k < n; ++k) {
        T xk = get(p_x, x_stride, k);
        T yk = get(p_y, y_stride, k);
        numer += std::min(xk, yk);
        denom += std::max(xk, yk);
    }
    if (denom == 0) {
        if (numer == 0) {
            *p_out = NAN;
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
