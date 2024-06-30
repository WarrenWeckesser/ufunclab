#ifndef CORR_GUFUNC_H
#define CORR_GUFUNC_H

#include <cmath>

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T, typename U>
static U
strided_mean(
        npy_intp n,           // core dimension n
        T *p_x,               // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride     // stride (in bytes) for elements of x
)
{
    U mean = 0.0;
    U c1 = 0.0;
    for (npy_intp k = 0; k < n; ++k) {
        U y1, t1;
        T xk = get(p_x, x_stride, k);
        U delta = xk - mean;
        y1 = delta/(k + 1) - c1;
        t1 = mean + y1;
        c1 = (t1 - mean) - y1;
        mean = t1;
    }
    return mean;
}


//
// Compute max_i | x_i - xmean |
//
template<typename T, typename U>
static U
strided_delta_maxabs(npy_intp n, T *p_x, npy_intp x_stride, U xmean)
{
    // This code assumes n > 0.
    U maxabs = fabs(*p_x - xmean);
    for (npy_intp k = 1; k < n; ++k) {
        U absxk = fabs(get(p_x, x_stride, k) - xmean);
        if (absxk > maxabs) {
            maxabs = absxk;
        }
    }
    return maxabs;
}

//
// Compute norm(x - xmean)
//
// deltax_max is expected to be max_i | x_i - xmean |
//
// It is assumed that delta_max is not 0.
//
template<typename T, typename U>
static U
strided_delta_norm(npy_intp n, T *p_x, npy_intp x_stride, U xmean, U deltax_max)
{
    U sumsq = 0;
    U c = 0;
    for (npy_intp k = 0; k < n; ++k) {
        U z2 = std::pow((get(p_x, x_stride, k) - xmean) / deltax_max, 2.0);
        U y = z2 - c;
        U t = sumsq + y;
        c = (t - sumsq) - y;
        sumsq = t;
    }
    return deltax_max * std::sqrt(sumsq);
}

//
// XXX D.R.Y.  Fix repeated code in pearson_corr_int_core and
// pearson_core_core.
//

template<typename T, typename U>
static void pearson_corr_int_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_y,             // pointer to first element of y, a strided 1-d array with n elements
        npy_intp y_stride,  // stride (in bytes) for elements of y
        U *p_out            // pointer to out
)
{
    if (n == 1) {
        *p_out = NPY_NAN;
        return;
    }
    if (n == 2) {
        T x0 = *p_x;
        T x1 = get(p_x, x_stride, 1);
        if (x0 == x1) {
            *p_out = NPY_NAN;
            return;
        }
        T y0 = *p_y;
        T y1 = get(p_y, y_stride, 1);
        if (y0 == y1) {
            *p_out = NPY_NAN;
            return;
        }
        if (((x0 < x1) && (y0 < y1)) || ((x0 > x1) && (y0 > y1))) {
            *p_out = 1;
        }
        else {
            *p_out = -1;
        }
        return;
    }

    U xmean = strided_mean<T, U>(n, p_x, x_stride);
    U x_delta_maxabs = strided_delta_maxabs<T, U>(n, p_x, x_stride, xmean);
    if (x_delta_maxabs == 0) {
        *p_out = NPY_NAN;
        return;
    }
    U xnorm = strided_delta_norm<T, U>(n, p_x, x_stride, xmean, x_delta_maxabs);

    U ymean = strided_mean<T, U>(n, p_y, y_stride);
    U y_delta_maxabs = strided_delta_maxabs<T, U>(n, p_y, y_stride, ymean);
    if (y_delta_maxabs == 0) {
        *p_out = NPY_NAN;
        return;
    }
    U ynorm = strided_delta_norm<T, U>(n, p_y, y_stride, ymean, y_delta_maxabs);

    U r = 0.0;
    for (npy_intp k = 0; k < n; ++k) {
        U xk = (get(p_x, x_stride, k) - xmean) / xnorm;
        U yk = (get(p_y, y_stride, k) - ymean) / ynorm;
        r += xk*yk;
    }
    *p_out = r;
}


template<typename T, typename U>
static void pearson_corr_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_y,             // pointer to first element of y, a strided 1-d array with n elements
        npy_intp y_stride,  // stride (in bytes) for elements of y
        U *p_out            // pointer to out
)
{
    if (n == 1) {
        *p_out = NPY_NAN;
        return;
    }
    if (n == 2) {
        T x0 = *p_x;
        T x1 = get(p_x, x_stride, 1);
        if (!std::isfinite(x0) || !std::isfinite(x1) || (x0 == x1)) {
            *p_out = NPY_NAN;
            return;
        } 
        T y0 = *p_y;
        T y1 = get(p_y, y_stride, 1);
        if (!std::isfinite(y0) || !std::isfinite(y1) || (y0 == y1)) {
            *p_out = NPY_NAN;
            return;
        }
        if (((x0 < x1) && (y0 < y1)) || ((x0 > x1) && (y0 > y1))) {
            *p_out = 1;
        }
        else {
            *p_out = -1;
        }
        return;
    }

    U xmean = strided_mean<T, U>(n, p_x, x_stride);
    U x_delta_maxabs = strided_delta_maxabs<T, U>(n, p_x, x_stride, xmean);
    if (x_delta_maxabs == 0) {
        *p_out = NPY_NAN;
        return;
    }
    U xnorm = strided_delta_norm<T, U>(n, p_x, x_stride, xmean, x_delta_maxabs);

    U ymean = strided_mean<T, U>(n, p_y, y_stride);
    U y_delta_maxabs = strided_delta_maxabs<T, U>(n, p_y, y_stride, ymean);
    if (y_delta_maxabs == 0) {
        *p_out = NPY_NAN;
        return;
    }
    U ynorm = strided_delta_norm<T, U>(n, p_y, y_stride, ymean, y_delta_maxabs);

    U r = 0.0;
    for (npy_intp k = 0; k < n; ++k) {
        U xk = (get(p_x, x_stride, k) - xmean) / xnorm;
        U yk = (get(p_y, y_stride, k) - ymean) / ynorm;
        r += xk*yk;
    }
    *p_out = r;
}

#endif
