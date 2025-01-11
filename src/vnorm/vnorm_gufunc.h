
#ifndef VNORM_GUFUNC_H
#define VNORM_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <complex>
#include <cmath>
#include <algorithm>
#include <type_traits>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


//
// Create an abs function that we can specialize for the numpy complex types.
//

template<typename T>
static inline T
my_abs(T x)
{
    if constexpr (std::is_unsigned_v<T>) {
        return x;
    }
    else {
        return std::abs(x);
    }
}

static inline float
my_abs(npy_cfloat z)
{
    std::complex<float> zs{npy_crealf(z), npy_cimagf(z)};
    return std::abs(zs);
}

static inline double
my_abs(npy_cdouble z)
{
    std::complex<double> zs{npy_creal(z), npy_cimag(z)};
    return std::abs(zs);
}

static inline long double
my_abs(npy_clongdouble z)
{
    std::complex<long double> zs{npy_creall(z), npy_cimagl(z)};
    return std::abs(zs);
}


//
// `vnorm_core_calc`, the C++ core function
// for the gufunc `vnorm` with signature '(n),()->()'
// for types ['ff->f', 'dd->d', 'gg->g', 'Ff->f', 'Dd->d', 'Gg->g'].
//
template<typename T, typename U>
static void vnorm_core_calc(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        U *p_order,         // pointer to order
        U *p_out            // pointer to out
)
{
    U maxabsx = 0;
    U order = *p_order;
    if (order <= 0) {
        p_out[0] = NPY_NAN;
        return;
    }

    for (int k = 0; k < n; ++k) {
        T current_x = get(p_x, x_stride, k);
        U abs_x = my_abs(current_x);
        if (abs_x > maxabsx) {
            maxabsx = abs_x;
        }
    }
    if (maxabsx == 0) {
        p_out[0] = 0;
    }
    else {
        U sum = 0;
        for (int k = 0; k < n; ++k) {
            T current_x = get(p_x, x_stride, k);
            U abs_x = my_abs(current_x);
            if (isinf(order)) {
                sum = std::max(sum, abs_x);
            }
            else if (order == 1) {
                sum += abs_x;
            }
            else {
                sum += pow(abs_x/maxabsx, order);
            }
        }
        if (isinf(order) | (order == 1)) {
            p_out[0] = sum;
        }
        else {
            p_out[0] = maxabsx * pow(sum, 1/order);
        }
    }
}

template<typename T>
T rms_core_contig(npy_intp n, T *p_x)
{
    T sum = 0;
    for (npy_intp k = 0; k < n; ++k) {
        T x = p_x[k];
        sum += x*x;
    }
    return sqrt(sum/n);
}

//
// `rms_core_calc`, the C++ core function
// for the gufunc `vnorm` with signature '(n)->()'
// for types ['f->f', 'd->d', 'g->g'].
//
// XXX This calculation can overflow if elements
// of the input array are large.
//
template<typename T>
static void rms_core_calc(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_out            // pointer to out
)
{
    if (x_stride == sizeof(T)) {
        *p_out = rms_core_contig(n, p_x);
    }
    T sum = 0;
    for (npy_intp k = 0; k < n; ++k) {
        T xk = get(p_x, x_stride, k);
        sum += xk*xk;
    }
    p_out[0] = sqrt(sum/n);
}


template<typename T>
T crms_core_contig(npy_intp n, T *p_z)
{
    T sum = 0;
    for (npy_intp k = 0; k < n; ++k) {
        T x = p_z[2*k];
        T y = p_z[2*k+1];
        sum += x*x + y*y;
    }
    return sqrt(sum/n);
}

//
// `crms_core_calc`, the C++ core function
// for the gufunc `vnorm` with signature '(n),()->()'
// for types ['F->f', 'D->d', 'G->g'].
//
// XXX This calculation can overflow if elements
// of the input array are large.
//
template<typename T, typename U>
static void crms_core_calc(
        npy_intp n,         // core dimension n
        T *p_z,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp z_stride,  // stride (in bytes) for elements of x
        U *p_out            // pointer to out
)
{
    if (z_stride == sizeof(T)) {
        *p_out = crms_core_contig(n, reinterpret_cast<U *>(p_z));
        return;
    }
    U sum = 0;
    for (npy_intp k = 0; k < n; ++k) {
        T zk = get(p_z, z_stride, k);
        U x = (reinterpret_cast<U *>(&zk))[0];
        U y = (reinterpret_cast<U *>(&zk))[1];
        sum += x*x + y*y;
    }
    p_out[0] = sqrt(sum/n);
}

//
// `vdot_core_calc`, the C++ core function
// for the gufunc `vdot` with signature '(n),(n)->()'
// for types ['ff->f', 'dd->d', 'gg->g'].
//
template<typename T>
static void vdot_core_calc(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_y,             // pointer to first element of y, a strided 1-d array with n elements
        npy_intp y_stride,  // stride (in bytes) for elements of y
        T *p_out            // pointer to out
)
{
    T sum = 0;
    if (x_stride == sizeof(T) && y_stride == sizeof(T)) {
        // Give the compiler a chance to optimize the contiguous case.
        for (int k = 0; k < n; ++k) {
            sum += p_x[k] * p_y[k];
        }
    }
    else {
        for (int k = 0; k < n; ++k) {
            T xk = get(p_x, x_stride, k);
            T yk = get(p_y, y_stride, k);
            sum += xk * yk;
        }
    }
    p_out[0] = sum;
}

#endif
