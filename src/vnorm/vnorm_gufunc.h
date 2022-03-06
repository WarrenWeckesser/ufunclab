
#ifndef VNORM_GUFUNC_H
#define VNORM_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <complex.h>
#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"


static inline npy_float
complex_abs(npy_cfloat z) {
    return npy_cabsf(z);
}

static inline npy_double
complex_abs(npy_cdouble z) {
    return npy_cabs(z);
}

static inline npy_longdouble
complex_abs(npy_clongdouble z) {
    return npy_cabsl(z);
}

#define GET(T, px, stride, index) (*((T *) ((char *) px + k*stride)))

//
// `vnorm_core_calc`, the C++ core function
// for the gufunc `vnorm` with signature '(n),()->()'
// for types ['ff->f', 'dd->d', 'gg->g'].
//
template<typename T>
static void vnorm_core_calc(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_order,         // pointer to order
        T *p_out            // pointer to out
) {
    T maxabsx = 0;
    T order = *p_order;
    if (order <= 0) {
        p_out[0] = NPY_NAN;
        return;
    }

    for (int k = 0; k < n; ++k) {
        T current_x = GET(T, p_x, x_stride, k);
        if (current_x < 0) {
            current_x = -current_x;
        }
        if (current_x > maxabsx) {
            maxabsx = current_x;
        }
    }
    if (maxabsx == 0) {
        p_out[0] = 0;
    }
    else {
        T sum = 0;
        for (int k = 0; k < n; ++k) {
            T current_x = GET(T, p_x, x_stride, k);
            if (current_x < 0) {
                current_x = -current_x;
            }
            if (npy_isinf(order)) {
                sum = std::max(sum, current_x);
            }
            else if (order == 1) {
                sum += current_x;
            }
            else {
                sum += pow(current_x/maxabsx, order);
            }
        }
        if (npy_isinf(order) | (order == 1)) {
            p_out[0] = sum;
        }
        else {
            p_out[0] = maxabsx * pow(sum, 1/order);
        }
    }
}

//
// `cvnorm_core_calc`, the C++ core function
// for the gufunc `vnorm` with signature '(n),()->()'
// for types ['Ff->f', 'Dd->d', 'Gg->g'].
//
template<typename T, typename U>
static void cvnorm_core_calc(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        U *p_order,         // pointer to order
        U *p_out            // pointer to out
) {
    U maxmag = 0;
    U order = *p_order;
    if (order <= 0) {
        p_out[0] = NPY_NAN;
        return;
    }

    for (int k = 0; k < n; ++k) {
        T current_x = GET(T, p_x, x_stride, k);
        U mag = complex_abs(current_x);
        if (mag > maxmag) {
            maxmag = mag;
        }
    }
    if (maxmag == 0) {
        p_out[0] = 0;
    }
    else {
        U sum = 0;
        for (int k = 0; k < n; ++k) {
            T current_x = GET(T, p_x, x_stride, k);
            U mag = complex_abs(current_x);
            if (npy_isinf(order)) {
                sum = fmax(sum, mag);
            }
            else if (order == 1) {
                sum += mag;
            }
            else {
                sum += pow(mag/maxmag, order);
            }
        }
        if (npy_isinf(order) | (order == 1)) {
            p_out[0] = sum;
        }
        else {
            p_out[0] = maxmag * pow(sum, 1/order);
        }
    }
}

#endif
