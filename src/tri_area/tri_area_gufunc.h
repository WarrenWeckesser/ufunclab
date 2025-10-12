
#ifndef UFUNCLAB_TRI_AREA_GUFUNC_H
#define UFUNCLAB_TRI_AREA_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T>
static T norm_diff(npy_intp n, T *p_p, const npy_intp p_strides[2], npy_intp i1, npy_intp i2)
{
    T sumsq = 0.0;
    for (int k = 0; k < n; ++k) {
        T x1 = get2d(p_p, p_strides, i1, k);
        T x2 = get2d(p_p, p_strides, i2, k);
        T dx = x2 - x1;
        sumsq += dx*dx;
    }
    return std::sqrt(sumsq);
}

//
// This function is used by tri_area_core_calc and tri_area_indexed_core_calc.
//
template<typename T>
static void
tri_area_common_calc(
    npy_intp n,                   // core dimension n
    T *p_p,                       // pointer to first element of p, a strided 2-d array with shape (m, n)
    const npy_intp p_strides[2],  // array of length 2 of strides (in bytes) of p
    npy_intp i0,                  // index of first point
    npy_intp i1,                  // index of second point
    npy_intp i2,                  // index of third point
    T *p_out                      // pointer to out
)
{
    T a = norm_diff(n, p_p, p_strides, i0, i1);
    T b = norm_diff(n, p_p, p_strides, i0, i2);
    T c = norm_diff(n, p_p, p_strides, i1, i2);
    if (b > a) {
        T t = a;
        a = b;
        b = t;
    }
    if (c > a) {
        T t = c;
        c = b;
        b = a;
        a = t;
    }
    else if (c > b) {
        T t = c;
        c = b;
        b = t;
    }
    // a, b, and c are the lengths of the sides of the triangle,
    // in decreasing order.

    // Numerically stable version of Heron's formula.  See for details:
    //
    //   W. Kahan, "Miscalculating area and angles of a needle-like
    //   triangle", 2014, preprint available at
    //   https://people.eecs.berkeley.edu/~wkahan/Triangle.pdf
    //
    T f1 = a + (b + c);
    T f2 = (c - (a - b));
    T f3 = (c + (a - b));
    T f4 = (a + (b - c));
    *p_out = std::sqrt(f1*f2*f3*f4)/4;
}

//
// `tri_area_core_calc`, the C++ core function
// for the gufunc `tri_area` with signature '(3,n)->()'
// for types ['f->f', 'd->d', 'g->g'].
//
template<typename T>
static void tri_area_core_calc(
    npy_intp n,                   // core dimension n
    T *p_p,                       // pointer to first element of p, a strided 2-d array with shape (3, n)
    const npy_intp p_strides[2],  // array of length 2 of strides (in bytes) of p
    T *p_out                      // pointer to out
)
{
    tri_area_common_calc(n, p_p, p_strides, 0, 1, 2, p_out);
}


static inline
npy_intp get_index(npy_intp *p_i, npy_intp i_stride, npy_intp index, npy_intp m)
{
    npy_intp i = get(p_i, i_stride, index);
    if (i < 0) {
        i += m;
    }
    if (i < 0 || i >= m) {
        return -1;
    }
    return i;
}

//
// `tri_area_indexed_core_calc`, the C++ core function
// for the gufunc `tri_area_indexed` with signature '(m, n),(3)->()'
// for types ['fp->f', 'dp->d', 'gp->g'].
//
template<typename T>
static void tri_area_indexed_core_calc(
    npy_intp m,                   // core dimension m
    npy_intp n,                   // core dimension n
    T *p_p,                       // pointer to first element of p, a strided
                                  // 2-d array with shape (m, n)
    const npy_intp p_strides[2],  // array of length 2 of strides (in bytes) of p
    npy_intp *p_i,                // pointer to first element of i, a strided 1-d array
    npy_intp i_stride,            // stride of elements in i
    T *p_out                      // pointer to out
)
{
    npy_intp i0, i1, i2;
    i0 = get_index(p_i, i_stride, 0, m);
    i1 = get_index(p_i, i_stride, 1, m);
    i2 = get_index(p_i, i_stride, 2, m);
    if (i0 == -1 || i1 == -1 || i2 == -1) {
        // XXX Should probably raise IndexError instead of returning NAN.
        *p_out = NPY_NAN;
        return;
    }
    tri_area_common_calc(n, p_p, p_strides, i0, i1, i2, p_out);
}

#endif  // UFUNCLAB_TRI_AREA_GUFUNC_H
