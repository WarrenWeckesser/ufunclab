
#ifndef TRI_AREA_GUFUNC_H
#define TRI_AREA_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"


#define GET2(T, px, strides, i, j) (*((T *) ((char *) px + i*strides[0] + j*strides[1])))


template<typename T>
static T norm_diff(npy_intp n, T *p_p, const npy_intp p_strides[2], npy_intp i1, npy_intp i2)
{
    T sumsq = 0.0;
    for (int k = 0; k < n; ++k) {
        T x1 = GET2(T, p_p, p_strides, i1, k);
        T x2 = GET2(T, p_p, p_strides, i2, k);
        T dx = x2 - x1;
        sumsq += dx*dx;
    }
    return sqrt(sumsq);
}

//
// `tri_area_core_calc`, the C++ core function
// for the gufunc `tri_area` with signature '(3,n)->()'
// for types ['ff->f', 'dd->d', 'gg->g'].
//
template<typename T>
static void tri_area_core_calc(
    npy_intp n,                   // core dimension n
    T *p_p,                       // pointer to first element of p, a strided 2-d array with shape (3, n)
    const npy_intp p_strides[2],  // array of length 2 of strides (in bytes) of p
    T *p_out                      // pointer to out
)
{
    T a = norm_diff(n, p_p, p_strides, 0, 1);
    T b = norm_diff(n, p_p, p_strides, 0, 2);
    if (b > a) {
        T t = a;
        a = b;
        b = t;
    }
    T c = norm_diff(n, p_p, p_strides, 1, 2);
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

    // Numerically stable Heron's formula.
    T f1 = a + (b + c);
    T f2 = (c - (a - b));
    T f3 = (c + (a - b));
    T f4 = (a + (b - c));
    *p_out = sqrt(f1*f2*f3*f4)/4;
}

#endif
