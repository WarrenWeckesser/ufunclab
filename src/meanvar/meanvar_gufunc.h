#ifndef MEANVAR_GUFUNC_H
#define MEANVAR_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T, typename U>
static void meanvar_core(
        npy_intp n,           // core dimension n
        T *p_x,               // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        npy_intp *p_ddof,     // pointer to ddof
        U *p_out,             // pointer to first element of out, a strided 1-d array with 2 elements
        npy_intp out_stride   // stride (in bytes) for elements of out
) {
    U mean = 0.0;
    U var = 0.0;
    U c1 = 0.0, m2 = 0.0, c2 = 0.0;
    for (npy_intp k = 0; k < n; ++k) {
        T xk = get(p_x, x_stride, k);
        U delta = xk - mean;
        {
            U y1 = delta/(k + 1) - c1;
            U t1 = mean + y1;
            c1 = (t1 - mean) - y1;
            mean = t1;
        }
        {
            U y2 = delta * (xk - mean) - c2;
            U t2 = m2 + y2;
            c2 = (t2 - m2) - y2;
            m2 = t2;
        }
    }
    var = m2 / (n - *p_ddof);
    *p_out = mean;
    set(p_out, out_stride, 1, var);
}

#endif
