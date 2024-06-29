#ifndef BACKLASH_GUFUNC_H
#define BACKLASH_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T>
static void
backlash_core(
        npy_intp n,                     // core dimension n
        T *p_x,                         // pointer to first element of x, a strided 1-d array with shape (n,)
        const npy_intp x_stride,        // stride (in bytes) of x
        T *p_deadband,                  // pointer to deadband
        T *p_initial,                   // pointer to initial
        T *p_out,                       // pointer to out, a strided 1-d array with shape (n,)
        const npy_intp out_stride       // stride (in bytes) of out
)
{
    T deadband = *p_deadband;
    T initial = *p_initial;
    T halfband = deadband/2;
    T current_y = initial;
    for (npy_intp k = 0; k < n; ++k) {
        T current_x = get(p_x, x_stride, k);
        T xminus = current_x - halfband;
        if (xminus > current_y) {
            current_y = xminus;
        }
        else {
            T xplus = current_x + halfband;
            if (xplus < current_y) {
                current_y = xplus;
            }
        }
        set(p_out, out_stride, k, current_y);
    }
}

#endif
