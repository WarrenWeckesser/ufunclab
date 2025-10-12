#ifndef UFUNCLAB_BACKLASH_GUFUNC_H
#define UFUNCLAB_BACKLASH_GUFUNC_H

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


template<typename T>
static void
backlash_sum_core(
        npy_intp n,                     // core dimension n
        npy_intp m,                     // core dimension m
        T *p_x,                         // pointer to first element of x, a strided 1-d array with shape (n,)
        const npy_intp x_stride,        // stride (in bytes) of x
        T *p_w,                         // pointer to first element of w, a strided 1-d array with shape (m,)
        const npy_intp w_stride,        // stride (in bytes) of w
        T *p_deadband,                  // pointer to first element of deadband, a strided 1-d array with shape (m,)
        const npy_intp deadband_stride, // stride (in bytes) of deadband
        T *p_initial,                   // pointer to initial
        const npy_intp initial_stride,  // stride (in bytes) of initial, a strided 1-d array with shape (m,)
        T *p_out,                       // pointer to out, a strided 1-d array with shape (n,)
        const npy_intp out_stride,      // stride (in bytes) of out
        T *p_final,                     // point to final, a strided 1-d array with shape (m,)
        const npy_intp final_stride     // stride (in bytes) of final
)
{
    for (npy_intp j = 0; j < m; ++j) {
        set(p_final, final_stride, j, get(p_initial, initial_stride, j));
    }
    for (npy_intp k = 0; k < n; ++k) {
        T current_x = get(p_x, x_stride, k);
        T y_total = 0.0;
        for (npy_intp j = 0; j < m; ++j) {
            T current_y = get(p_final, final_stride, j);
            T halfband = get(p_deadband, deadband_stride, j)/2;
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
            set(p_final, final_stride, j, current_y);
            T w = get(p_w, w_stride, j);
            y_total += w*current_y;
        }
        set(p_out, out_stride, k, y_total);
    }
}

#endif  // UFUNCLAB_BACKLASH_GUFUNC_H
