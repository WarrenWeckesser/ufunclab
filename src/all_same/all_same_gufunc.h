#ifndef ALL_SAME_GUFUNC_H
#define ALL_SAME_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../util/strided.hpp"


template<typename T>
static void all_same_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_bool *p_out     // pointer to out
) {
    *p_out = true;
    if (n > 1) {
        T first = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            T x = get(p_x, x_stride, k);
            if (x != first) {
                *p_out = false;
                break;
            }
        }
    }
}


static void all_same_core_object(
        npy_intp n,         // core dimension n
        PyObject * *p_x,    // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_bool *p_out     // pointer to out
) {
    *p_out = true;
    if (n > 1) {
        PyObject *first = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            PyObject *x = get(p_x, x_stride, k);
            int ne = PyObject_RichCompareBool(x, first, Py_NE);
            if (ne == -1) {
                return;
            }
            if (ne == 1) {
                *p_out = false;
                break;
            }
        }
    }
}

#endif
