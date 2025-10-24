#ifndef UFUNCLAB_ORDERED_GUFUNC_H
#define UFUNCLAB_ORDERED_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


// XXX/DRY: comparison_op copied from first_gufunc.h.

enum comparison_op: int8_t {
    LT = Py_LT,
    LE = Py_LE,
    EQ = Py_EQ,
    NE = Py_NE,
    GT = Py_GT,
    GE = Py_GE
};

template<typename T, comparison_op OP>
static inline bool
check_ordered(npy_intp n, const T *p_x, const npy_intp x_stride)
{
    if constexpr (OP == LT) {
        T x1 = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            T x2 = get(p_x, x_stride, k);
            if (x1 >= x2) {
                return false;
            }
            x1 = x2;
        }
    }
    else if constexpr (OP == LE) {
        T x1 = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            T x2 = get(p_x, x_stride, k);
            if (x1 > x2) {
                return false;
            }
            x1 = x2;
        }
    }
    else if constexpr (OP == EQ) {
        T x1 = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            T x2 = get(p_x, x_stride, k);
            if (x1 != x2) {
                return false;
            }
            x1 = x2;
        }
    }
    else if constexpr (OP == NE) {
        T x1 = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            T x2 = get(p_x, x_stride, k);
            if (x1 == x2) {
                return false;
            }
            x1 = x2;
        }
    }
    else if constexpr (OP == GT) {
        T x1 = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            T x2 = get(p_x, x_stride, k);
            if (x1 <= x2) {
                return false;
            }
            x1 = x2;
        }
    }
    else {
        // OP == GE
        T x1 = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            T x2 = get(p_x, x_stride, k);
            if (x1 < x2) {
                return false;
            }
            x1 = x2;
        }
    }
    return true;
}

template<typename T>
static void ordered_core(
        npy_intp n,         // core dimension n
        const T *p_x,       // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        const int8_t *p_op, // pointer to op (LT, LE, etc)
        npy_bool *p_out     // pointer to out
) {
    int8_t op = *p_op;
    *p_out = true;
    if (n > 1) {
        if (op == LT) {
            *p_out = check_ordered<T, LT>(n, p_x, x_stride);
        }
        else if (op == LE) {
            *p_out = check_ordered<T, LE>(n, p_x, x_stride);
        }
        else if (op == EQ) {
            *p_out = check_ordered<T, EQ>(n, p_x, x_stride);
        }
        else if (op == NE) {
            *p_out = check_ordered<T, NE>(n, p_x, x_stride);
        }
        else if (op == GT) {
            *p_out = check_ordered<T, GT>(n, p_x, x_stride);
        }
        else {
            *p_out = check_ordered<T, GE>(n, p_x, x_stride);
        }
    }
}


static void ordered_core_object(
        npy_intp n,         // core dimension n
        PyObject * *p_x,    // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        const int8_t *p_op, // pointer to op (LT, LE, etc)
        npy_bool *p_out     // pointer to out
) {
    int op = *p_op;
    *p_out = true;
    if (n > 1) {
        PyObject *x1 = *p_x;
        for (npy_intp k = 1; k < n; ++k) {
            PyObject *x2 = get(p_x, x_stride, k);
            int cmp = PyObject_RichCompareBool(x1, x2, op);
            if (cmp == -1) {
                return;
            }
            if (cmp == 0) {
                *p_out = false;
                break;
            }
        }
    }
}

#endif  // UFUNCLAB_ORDERED_GUFUNC_H
