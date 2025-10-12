#ifndef UFUNCLAB_MINMAX_GUFUNC_H
#define UFUNCLAB_MINMAX_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stdbool.h>
#include <algorithm>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"



template<typename T>
static void minmax_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_out,           // pointer to out
        npy_intp out_stride // stride (in bytes) for out
)
{
    T xmin = *p_x;
    T xmax = xmin;

    for (npy_intp k = 1; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        xmin = std::min(xmin, x);
        xmax = std::max(xmax, x);
    }
    *p_out = xmin;
    set(p_out, out_stride, 1, xmax);
}


static void minmax_object_core(
        npy_intp n,         // core dimension n
        PyObject **p_x,     // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        PyObject **p_out,   // pointer to out
        npy_intp out_stride // stride (in bytes) for out
)
{
    PyObject *xmin = *p_x;
    PyObject *xmax = xmin;

    for (npy_intp k = 1; k < n; ++k) {
        PyObject *x = get(p_x, x_stride, k);
        int lt = PyObject_RichCompareBool(x, xmin, Py_LT);
        if (lt == -1) {
            return;  // XXX FIXME
        }
        if (lt == 1) {
            xmin = x;
        }
        else {
            int gt = PyObject_RichCompareBool(x, xmax, Py_GT);
            if (gt == -1) {
                return;  // XXX FIXME
            }
            if (gt == 1) {
                xmax = x;
            }
        }
    }
    Py_INCREF(xmin);
    Py_INCREF(xmax);
    *p_out = xmin;
    set(p_out, out_stride, 1, xmax);
}


template<typename T>
static void argminmax_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_intp *p_out,    // pointer to out
        npy_intp out_stride // stride (in bytes) for out
)
{
    T xmin = *p_x;
    T xmax = xmin;
    npy_intp xmin_index = 0;
    npy_intp xmax_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (x < xmin) {
            xmin = x;
            xmin_index = k;
        }
        else if (x > xmax) {
            xmax = x;
            xmax_index = k;
        }
    }
    *p_out = xmin_index;
    set(p_out, out_stride, 1, xmax_index);
}


static void argminmax_object_core(
        npy_intp n,         // core dimension n
        PyObject **p_x,     // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_intp *p_out,    // pointer to out
        npy_intp out_stride // stride (in bytes) for out
)
{
    PyObject *xmin = *p_x;
    PyObject *xmax = xmin;
    npy_intp xmin_index = 0;
    npy_intp xmax_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        PyObject *x = get(p_x, x_stride, k);
        int lt = PyObject_RichCompareBool(x, xmin, Py_LT);
        if (lt == -1) {
            return;  // XXX FIXME
        }
        if (lt == 1) {
            xmin = x;
            xmin_index = k;
        }
        else {
            int gt = PyObject_RichCompareBool(x, xmax, Py_GT);
            if (gt == -1) {
                return;  // XXX FIXME
            }
            if (gt == 1) {
                xmax = x;
                xmax_index = k;
            }
        }
    }
    *p_out = xmin_index;
    set(p_out, out_stride, 1, xmax_index);
}


// XXX To do: Find an elegant way to combine argmin_core and argmax_core, to
//            eliminate the duplicated code.

template<typename T>
static void argmin_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_intp *p_out     // pointer to out
)
{
    T xmin = *p_x;
    npy_intp xmin_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (x < xmin) {
            xmin = x;
            xmin_index = k;
        }
    }
    *p_out = xmin_index;
}

static void argmin_object_core(
        npy_intp n,         // core dimension n
        PyObject **p_x,     // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_intp *p_out     // pointer to out
)
{
    PyObject *xmin = *p_x;
    npy_intp xmin_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        PyObject *x = get(p_x, x_stride, k);
        int test = PyObject_RichCompareBool(x, xmin, Py_LT);
        if (test == -1) {
            return;  // XXX FIXME
        }
        if (test == 1) {
            xmin = x;
            xmin_index = k;
        }
    }
    *p_out = xmin_index;
}


template<typename T>
static void argmax_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_intp *p_out     // pointer to out
)
{
    T xmax = *p_x;
    npy_intp xmax_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (x > xmax) {
            xmax = x;
            xmax_index = k;
        }
    }
    *p_out = xmax_index;
}

static void argmax_object_core(
        npy_intp n,         // core dimension n
        PyObject **p_x,     // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        npy_intp *p_out     // pointer to out
)
{
    PyObject *xmax = *p_x;
    npy_intp xmax_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        PyObject *x = get(p_x, x_stride, k);
        int test = PyObject_RichCompareBool(x, xmax, Py_GT);
        if (test == -1) {
            return;  // XXX FIXME
        }
        if (test == 1) {
            xmax = x;
            xmax_index = k;
        }
    }
    *p_out = xmax_index;
}


template<typename T>
static void min_argmin_core(
        npy_intp n,           // core dimension n
        T *p_x,               // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        T *p_out1,            // pointer to out1
        npy_intp *p_out2      // pointer to out2
)
{
    T xmin = *p_x;
    npy_intp xmin_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (x < xmin) {
            xmin = x;
            xmin_index = k;
        }
    }
    *p_out1 = xmin;
    *p_out2 = xmin_index;
}

static void min_argmin_object_core(
        npy_intp n,           // core dimension n
        PyObject **p_x,       // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        PyObject **p_out1,    // pointer to out1
        npy_intp *p_out2      // pointer to out2
)
{
    PyObject *xmin = *p_x;
    npy_intp xmin_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        PyObject *x = get(p_x, x_stride, k);
        int test = PyObject_RichCompareBool(x, xmin, Py_LT);
        if (test == -1) {
            return;  // XXX FIXME
        }
        if (test == 1) {
            xmin = x;
            xmin_index = k;
        }
    }
    Py_INCREF(xmin);
    *p_out1 = xmin;
    *p_out2 = xmin_index;
}


template<typename T>
static void max_argmax_core(
        npy_intp n,           // core dimension n
        T *p_x,               // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        T *p_out1,            // pointer to out1
        npy_intp *p_out2      // pointer to out2
)
{
    T xmax = *p_x;
    npy_intp xmax_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        if (x > xmax) {
            xmax = x;
            xmax_index = k;
        }
    }
    *p_out1 = xmax;
    *p_out2 = xmax_index;
}

static void max_argmax_object_core(
        npy_intp n,           // core dimension n
        PyObject **p_x,       // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        PyObject **p_out1,    // pointer to out1
        npy_intp *p_out2      // pointer to out2
)
{
    PyObject *xmax = *p_x;
    npy_intp xmax_index = 0;

    for (npy_intp k = 1; k < n; ++k) {
        PyObject *x = get(p_x, x_stride, k);
        int test = PyObject_RichCompareBool(x, xmax, Py_GT);
        if (test == -1) {
            return;  // XXX FIXME
        }
        if (test == 1) {
            xmax = x;
            xmax_index = k;
        }
    }
    Py_INCREF(xmax);
    *p_out1 = xmax;
    *p_out2 = xmax_index;
}

#endif  // UFUNCLAB_MINMAX_GUFUNC_H
