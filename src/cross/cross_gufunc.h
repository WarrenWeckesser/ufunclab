
#ifndef UFUNCLAB_CROSS_GUFUNC_H
#define UFUNCLAB_CROSS_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayscalars.h"
#include "numpy/ufuncobject.h"

#include "../src/util/strided.hpp"

//
// `cross3_core_calc`, the C++ core function
// for the gufunc `cross3` with signature '(3),(3)->(3)'
// for the real numeric types (integers and floating point).
//
template<typename T>
static void
cross3_core_calc(
    T *p_u,              // pointer to first element of u, a strided 1-d array with n elements
    npy_intp u_stride,   // stride (in bytes) for elements of u
    T *p_v,              // pointer to first element of v, a strided 1-d array with n elements
    npy_intp v_stride,   // stride (in bytes) for elements of v
    T *p_out,            // pointer to out
    npy_intp out_stride  // stride (in bytes) for elements of out
)
{
    T u0 = get(p_u, u_stride, 0);
    T u1 = get(p_u, u_stride, 1);
    T u2 = get(p_u, u_stride, 2);
    T v0 = get(p_v, v_stride, 0);
    T v1 = get(p_v, v_stride, 1);
    T v2 = get(p_v, v_stride, 2);
    set(p_out, out_stride, 0, u1*v2 - u2*v1);
    set(p_out, out_stride, 1, u2*v0 - u0*v2);
    set(p_out, out_stride, 2, u0*v1 - u1*v0);
}

//
// Computes a*d - b*c
// Returns a new reference, or NULL on error.
//
static PyObject *det2(PyObject *a, PyObject *b, PyObject *c, PyObject *d)
{
    PyObject *p0, *p1, *diff;

    p0 = PyNumber_Multiply(a, d);
    if (p0 == NULL) {
        return NULL;
    }
    p1 = PyNumber_Multiply(b, c);
    if (p1 == NULL) {
        Py_DECREF(p0);
        return NULL;
    }
    diff = PyNumber_Subtract(p0, p1);
    Py_DECREF(p0);
    Py_DECREF(p1);
    return diff;
}

//
// `cross3_core_calc_object`, the C++ core function
// for the gufunc `cross3` with signature '(3),(3)->(3)'
// for the object type.
//
static void
cross3_core_calc_object(
    PyObject* *p_u,      // pointer to first element of u, a strided 1-d array with n elements
    npy_intp u_stride,   // stride (in bytes) for elements of u
    PyObject* *p_v,      // pointer to first element of v, a strided 1-d array with n elements
    npy_intp v_stride,   // stride (in bytes) for elements of v
    PyObject* *p_out,    // pointer to out
    npy_intp out_stride  // stride (in bytes) for elements of out
)
{
    PyObject* u0 = get(p_u, u_stride, 0);
    PyObject* u1 = get(p_u, u_stride, 1);
    PyObject* u2 = get(p_u, u_stride, 2);
    PyObject* v0 = get(p_v, v_stride, 0);
    PyObject* v1 = get(p_v, v_stride, 1);
    PyObject* v2 = get(p_v, v_stride, 2);

    PyObject* out0 = det2(u1, u2, v1, v2);
    if (out0 == NULL) {
        return;
    }

    PyObject* out1 = det2(u2, u0, v2, v0);
    if (out1 == NULL) {
        Py_DECREF(out0);
        return;
    }

    PyObject* out2 = det2(u0, u1, v0, v1);
    if (out2 == NULL) {
        Py_DECREF(out0);
        Py_DECREF(out1);
        return;
    }

    set(p_out, out_stride, 0, out0);
    set(p_out, out_stride, 1, out1);
    set(p_out, out_stride, 2, out2);
}

//
// `cross2_core_calc`, the C++ core function
// for the gufunc `cross2` with signature '(2),(2)->()'
// for the real numeric types (integers and floating point).
//
template<typename T>
static void
cross2_core_calc(
        T *p_u,              // pointer to first element of u, a strided 1-d array with n elements
        npy_intp u_stride,   // stride (in bytes) for elements of u
        T *p_v,              // pointer to first element of v, a strided 1-d array with n elements
        npy_intp v_stride,   // stride (in bytes) for elements of v
        T *p_out             // pointer to out
)
{
    T u0 = get(p_u, u_stride, 0);
    T u1 = get(p_u, u_stride, 1);
    T v0 = get(p_v, v_stride, 0);
    T v1 = get(p_v, v_stride, 1);
    p_out[0] = u0*v1 - u1*v0;
}

//
// `cross2_core_calc_object`, the C++ core function
// for the gufunc `cross2` with signature '(2),(2)->()'
// for the object type.
//
static void
cross2_core_calc_object(
    PyObject* *p_u,      // pointer to first element of x, a strided 1-d array with n elements
    npy_intp u_stride,   // stride (in bytes) for elements of x
    PyObject* *p_v,      // pointer to first element of x, a strided 1-d array with n elements
    npy_intp v_stride,   // stride (in bytes) for elements of x
    PyObject* *p_out     // pointer to out
)
{
    PyObject* u0 = get(p_u, u_stride, 0);
    PyObject* u1 = get(p_u, u_stride, 1);
    PyObject* v0 = get(p_v, v_stride, 0);
    PyObject* v1 = get(p_v, v_stride, 1);

    PyObject* out = det2(u0, u1, v0, v1);
    if (out == NULL) {
        return;
    }
    p_out[0] = out;
}

#endif  // UFUNCLAB_CROSS_GUFUNC_H
