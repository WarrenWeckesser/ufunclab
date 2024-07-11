
#ifndef FIRST_GUFUNC_H
#define FIRST_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


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
cmp(const T& a, const T& b)
{
    if constexpr (OP == LT) {
        return a < b;
    }
    else if constexpr (OP == LE) {
        return a <= b;
    }
    else if constexpr (OP == EQ) {
        return a == b;
    }
    else if constexpr (OP == NE) {
        return a != b;
    }
    else if constexpr (OP == GT) {
        return a > b;
    }
    else {
        // OP == GE
        return a >= b;
    }
}

//--------------------------------------------------------
// 'first' functions
//--------------------------------------------------------

template<typename T, comparison_op OP>
static inline T
first(npy_intp n,
      const T *p_x,
      npy_intp stride,
      const T* p_target,
      const T* p_otherwise)
{
    T target = *p_target;
    T result = *p_otherwise;
    for (npy_intp k = 0; k < n; ++k) {
        T value = get(p_x, stride, k);
        if (cmp<T, OP>(value, target)) {
            result = value;
            break;
        }
    }
    return result;
}

//
// `first_core_calc`, the C++ core function
// for the gufunc `first` with signature '(n),(),(),()->()'
// for types ['fbff->f', 'dbdd->d', etc.].
//
template<typename T>
static void
first_core_calc(
    npy_intp n,         // core dimension n
    T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
    npy_intp x_stride,  // stride (in bytes) for elements of x
    int8_t *p_op,
    T *p_target,
    T *p_otherwise,
    T *p_out
)
{
    T result = *p_otherwise;
    int8_t op = *p_op;

    switch (op) {
        case Py_LT:
            result = first<T, LT>(n, p_x, x_stride, p_target, p_otherwise);
            break;
        case Py_LE:
            result = first<T, LE>(n, p_x, x_stride, p_target, p_otherwise);
            break;
        case Py_EQ:
            result = first<T, EQ>(n, p_x, x_stride, p_target, p_otherwise);
            break;
        case Py_NE:
            result = first<T, NE>(n, p_x, x_stride, p_target, p_otherwise);
            break;
        case Py_GT:
            result = first<T, GT>(n, p_x, x_stride, p_target, p_otherwise);
            break;
        case Py_GE:
            result = first<T, GE>(n, p_x, x_stride, p_target, p_otherwise);
            break;
    }
    p_out[0] = result;
}

static inline PyObject *
first_object(npy_intp n,
             PyObject* *p_x,
             npy_intp x_stride,
             int py_op,
             PyObject* *p_target,
             PyObject* *p_otherwise)
{
    PyObject* result = *p_otherwise;
    // XXX This relies on the enumerated values of the Python comparison
    // operations (Py_LT, etc) being between 0 and 5.
    if (py_op < 0 || py_op > 5) {
        return result;
    }
    PyObject* target = *p_target;
    for (npy_intp k = 0; k < n; ++k) {
        PyObject* x = get(p_x, x_stride, k);
        int cmp = PyObject_RichCompareBool(x, target, py_op);
        if (cmp == -1) {
            return NULL;  // Error occurred in PyObject_RichCompareBool().
        }
        if (cmp == 1) {
            result = x;
            break;
        }
    }
    return result;
}

static void
first_core_calc_object(
    npy_intp n,         // core dimension n
    PyObject* *p_x,     // pointer to first element of x, a strided 1-d array with n elements
    npy_intp x_stride,  // stride (in bytes) for elements of x
    int8_t *p_op,
    PyObject* *p_target,
    PyObject* *p_otherwise,
    PyObject* *p_out
)
{
    PyObject* result = *p_otherwise;
    int py_op = *p_op;
    // XXX This relies on the enumerated values of the Python comparison
    // operations (Py_LT, etc) being between 0 and 5.
    if (py_op >= 0 && py_op <= 5) {
        PyObject* target = *p_target;
        for (npy_intp k = 0; k < n; ++k) {
            PyObject* x = get(p_x, x_stride, k);
            int cmp = PyObject_RichCompareBool(x, target, py_op);
            if (cmp == -1) {
                // Error occurred in PyObject_RichCompareBool().
                result = NULL;
                break;
            }
            if (cmp == 1) {
                result = x;
                break;
            }
        }
    }
    Py_XINCREF(result);
    *p_out = result;
}


//--------------------------------------------------------
// 'argfirst' functions
//--------------------------------------------------------

template<typename T, comparison_op OP>
static inline npy_intp
argfirst(npy_intp n,
         const T *p_x,
         npy_intp stride,
         const T *p_target)
{
    npy_intp result = -1;
    T target = *p_target;
    for (npy_intp k = 0; k < n; ++k) {
        T value = get(p_x, stride, k);
        if (cmp<T, OP>(value, target)) {
            result = k;
            break;
        }
    }
    return result;
}

//
// `argfirst_core_calc`, the C++ core function
// for the gufunc `first` with signature '(n),(),()->()'
// for types ['fbf->l', 'dbd->l', etc.].
//
template<typename T>
static void
argfirst_core_calc(
    npy_intp n,         // core dimension n
    T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
    npy_intp x_stride,  // stride (in bytes) for elements of x
    int8_t *p_op,
    T *p_target,
    npy_intp *p_out
)
{
    npy_intp result = -1;
    int8_t op = *p_op;

    switch (op) {
        case Py_LT:
            result = argfirst<T, LT>(n, p_x, x_stride, p_target);
            break;
        case Py_LE:
            result = argfirst<T, LE>(n, p_x, x_stride, p_target);
            break;
        case Py_EQ:
            result = argfirst<T, EQ>(n, p_x, x_stride, p_target);
            break;
        case Py_NE:
            result = argfirst<T, NE>(n, p_x, x_stride, p_target);
            break;
        case Py_GT:
            result = argfirst<T, GT>(n, p_x, x_stride, p_target);
            break;
        case Py_GE:
            result = argfirst<T, GE>(n, p_x, x_stride, p_target);
            break;
    }
    p_out[0] = result;
}

static void
argfirst_core_calc_object(
    npy_intp n,         // core dimension n
    PyObject* *p_x,     // pointer to first element of x, a strided 1-d array with n elements
    npy_intp x_stride,  // stride (in bytes) for elements of x
    int8_t *p_op,
    PyObject* *p_target,
    npy_intp *p_out
)
{
    npy_intp result = -1;
    int py_op = *p_op;
    // XXX This relies on the enumerated values of the Python comparison
    // operations (Py_LT, etc) being between 0 and 5.
    if (py_op >= 0 && py_op <= 5) {
        PyObject* target = *p_target;
        for (npy_intp k = 0; k < n; ++k) {
            PyObject* x = get(p_x, x_stride, k);
            int cmp = PyObject_RichCompareBool(x, target, py_op);
            if (cmp == -1) {
                // Error occurred in PyObject_RichCompareBool().
                result = -2;
                break;
            }
            if (cmp == 1) {
                result = k;
                break;
            }
        }
    }
    *p_out = result;
}

//--------------------------------------------------------

//
// The name of this function is listed in the `extra_module_funcs`
// attribute of the UFuncExtMod object that defines the gufuncs
// `first` and `argfirst`.  A call of this function will be added
// to the end of the generated extension module.
//
int add_comparison_constants(PyObject *module)
{
    // Expose the numerical values Py_LT, Py_LE, etc., as integers
    // in this module.
    const char *opnames[] = {"_LT", "_LE", "_EQ", "_NE", "_GT", "_GE"};
    const int opcodes[] = {Py_LT, Py_LE, Py_EQ, Py_NE, Py_GT, Py_GE};
    for (int k = 0; k < 6; ++k) {
        int status = PyModule_AddIntConstant(module, opnames[k], opcodes[k]);
        if (status == -1) {
            return -1;
        }
    }
    return 0;
}

#endif
