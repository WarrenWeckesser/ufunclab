#ifndef UFUNCLAB_FIRST_GUFUNC_H
#define UFUNCLAB_FIRST_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>
#include <limits>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


//
// `nextn_greater_core_calc` is the C++ core function
// for the gufunc `nextn_greater` with signature '()->(n)'
//
template<typename T>
static void
nextn_greater_core_calc(
    npy_intp n,         // core dimension n
    T *p_x,             // pointer to x
    T *p_out,           // pointer to out, a strided 1-d array
    npy_intp out_stride
)
{
    T to = std::numeric_limits<T>::infinity();
    T x = *p_x;
    for (npy_intp i = 0; i < n; ++i) {
        x = std::nextafter(x, to);
        set(p_out, out_stride, i, x);
    }
}

//
// `nextn_less_core_calc` is the C++ core function
// for the gufunc `nextn_less` with signature '()->(n)'
//
template<typename T>
static void
nextn_less_core_calc(
    npy_intp n,         // core dimension n
    T *p_x,             // pointer to x
    T *p_out,           // pointer to out, a strided 1-d array
    npy_intp out_stride
)
{
    T to = -std::numeric_limits<T>::infinity();
    T x = *p_x;
    for (npy_intp i = 0; i < n; ++i) {
        x = std::nextafter(x, to);
        set(p_out, out_stride, i, x);
    }
}

#endif  // UFUNCLAB_FIRST_GUFUNC_H
