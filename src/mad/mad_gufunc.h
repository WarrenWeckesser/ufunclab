
#ifndef UFUNCLAB_MAD_GUFUNC_H
#define UFUNCLAB_MAD_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <algorithm>
#ifdef __clang__
#include <cfenv>
#endif
#include <cstdlib>
#include <cmath>
#include <limits>
#include <new>      // for std::bad_alloc
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T>
static int unnormalized_mad(npy_intp n, T *p_x, npy_intp x_stride, T& sum, T& total)
{
    std::vector<T> tmp;
    try {
        tmp.resize(n);
    } catch (const std::bad_alloc& e) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
            PyErr_Format(PyExc_MemoryError,
                "Unable to allocate internal array with length %ld "
                "for intermediate calculation", n);
        NPY_DISABLE_C_API
        return -1;
    }

    // Copy x into tmp.
    if (x_stride == sizeof(T)) {
        memcpy(tmp.data(), p_x, n*sizeof(T));
    }
    else {
        for (npy_intp k = 0; k < n; ++k) {
            tmp[k] = get(p_x, x_stride, k);
        }
    }

    try {
        std::sort(tmp.begin(), tmp.end());
    } catch (const std::bad_alloc& e) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
            PyErr_Format(PyExc_MemoryError,
                         "Internal memory allocation failure during attempt "
                         "to sort the data");
        NPY_DISABLE_C_API
        return -1;
    }

    sum = 0;
    total = tmp[0];
    for (npy_intp k = 1; k < n; ++k) {
        sum += k*tmp[k] - total;
        total += tmp[k];
    }
    return 0;
}

//
// `mad_core`, the C++ core function
// for the gufunc `mad` with signature '(n),()->()'
// for types ['f?->f', 'd?->d', 'g?->g'].
//
template<typename T>
static void mad_core(
        npy_intp n,            // core dimension n
        T *p_x,                // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,     // stride (in bytes) for elements of x
        npy_bool *p_unbiased,  // pointer to unbiased
        T *p_out               // pointer to out
)
{
    T sum, total;
    if (unnormalized_mad(n, p_x, x_stride, sum, total) != 0) {
        return;
    }

    if (*p_unbiased) {
        *p_out = 2*sum/n/(n-1);
    }
    else {
        *p_out = 2*sum/(n*n);
    }
}

//
// `gini_core`, the C++ core function
// for the gufunc `gini` with signature '(n),()->()'
// for types ['f?->f', 'd?->d', 'g?->g'].
//
template<typename T>
static void gini_core(
        npy_intp n,            // core dimension n
        T *p_x,                // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,     // stride (in bytes) for elements of x
        npy_bool *p_unbiased,  // pointer to unbiased
        T *p_out               // pointer to out
)
{
    T sum, total;
    if (unnormalized_mad(n, p_x, x_stride, sum, total) != 0) {
        return;
    }

    T denom = (*p_unbiased) ? (n - 1)*total : n*total;

    if (sum == 0 && denom == 0) {
        *p_out = std::numeric_limits<T>::quiet_NaN();
#ifdef __clang__
        feclearexcept(FE_INVALID);
#endif
    }
    else {
        if (denom == 0) {
            *p_out = std::numeric_limits<T>::infinity();
        }
        else {
            *p_out = sum / denom;
        }
    }
}

//
// `rmad_core`, the C++ core function
// for the gufunc `rmad` with signature '(n),()->()'
// for types ['f?->f', 'd?->d', 'g?->g'].
//
template<typename T>
static void rmad_core(
        npy_intp n,            // core dimension n
        T *p_x,                // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,     // stride (in bytes) for elements of x
        npy_bool *p_unbiased,  // pointer to unbiased
        T *p_out               // pointer to out
)
{
    // RMAD is twice the Gini index.
    gini_core(n, p_x, x_stride, p_unbiased, p_out);
    if (!std::isnan(*p_out)) {
        *p_out *= 2;
    }
}

#endif  // UFUNCLAB_MAD_GUFUNC_H
