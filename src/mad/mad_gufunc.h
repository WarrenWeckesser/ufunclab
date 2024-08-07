
#ifndef MAD_GUFUNC_H
#define MAD_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#ifdef __clang__
#include <cfenv>
#endif
#include <cstdlib>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T>
static int cmp(const void *px, const void *py) {
    T x = *(T *)px;
    T y = *(T *)py;
    if (x < y) {
        return -1;
    }
    else if (x > y) {
        return 1;
    }
    else {
        return 0;
    }
}


template<typename T>
static int unnormalized_mad(npy_intp n, T *p_x, npy_intp x_stride, T& sum, T& total)
{
    // XXX Use a C++ array?  Something else more C++ish?
    // XXX Use PyArray_malloc?
    T *tmp = (T *) malloc(n * sizeof(T));
    if (tmp == NULL) {
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API
            PyErr_Format(PyExc_MemoryError,
                "Unable to allocate %ld bytes (%ld items, each with size %ld) "
                "for intermediate calculation",
                n * sizeof(T), n, sizeof(T));
        NPY_DISABLE_C_API
        return -1;
    }

    // Copy x into tmp.
    if (x_stride == sizeof(T)) {
        memcpy(tmp, p_x, n*sizeof(T));
    }
    else {
        for (npy_intp k = 0; k < n; ++k) {
            tmp[k] = get(p_x, x_stride, k);
        }
    }

    qsort(tmp, n, sizeof(T), cmp<T>);

    sum = 0;
    total = tmp[0];
    for (npy_intp k = 1; k < n; ++k) {
        sum += k*tmp[k] - total;
        total += tmp[k];
    }
    free(tmp);
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
        *p_out = NPY_NAN;
#ifdef __clang__
        feclearexcept(FE_INVALID);
#endif
    }
    else {
        if (denom == 0) {
            *p_out = INFINITY;
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
    if (!isnan(*p_out)) {
        *p_out *= 2;
    }
}

#endif
