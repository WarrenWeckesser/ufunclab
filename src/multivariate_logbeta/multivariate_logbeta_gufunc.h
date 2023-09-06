#ifndef MULTIVARIATE_LOGBETA_GUFUNC_H
#define MULTIVARIATE_LOGBETA_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <cmath>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T>
static void multivariate_logbeta_core_contig(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, an array with n elements
        T *p_out            // pointer to out
)
{
    T log_sum = 0.0;
    T alpha_sum = 0.0;
    for (npy_intp k = 0; k < n; ++k) {
        T x = p_x[k];
        alpha_sum += x;
        log_sum += std::lgamma(x);
    }
    log_sum -= std::lgamma(alpha_sum);
    *p_out = log_sum;
}


template<typename T>
static void multivariate_logbeta_core(
        npy_intp n,         // core dimension n
        T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,  // stride (in bytes) for elements of x
        T *p_out            // pointer to out
)
{
    if (n == 0) {
        *p_out = NAN;
        return;
    }
    if (x_stride == sizeof(T)) {
        multivariate_logbeta_core_contig(n, p_x, p_out);
        return;
    }
    T log_sum = 0.0;
    T alpha_sum = 0.0;
    for (npy_intp k = 0; k < n; ++k) {
        T x = get(p_x, x_stride, k);
        alpha_sum += x;
        log_sum += std::lgamma(x);
    }
    log_sum -= std::lgamma(alpha_sum);
    *p_out = log_sum;
}

#endif
