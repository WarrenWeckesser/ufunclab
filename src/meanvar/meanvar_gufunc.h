#ifndef UFUNCLAB_MEANVAR_GUFUNC_H
#define UFUNCLAB_MEANVAR_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include <utility>
#include <tuple>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename U>
inline std::pair<U, U>
compensated_update(U val, U incr, U comp)
{
    U y = incr - comp;
    U t = val + y;
    U c = (t - val) - y;
    return std::make_pair(t, c);
}

template<typename T, typename U>
static void meanvar_core(
        npy_intp n,           // core dimension n
        T *p_x,               // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        npy_intp *p_ddof,     // pointer to ddof
        U *p_out,             // pointer to first element of out, a strided 1-d array with 2 elements
        npy_intp out_stride   // stride (in bytes) for elements of out
)
{
    U mean = 0.0;
    U m2 = 0.0;
    U c1 = 0.0, c2 = 0.0;
    for (npy_intp k = 0; k < n; ++k) {
        T xk = get(p_x, x_stride, k);
        U delta = xk - mean;
        std::tie(mean, c1) = compensated_update(mean, delta/(k + 1), c1);
        std::tie(m2, c2) = compensated_update(m2, delta*(xk - mean), c2);
    }
    U var = m2 / (n - *p_ddof);
    *p_out = mean;
    set(p_out, out_stride, 1, var);
}

#endif  // UFUNCLAB_MEANVAR_GUFUNC_H
