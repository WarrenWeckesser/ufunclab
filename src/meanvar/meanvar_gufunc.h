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

template<typename T, typename U>
static void meanvar_twopass_core(
        npy_intp n,           // core dimension n
        const T * const p_x,               // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        const npy_intp * const p_ddof,     // pointer to ddof
        U *p_out,             // pointer to first element of out, a strided 1-d array with 2 elements
        npy_intp out_stride   // stride (in bytes) for elements of out
)
{
    U mean = 0.0;
    U var = 0.0, tmp;
    for (npy_intp k = 0; k < n; ++k) {
        T xk = get(p_x, x_stride, k);
	mean += xk;
    }
    mean /= n;
    for (npy_intp k = 0; k < n; ++k) {
      T xk = get(p_x, x_stride, k);
      tmp = xk - mean;
      var += tmp * tmp;
    }
    var /= (n - *p_ddof);
    *p_out = mean;
    set(p_out, out_stride, 1, var);
}

template<typename T, typename U>
static void covariance_matrix_core(
				   npy_intp n,
				   npy_intp m,
				   const T * const p_x,
				   const npy_intp x_strides[2],
				   const npy_intp * const p_ddof,
				   U *p_out,
				   const npy_intp out_strides[2]
				   )
{
  U mean[m] = {0.0};
  U cov[m*m] = {0.0};
  U tmp1, tmp2;
  for (npy_intp i = 0; i < n; ++i) {
    #pragma GCC ivdep
    for (npy_intp j = 0; j < m; ++j) {
      T x_ij = get2d(p_x, x_strides, i, j);
      mean[j] += x_ij;
    }
  }
  for (npy_intp j = 0; j < m; ++j) {
    mean[j] /= n;
  }
  for (npy_intp i = 0; i < n; ++i) {
    #pragma GCC ivdep
    for (npy_intp j = 0; j < m; ++j) {
      T x_ij = get2d(p_x, x_strides, i, j);
      tmp1 = x_ij - mean[j];
      #pragma GCC ivdep
      for (npy_intp k = j; k < m; ++k) {
	T x_ik = get2d(p_x, x_strides, i, k);
	tmp2 = x_ik - mean[k];
	cov[j * m + k] += tmp1 * tmp2;
      }
    }
  }
  for (npy_intp j = 0; j < m; ++j) {
    for (npy_intp k = j; k < m; ++k) {
      tmp1 = cov[j * m + k] / (n - *p_ddof);
      set2d(p_out, out_strides, j, k, tmp1);
      set2d(p_out, out_strides, k, j, tmp1);
    }
  }
}

extern "C" {
  static int
  preprocess_cov_core_dims(PyUFuncObject *ufunc, npy_intp *core_dim_sizes) {
    npy_intp n = core_dim_sizes[0];
    npy_intp m = core_dim_sizes[1];
    if (n < 2) {
      PyErr_SetString(PyExc_ValueError,
		      "covariance_matrix requires at least two samples");
      return -1;
    }
    if (m < 1) {
      PyErr_SetString(PyExc_ValueError,
		      "covariance_matrix requires at least one feature");
      return -1;
    }
    return 0;
  }
}

#endif  // UFUNCLAB_MEANVAR_GUFUNC_H
