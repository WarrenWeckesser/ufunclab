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

template <typename Real, size_t N_accumulators=10>
class pairwise_accumulator {
private:
  Real accumulators[N_accumulators] = {0};
  std::size_t n_elements = 0, accum_index = 0;

public:
  inline void reset() {
    for (size_t i = 0; i < N_accumulators; ++i) {
      accumulators[i] = 0;
    }
    n_elements = 0;
    accum_index = 0;
  }

  inline void add_data(Real num) {
    ++n_elements;
    accumulators[accum_index] += num;
    for (std::size_t tst = 1;
	 tst < n_elements && (n_elements & tst) == 0 && accum_index > 0;
	 tst <<= 1, --accum_index) {
      accumulators[accum_index - 1] += accumulators[accum_index];
      accumulators[accum_index] = 0;
    }
    ++accum_index;
    if (accum_index == N_accumulators) {
      Real tmp = 0;
      for (size_t i = N_accumulators - 1; i > 0; --i) {
	tmp += accumulators[i];
	accumulators[i] = 0;
      }
      accumulators[0] += tmp;
      accum_index = 1;
    }
  }

  constexpr inline Real get_sum() {
    Real result = 0;
    for (long i = N_accumulators - 1; i >= 0; --i) {
      result += accumulators[i];
    }
    return result;
  }
};

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
        const T * const p_x,          // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,    // stride (in bytes) for elements of x
        const npy_intp * const p_ddof,   // pointer to ddof
        U *p_out,             // pointer to first element of out, a strided 1-d array with 2 elements
        npy_intp out_stride   // stride (in bytes) for elements of out
)
{
  pairwise_accumulator<U, 10> accumulator;
    U mean = 0.0;
    U var = 0.0, tmp;
    for (npy_intp k = 0; k < n; ++k) {
      T xk = get(p_x, x_stride, k);
      accumulator.add_data(xk);
    }
    mean = accumulator.get_sum() / n;
    accumulator.reset();
    for (npy_intp k = 0; k < n; ++k) {
      T xk = get(p_x, x_stride, k);
      tmp = xk - mean;
      accumulator.add_data(tmp * tmp);
    }
    var = accumulator.get_sum() / (n - *p_ddof);
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
  pairwise_accumulator<U, 10> mean_accum[m], cov_accum[m*m];
  U tmp1, tmp2;
  for (npy_intp i = 0; i < n; ++i) {
    #pragma GCC ivdep
    for (npy_intp j = 0; j < m; ++j) {
      T x_ij = get2d(p_x, x_strides, i, j);
      mean_accum[j].add_data(x_ij);
    }
  }
  for (npy_intp j = 0; j < m; ++j) {
    mean[j] = mean_accum[j].get_sum() / n;
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
	cov_accum[j * m + k].add_data(tmp1 * tmp2);
      }
    }
  }
  for (npy_intp j = 0; j < m; ++j) {
    for (npy_intp k = j; k < m; ++k) {
      tmp1 = cov_accum[j * m + k].get_sum() / (n - *p_ddof);
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
