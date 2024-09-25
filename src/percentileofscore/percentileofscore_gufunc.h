
#ifndef PERCENTILEOFSCORE_GUFUNC_H
#define PERCENTILEOFSCORE_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


enum percentileofscore_kind: int {
    RANK = 0,
    WEAK,
    STRICT,
    MEAN,
};

template<typename T>
static inline void
count(npy_intp n,
      const T *p_x,
      npy_intp stride,
      const T score,
      npy_intp *nless,
      npy_intp *nequal,
      npy_intp *ngreater)
{
    *nless = 0;
    *nequal = 0;
    *ngreater = 0;
    for (npy_intp k = 0; k < n; ++k) {
        T value = get(p_x, stride, k);
        if (value < score) {
            ++*nless;
        }
        else if(value == score) {
            ++*nequal;
        }
        else {
            ++*ngreater;
        }
    }
}

//
// `percentileofscore_core_calc`, the C++ core function
// for the gufunc `percentileofscore` with signature '(n),(),()->()'.
//
template<typename T>
static void
percentileofscore_core_calc(
    npy_intp n,         // core dimension n
    T *p_x,             // pointer to first element of x, a strided 1-d array with n elements
    npy_intp x_stride,  // stride (in bytes) for elements of x
    T *p_score,
    npy_intp *p_kind,
    double *p_out
)
{
    npy_intp nless, nequal, ngreater, sum;

    count(n, p_x, x_stride, *p_score, &nless, &nequal, &ngreater);

    switch (static_cast<int>(*p_kind)) {
        case RANK:
            sum = 2*nless + nequal;
            if (nequal > 0) {
                ++sum;
            }
            *p_out = sum*50.0/n;
            return;
        case WEAK:
            *p_out = (nless + nequal)*100.0/n;
            return;
        case STRICT:
            *p_out = nless*100.0/n;
            return;
        case MEAN:
            *p_out = (2*nless + nequal)*50.0/n;
            return;
    }
}

//--------------------------------------------------------

//
// The name of this function is listed in the `extra_module_funcs`
// attribute of the UFuncExtMod object that defines the gufuncs
// `first` and `argfirst`.  A call of this function will be added
// to the end of the generated extension module.
//
int add_percentileofscore_kind_constants(PyObject *module)
{
    // Expose the numerical values RANK, WEAK, STRICT and MEAN as integers
    // in this module.
    const char *opnames[] = {"_RANK", "_WEAK", "_STRICT", "_MEAN"};
    const int opcodes[] = {RANK, WEAK, STRICT, MEAN};
    for (int k = 0; k < 4; ++k) {
        int status = PyModule_AddIntConstant(module, opnames[k], opcodes[k]);
        if (status == -1) {
            return -1;
        }
    }
    return 0;
}

#endif