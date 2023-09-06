#ifndef SOSFILTER_GUFUNC_H
#define SOSFILTER_GUFUNC_H

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"

#include "../src/util/strided.hpp"


template<typename T>
static void sosfilter_ic_core(
        npy_intp m,                     // core dimension m
        npy_intp n,                     // core dimension n
        T *p_sos,                       // pointer to first element of sos, a strided 2-d array with shape (m, 6)
        const npy_intp sos_strides[2],  // array of length 2 of strides (in bytes) of sos
        T *p_x,                         // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,              // stride (in bytes) for elements of x
        T *p_zi,                        // pointer to first element of zi, a strided 2-d array with shape (m, 2)
        const npy_intp zi_strides[2],   // array of length 2 of strides (in bytes) of zi
        T *p_y,                         // pointer to first element of y (the output array), a strided 1-d array with n elements
        npy_intp y_stride,              // stride (in bytes) for elements of y
        T *p_zf,                        // pointer to first element of zf (the output array), a strided 2-d array with shape (m, 2)
        const npy_intp zf_strides[2]    // array of length 2 of strides (in bytes) of zf
)
{
    // Copy zi to zf
    if (p_zi != p_zf) {
        for (npy_intp k = 0; k < m; ++k) {
            T value = get2d(p_zi, zi_strides, k, 0);
            set2d(p_zf, zf_strides, k, 0, value);
            value = get2d(p_zi, zi_strides, k, 1);
            set2d(p_zf, zf_strides, k, 1, value);
        }
    }

    if (x_stride == sizeof(T) && y_stride == sizeof(T)) {
        // Slight optimization when x and y are both contiguous arrays.
        for (npy_intp i = 0; i < n; ++i) {
            T xi = p_x[i];
            for (npy_intp k = 0; k < m; ++k) {

                // y[i] = sos[k, 0]*xi + zf[k, 0];
                T yi = get2d(p_sos, sos_strides, k, 0) * xi + get2d(p_zf, zf_strides, k, 0);
                p_y[i] = yi;

                // zf[k, 0] = zf[k, 1] + sos[k, 1]*xi - sos[k, 4]*y[i];
                T tmp1 = get2d(p_zf, zf_strides, k, 1) + get2d(p_sos, sos_strides, k, 1)*xi
                         - (get2d(p_sos, sos_strides, k, 4)*yi);
                set2d(p_zf, zf_strides, k, 0, tmp1);

                // zf[k, 1] = sos[k, 2]*xi - sos[k, 5]*y[i];
                T tmp2 = get2d(p_sos, sos_strides, k, 2) * xi - (get2d(p_sos, sos_strides, k, 5)*yi);
                set2d(p_zf, zf_strides, k, 1, tmp2);

                xi = yi;
            }
        }
    }
    else {
        for (npy_intp i = 0; i < n; ++i) {
            T xi = get(p_x, x_stride, i);
            for (npy_intp k = 0; k < m; ++k) {

                // y[i] = sos[k, 0]*xi + zf[k, 0];
                T yi = get2d(p_sos, sos_strides, k, 0) * xi +get2d(p_zf, zf_strides, k, 0);
                set(p_y, y_stride, i, yi);

                // zf[k, 0] = zf[k, 1] + sos[k, 1]*xi - sos[k, 4]*y[i];
                T tmp1 = get2d(p_zf, zf_strides, k, 1) + get2d(p_sos, sos_strides, k, 1)*xi
                         - (get2d(p_sos, sos_strides, k, 4)*yi);
                set2d(p_zf, zf_strides, k, 0, tmp1);

                // zf[k, 1] = sos[k, 2]*xi - sos[k, 5]*y[i];
                T tmp2 = get2d(p_sos, sos_strides, k, 2) * xi - (get2d(p_sos, sos_strides, k, 5)*yi);
                set2d(p_zf, zf_strides, k, 1, tmp2);

                xi = yi;
            }
        }
    }
}

// In sosfilter_ic_contig_core, it is assumed that all arrays
// are C-contiguous.  Passing noncontiguous arrays will result
// in incorrect output and might crash the program.
template<typename T>
static void sosfilter_ic_contig_core(
        npy_intp m,                     // core dimension m
        npy_intp n,                     // core dimension n
        T *p_sos,                       // pointer to first element of sos, a strided 2-d array with shape (m, 6)
        const npy_intp sos_strides[2],  // array of length 2 of strides (in bytes) of sos
        T *p_x,                         // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,              // stride (in bytes) for elements of x
        T *p_zi,                        // pointer to first element of zi, a strided 2-d array with shape (m, 2)
        const npy_intp zi_strides[2],   // array of length 2 of strides (in bytes) of zi
        T *p_y,                         // pointer to first element of y (the output array), a strided 1-d array with n elements
        npy_intp y_stride,              // stride (in bytes) for elements of y
        T *p_zf,                        // pointer to first element of zf (the output array), a strided 2-d array with shape (m, 2)
        const npy_intp zf_strides[2]    // array of length 2 of strides (in bytes) of zf
)
{
    if (p_zf != p_zi) {
        // Copy zi to zf
        memcpy(p_zf, p_zi, 2*m*sizeof(T));
    }

    for (npy_intp i = 0; i < n; ++i) {
        // T xi = get(p_x, x_stride, i);
        T xi = p_x[i];
        for (npy_intp k = 0; k < m; ++k) {

            // y[i] = sos[k, 0]*xi + zf[k, 0];
            T yi = (*(p_sos + 6*k))*xi + (*(p_zf + 2*k));
            p_y[i] = yi;

            // zf[k, 0] = zf[k, 1] + sos[k, 1]*xi - sos[k, 4]*y[i];
            *(p_zf + 2*k) = (*(p_zf + 2*k + 1)) + (*(p_sos + 6*k + 1))*xi - (*(p_sos + 6*k + 4))*yi;

            // zf[k, 1] = sos[k, 2]*xi - sos[k, 5]*y[i];
            *(p_zf + 2*k + 1) = (*(p_sos + 6*k + 2))*xi - (*(p_sos + 6*k + 5))*yi;

            xi = yi;
        }
    }
}

template<typename T>
static void sosfilter_core(
        npy_intp m,                     // core dimension m
        npy_intp n,                     // core dimension n
        T *p_sos,                       // pointer to first element of sos, a strided 2-d array with shape (m, 6)
        const npy_intp sos_strides[2],  // array of length 2 of strides (in bytes) of sos
        T *p_x,                         // pointer to first element of x, a strided 1-d array with n elements
        npy_intp x_stride,              // stride (in bytes) for elements of x
        T *p_out,                       // pointer to first element of out, a strided 1-d array with n elements
        npy_intp out_stride             // stride (in bytes) for elements of out
)
{
    npy_intp zi_strides[2];

    // XXX FIXME: Check the return value of calloc for NULL.
    // What to do if calloc() fails?
    T *zi = (T *) calloc(m*2, sizeof(T));
    zi_strides[0] = 2*sizeof(T);
    zi_strides[1] = sizeof(T);
    if (sos_strides[0] == 6*sizeof(T) && sos_strides[1] == sizeof(T)
            && x_stride == sizeof(T) && out_stride == sizeof(T)) {
        // All arrays are contiguous; use the faster version.
        sosfilter_ic_contig_core(m, n,
                                 p_sos, sos_strides, p_x, x_stride, zi, zi_strides,
                                 p_out, out_stride, zi, zi_strides);
    }
    else {
        sosfilter_ic_core(m, n,
                          p_sos, sos_strides, p_x, x_stride, zi, zi_strides,
                          p_out, out_stride, zi, zi_strides);
    }
    free(zi);
}

#endif
