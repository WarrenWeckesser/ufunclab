#ifndef STRIDED_HPP
#define STRIDED_HPP

#include <stddef.h>

// XXX To do: replace C-style casts with C++ casts.

// XXX ptrdiff_t used instead of npy_intp.

template<typename T>
static inline T
get(const T *px, ptrdiff_t stride, ptrdiff_t index)
{
    return (*((T *) ((char *) px + index*stride)));
}

template<typename T>
static inline void
set(T *px, ptrdiff_t stride, ptrdiff_t index, T value)
{
    (*((T *) ((char *) px + index*stride))) = value;
}

template<typename T>
static inline T
get2d(const T *px, const ptrdiff_t *strides, ptrdiff_t i, ptrdiff_t j)
{
    return (*((T *) ((char *) px + i*strides[0] + j*strides[1])));
}

template<typename T>
static inline void
set2d(T *px, const ptrdiff_t *strides, ptrdiff_t i, ptrdiff_t j, T value)
{
    (*((T *) ((char *) px + i*strides[0] + j*strides[1]))) = value;
}

#endif