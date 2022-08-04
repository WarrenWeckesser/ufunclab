#ifndef STRIDED_HPP
#define STRIDED_HPP

#include <stddef.h>

// XXX To do: replace C-style casts with C++ casts.

// XXX ptrdiff_t used instead of npy_intp.

template<typename T>
static inline T
get(T *px, ptrdiff_t stride, ptrdiff_t index)
{
    return (*((T *) ((char *) px + index*stride)));
}

template<typename T>
static inline void
set(T *px, ptrdiff_t stride, ptrdiff_t index, T value)
{
    (*((T *) ((char *) px + index*stride))) = value;
}

#endif