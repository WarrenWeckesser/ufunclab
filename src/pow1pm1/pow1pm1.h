#ifndef UFUNCLAB_POW1PM1_H
#define UFUNCLAB_POW1PM1_H

#include <cmath>

//
// Compute (1 + x)**y - 1.
//
// Requires x > -1.
//
template<typename T>
T pow1pm1(T x, T y)
{
    if (x == -1.0 && y == 0.0) {
        return static_cast<T>(0.0);
    }
    return std::expm1(y * std::log1p(x));
}

#endif  // UFUNCLAB_POW1PM1_H
