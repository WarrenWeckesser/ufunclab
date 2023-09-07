#ifndef POW1PM1_H
#define POW1PM1_H

#include <cmath>

//
// Compute (1 + x)**y - 1.
//
// Requires x > -1.
//
template<typename T>
T pow1pm1(T x, T y)
{
    // XXX Maybe handle (x, y) == (-1, 0) as a special case, and return 0
    // instead of nan?
    return std::expm1(y * std::log1p(x));
}

#endif
