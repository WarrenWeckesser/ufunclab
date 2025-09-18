#ifndef HYPOT3_H
#define HYPOT3_H

#include <cmath>

#define RECIP_SQRT2 0.7071067811865475244008443621048490393L


//
// XXX/FIXME: This wrapper shouldn't be necessary!
//
template<typename T>
T hypot3(T x, T y, T z)
{
    return std::hypot(x, y, z);
}

#endif
