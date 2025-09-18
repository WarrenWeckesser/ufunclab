#ifndef HYPOT3_H
#define HYPOT3_H

#include <cmath>
#include <limits>


template<typename T>
T hypot3(T x, T y, T z)
{
    //
    // Not all standard libraries handle inf as expected; std::hypot(x, y, z)
    // might return nan when an input is inf.  So we check for inf in the
    // wrapper to maintain consistent behavior across different libs.
    //
    // When at least one input is inf, this function returns inf (even
    // if another input is nan).
    //
    if (std::isinf(x) || std::isinf(y) || std::isinf(z)) {
        return std::numeric_limits<T>::infinity();
    }
    return std::hypot(x, y, z);
}

#endif
