#ifndef NEXT_H
#define NEXT_H

#include <cmath>


namespace NextFunctions {

template<typename T>
T next_greater(T x)
{
    T to = INFINITY;
    return std::nextafter(x, to);
}

template<typename T>
T next_less(T x)
{
    T to = -INFINITY;
    return std::nextafter(x, to);
}

}  // namespace NextFunctions

#endif
