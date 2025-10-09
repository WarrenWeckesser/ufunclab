#ifndef NEXT_H
#define NEXT_H

#include <cmath>
#include <limits>


namespace NextFunctions {

template<typename T>
T next_greater(T x)
{
    T to = std::numeric_limits<T>::infinity();
    return std::nextafter(x, to);
}

template<typename T>
T next_less(T x)
{
    T to = -std::numeric_limits<T>::infinity();
    return std::nextafter(x, to);
}

}  // namespace NextFunctions

#endif
