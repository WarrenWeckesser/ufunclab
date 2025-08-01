#ifndef NAN_TO_NUM_H
#define NAN_TO_NUM_H

#include <cmath>

template<typename T>
inline T nan_to_num(T x, T replacement)
{
    return std::isnan(x) ? replacement : x;
}

#endif
