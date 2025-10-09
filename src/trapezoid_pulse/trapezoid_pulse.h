#ifndef TRAPEZOID_PULSE_H
#define TRAPEZOID_PULSE_H

#include <cmath>
#include <limits>

template<typename T>
T trapezoid_pulse(T x, T a, T b, T c, T d, T amp)
{
    T result;
    if (!((a <= b) && (b <= c) && (c <= d))) {
        result = std::numeric_limits<T>::quiet_NaN();
    }
    else {
        if ((x <= a) || (x >= d)) {
            result = 0.0;
        }
        else if ((x >= b) && (x <= c)) {
            result = amp;
        }
        else if (x < b) {
            result = amp*((x - a)/(b - a));
        }
        else {
            result = amp*((d - x)/(d - c));
        }
    }
    return result;
}

#endif
