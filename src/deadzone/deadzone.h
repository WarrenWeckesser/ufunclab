#ifndef UFUNCLAB_DEADZONE_H
#define UFUNCLAB_DEADZONE_H

template<typename T>
T deadzone(T x, T low, T high)
{
    if (x < low) {
        return x - low;
    }
    else if (x > high) {
        return x - high;
    }
    else {
        return 0;
    }
}

#endif  // UFUNCLAB_DEADZONE_H
