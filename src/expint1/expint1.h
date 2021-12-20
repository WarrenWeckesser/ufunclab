#include <cmath>

#define EULER 0.577215664901532860606512090082402431042159L


template<typename T>
static T expint1_01(T x)
{
    T e1 = 1.0;
    T r = 1.0;
    for (int k = 1; k < 26; ++k) {
        r = -r * k * x/((k + 1)*(k + 1));
        if (e1 + r == e1) {
            break;
        }
        e1 = e1 + r;
    }
    e1 = -EULER - log(x) + x*e1;
    return e1;
}

// Compute a factor of the result of the exponential integral E1.
// This is used in expint1 when x > 1.
// If t = expint1_t(x), then expint1(x) is exp(-x)*t.
//
// Uses the continued fraction expansion.
//
// This calculation is implemented as a separate function so it
// can be used in both expint1 and logexpint1.

template<typename T>
static T expint1_t(T x)
{
    int m = 20 + (int) ((T) 80.0 / x);
    T t0 = 0.0;
    for (int k = m; k > 0; --k) {
        t0 = k/(1 + k/(x + t0));
    }
    return 1/(x + t0);
}



// The exponential integral E1 for real x > 0.
//
// Returns inf if x = 0, and nan if x < 0.

template<typename T>
T expint1(T x)
{
    if (x < 0.0) {
        return NAN;
    }
    if (x == 0) {
        return INFINITY;
    }
    if (x <= 1.0) {
        return expint1_01(x);
    }
    // else x > 1
    T t = expint1_t(x);
    return exp(-x) * t;
}


// Log of the exponential integral function E1 (for real x only).

template<typename T>
T logexpint1(T x)
{
    if (x < 0) {
        return NAN;
    }
    if (x == 0) {
        return INFINITY;
    }
    if (x <= 1.0) {
        return log(expint1_01(x));
    }
    if (x <= 500) {
        T t = expint1_t(x);
        return -x + log(t);
    }
    T s = (-1 + (2 + (-6 + (24 - 120/x)/x)/x)/x)/x;
    return -x - log(x) + log1p(s);
}
