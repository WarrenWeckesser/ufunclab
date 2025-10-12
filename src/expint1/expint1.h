#ifndef UFUNCLAB_EXPINT1_H
#define UFUNCLAB_EXPINT1_H

#include <cmath>
#include <limits>

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
    e1 = -EULER - std::log(x) + x*e1;
    return e1;
}

//
// Compute a factor of the result of the exponential integral E1.
// This is used in expint1(x) for x > 1 and in logexp1(x) for 1 < x <= 500.
//
// The function uses the continued fraction expansion given in equation 5.1.22
// of Abramowitz & Stegun, "Handbook of Mathematical Functions".
// For n=1, this is
//    E1(x) = exp(-x)*F(x)
// where F(x) is expressed as a continued fraction:
//    F(x) =                 1
//           ---------------------------------------
//                              1
//           x + ------------------------------------
//                                 1
//               1 + ---------------------------------
//                                    2
//                   x + ------------------------------
//                                       2
//                       1 + ---------------------------
//                                          3
//                           x + ------------------------
//                                             3
//                               1 + ---------------------
//                                                4
//                                   x + ------------------
//                                       1 +     [...]
//
template<typename T>
static T expint1_t(T x)
{
    // The number of terms to use in the truncated continued fraction
    // depends on x.  Larger values of x require fewer terms.
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
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == 0) {
        return std::numeric_limits<T>::infinity();
    }
    if (x <= 1.0) {
        return expint1_01(x);
    }
    // else x > 1
    T t = expint1_t(x);
    return std::exp(-x) * t;
}


// Log of the exponential integral function E1 (for real x only).

template<typename T>
T logexpint1(T x)
{
    if (x < 0) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == 0) {
        return std::numeric_limits<T>::infinity();
    }
    if (x <= 1.0) {
        // For small x, the naive implementation is sufficiently accurate.
        return std::log(expint1_01(x));
    }
    if (x <= 500) {
        // For moderate x, use the continued fraction expansion.
        T t = expint1_t(x);
        return -x + std::log(t);
    }
    // For large x, use the asymptotic expansion.  This is equation 5.1.51
    // from Abramowitz & Stegun, "Handbook of Mathematical Functions".
    T s = (-1 + (2 + (-6 + (24 - 120/x)/x)/x)/x)/x;
    return -x - std::log(x) + std::log1p(s);
}

#endif  // UFUNCLAB_EXPINT1_H
