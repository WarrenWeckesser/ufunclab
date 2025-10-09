#ifndef NORMAL_H
#define NORMAL_H

#include <cmath>
#include <limits>
#include "erfcx_funcs.h"

#define RECIP_SQRT2 0.7071067811865475244008443621048490393L


template<typename T>
T normal_cdf(T x)
{
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T x2 = x * RECIP_SQRT2;
    if (x2 > 0) {
        return 1 - std::erfc(x2)/2;
    }
    else {
        return std::erfc(-x2)/2;
    }
}

//
// The survival function (aka complementary CDF) of the standard
// normal distribution.
//
// The function computes 1 - CDF(x).
//
template<typename T>
T normal_sf(T x)
{
    // Symmetry
    return normal_cdf(-x);
}

/*
 * Log of the CDF of the standard normal distribution.
 *
 * Let F(x) be the CDF of the standard normal distribution.
 * This implementation of log(F(x)) is based on the identities
 *
 *   F(x) = erfc(-x/√2)/2
 *        = 1 - erfc(x/√2)/2
 *
 * We use the first formula for x < -1, with erfc(z) replaced
 * by erfcx(z)*exp(-z**2) to ensure high precision for large
 * negative values when we take the logarithm:
 *
 *   log F(x) = log(erfc(-x/√2)/2)
 *            = log(erfcx(-x/√2)/2)*exp(-x**2/2))
 *            = log(erfcx(-x/√2)/2) - x**2/2
 *
 * For x >= -1, we use the second formula for F(x):
 *
 *   log F(x) = log(1 - erfc(x/√2)/2)
 *            = log1p(-erfc(x/√2)/2)
 */

template<typename T>
T normal_logcdf(T x)
{
    T t = x*RECIP_SQRT2;
    if (x < -1.0) {
        return std::log(erfcx(-t)/2) - t*t;
    }
    else {
        return std::log1p(-std::erfc(t)/2);
    }
}


//
// Logarithm of the survival function (aka complementary CDF) of the
// standard normal distribution.
//
template<typename T>
T normal_logsf(T x)
{
    // Symmetry
    return normal_logcdf(-x);
}

#endif
