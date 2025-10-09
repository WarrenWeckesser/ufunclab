#ifndef SEMIVAR_H
#define SEMIVAR_H

#include <cmath>

template<typename Real>
Real semivar_exponential(Real h, Real nugget, Real sill, Real range)
{
    return nugget + (sill - nugget) * (1 - std::exp(-3*h / range));
}

template<typename Real>
Real semivar_linear(Real h, Real nugget, Real sill, Real range)
{
    Real v;

    if (h > range) {
        v = sill;
    }
    else {
        Real slope = (sill - nugget) / range;
        v = nugget + slope * h;
    }
    return v;
}

template<typename Real>
Real semivar_spherical(Real h, Real nugget, Real sill, Real range)
{
    Real v, hor;

    if (h > range) {
        v = sill;
    }
    else {
        hor = h / range;
        v = nugget + (sill - nugget) * (hor / 2.0) * (3.0 - hor*hor);
    }
    return v;
}

template<typename Real>
Real semivar_parabolic(Real h, Real nugget, Real sill, Real range)
{
    Real v, hor;

    if (h > range) {
        v = sill;
    }
    else {
        hor = h / range;
        v = nugget + (sill - nugget)*hor*(2.0 - hor);
    }
    return v;
}

#endif
