#ifndef UFUNCLAB_STEP_H
#define UFUNCLAB_STEP_H

#include <cmath>
#include <limits>


namespace StepFunctions {

template<typename T>
T step(T x, T a, T flow, T fa, T fhigh)
{
    if (x < a) {
        return flow;
    }
    else if (x > a) {
        return fhigh;
    }
    else {
        return fa;
    }
}

template<typename T>
T linearstep(T x, T a, T b, T fa, T fb)
{
    if (a > b) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x <= a) {
        return fa;
    }
    if (x >= b) {
        return fb;
    }
    if (a == b) {
        return (fa + fb) / 2;
    }
    T u = (x - a)/(b - a);
    return fa*(1 - u) + fb*u;
}

/* smoothstep3 docstring:
smoothstep3(x, a, b, fa, fb)

The function returns fa for x < a, fb for x > b,
and interpolates between a and b with a cubic
polynomial for which f'(a) = 0 and f'(b) = 0.
*/

template<typename T>
T smoothstep3(T x, T a, T b, T fa, T fb)
{
    if (a > b) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x <= a) {
        return fa;
    }
    if (x >= b) {
        return fb;
    }
    if (a == b) {
        return (fa + fb) / 2;
    }
    T u = (x - a)/(b - a);
    return fa + (fb - fa)*u*u*(3 - 2*u);
}

template<typename T>
T invsmoothstep3(T y, T a, T b, T fa, T fb)
{
    T upper, lower;

    if (fa == fb) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (y == fa) {
        return a;
    }
    if (y == fb) {
        return b;
    }
    if (fa < fb) {
        upper = fb;
        lower = fa;
    }
    else {
        upper = fa;
        lower = fb;
    }
    if ((y > upper) || (y < lower)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    // See, for example,
    //     https://en.wikipedia.org/wiki/Smoothstep#Inverse_Smoothstep
    T t = (y - fa) / (fb - fa);
    T s = 0.5 - std::sin(std::asin(1 - 2 * t) / 3);
    return a + (b - a)*s;
}


template<typename T>
T smoothstep5(T x, T a, T b, T fa, T fb)
{
    if (a > b) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x <= a) {
        return fa;
    }
    if (x >= b) {
        return fb;
    }
    if (a == b) {
        return (fa + fb) / 2;
    }
    T u = (x - a)/(b - a);
    return fa + (fb - fa)*u*u*u*(u*(6*u - 15) + 10);
}

}  // namespace StepFunctions

#endif  // UFUNCLAB_STEP_H
