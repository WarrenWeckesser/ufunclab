#include <cmath>

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
        return NAN;
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
        return NAN;
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
T smoothstep5(T x, T a, T b, T fa, T fb)
{
    if (a > b) {
        return NAN;
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
