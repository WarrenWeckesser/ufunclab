#include <cmath>


//
// The "standard" hyperbolic ramp:
//
//   f(x) = x/2 + sqrt((x/2)**2 + 1)              (1)
//        = (x/2)*(1 + sqrt(1 + (2/x)**2))        (2)
//        = 1 / (sqrt((x/2)**2 + 1) - z)          (3)
//        = -1 / ((x/2)*(sqrt(1 + (2/x)**2) + 1)  (4)
//
// The formula depends on x, so that occurrences of
// overflow and subtractive cancellation are avoided.
//
template<typename T>
T std_hyperbolic_ramp(T x)
{
    if (x == 0) {
        return 1;
    }

    T z = x/2;

    if (z >= 1) {
        return z*(1 + std::sqrt(1 + 1.0/(z*z)));
    }
    else if (z > 0) {
        return z + std::sqrt(z*z + 1);
    }
    else if (z > -1) {
        return 1.0/(std::sqrt(z*z + 1) - z);
    }
    else {  // z <= -1
        return -1.0/z/(std::sqrt(1 + 1.0/(z*z)) + 1);
    }
}

//
// The hyperbolic ramp for a > 0 is
//
//   f(x, a) = 0.5*(x + sqrt(x**2 + 4*a**))
//
// The constant 4 is included so that f(0, a) = 1
// for all a.  API note: The function satisfies
//
//   f(x, a) = a*f(x/a, 1)
//
// so, in principle, the parameter a could be removed,
// and the user could achieve the same result by scaling
// the input and output, but we'll keep it for convenience.
// The "standard" version with a=1 is defined above, and
// it is used by this function after validating a.
//
template<typename T>
T hyperbolic_ramp(T x, T a)
{
    if (a == 0) {
        return (x < 0) ? 0.0 : x;
    } else if (a < 0) {
        return NAN;
    } else {
        return a*std_hyperbolic_ramp(x/a);
    }
}
