#include <cmath>


template<typename T>
T hyperbolic_ramp(T x, T a)
{
    if (a == 0) {
        return (x < 0) ? 0.0 : x;
    } else {
        T a2 = a*a;
        T d = std::sqrt(x*x + 4*a2);
        return (x < 0) ? 2*a2/(d - x) : (x + d)/2;
    }
}
