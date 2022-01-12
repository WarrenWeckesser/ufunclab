#include <cmath>

template<typename T>
T yeo_johnson(T x, T lmbda)
{
    if (x >= 0.0) {
        if (lmbda == 0.0) {
            return std::log1p(x);
        }
        else {
            return std::expm1(lmbda*std::log1p(x))/lmbda;
        }
    }
    else {
        if (lmbda == 2.0) {
            return -std::log1p(-x);
        }
        else {
            return -std::expm1((2 - lmbda)*std::log1p(-x))/(2 - lmbda);
        }
    }
}

template<typename T>
T inv_yeo_johnson(T x, T lmbda)
{
    if (x >= 0.0) {
        if (lmbda == 0.0) {
            return std::expm1(x);
        }
        else {
            if (lmbda < 0 && x >= -1/lmbda) {
                return x > -1/lmbda ? NAN : INFINITY;
            }
            else {
                return std::expm1(std::log1p(lmbda*x)/lmbda);
            }
        }
    }
    else {
        // x < 0
        if (lmbda == 2.0) {
            return -std::expm1(-x);
        }
        else {
            if (lmbda > 2 && x <= 1/(2 - lmbda)) {
                return x < 1/(2 - lmbda) ? NAN : -INFINITY;
            }
            else {
                return -std::expm1(std::log1p(-(2 - lmbda)*x)/(2 - lmbda));
            }
        }
    }
}
