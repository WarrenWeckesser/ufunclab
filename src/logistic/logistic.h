#include <cmath>


template<typename T>
inline T logistic(T x)
{
    return 1 / (1 + std::exp(-x));
}


template<typename T>
inline T logistic_deriv(T x)
{
    return logistic(x) * logistic(-x);
}


template<typename T>
T log_logistic(T x)
{
    if (x < 0.0) {
        return x - std::log1p(std::exp(x));
    }
    else {
        return -std::log1p(std::exp(-x));
    }
}
