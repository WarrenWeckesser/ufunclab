#include <cmath>


template<typename T>
T log_expit(T x)
{
    if (x < 0.0) {
        return x - std::log1p(std::exp(x));
    }
    else {
        return -std::log1p(std::exp(-x));
    }
}
