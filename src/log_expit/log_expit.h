#include <cmath>


template<typename T>
T log_expit(T x)
{
    if (x < 0.0) {
        return x - log1p(exp(x));
    }
    else {
        return -log1p(exp(-x));
    }
}
