
#include <complex>

template<typename T>
T abs_squared(T x)
{
    return x*x;
}

template<typename T>
T abs_squared(std::complex<T> z)
{
    return z.real()*z.real() + z.imag()*z.imag();
}
