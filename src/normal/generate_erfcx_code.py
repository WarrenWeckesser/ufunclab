
import warnings
import numpy as np
import mpmath


mpmath.mp.dps = 200


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def erfcx_mp(x):
    x = mpmath.mpf(x)
    return mpmath.erfc(x) * mpmath.exp(x*x)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def find_lower_bound(dtype):
    """
    For the given dtype, find the lowest value of x for which erfcx(x)
    can be represented as a finite number with dtype.  In the generated
    code, erfcx(x) will return inf for any input below this lower bound.
    """
    x0 = dtype(-1.0)
    x1 = dtype(-150.0)
    while True:
        xmid = (x0 + x1)/dtype(2)
        if xmid == x0 or xmid == x1:
            xmid = x0
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ymid = dtype(str(erfcx_mp(mpmath.mpf(str(xmid)))))
        if np.isinf(ymid):
            x1 = xmid
        else:
            x0 = xmid
    return xmid


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Use mpmath.chebyfit to create the Chebyshev polynomials used by
# Steven G. Johnson in erfcx_y100.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def func(t, k):
    # k must be an integer in [0, 100)
    t = mpmath.mpf(t)
    y100 = (t + 2*k + 1)/2
    y = y100/100
    x = 4/y - 4
    return mpmath.erfc(x)*mpmath.exp(x**2)


def getpoly(k, N=8):
    # N is the number of terms in the Chebyshev polynomial.
    # (The degree of the polynomial is N - 1.)
    poly, err = mpmath.chebyfit(lambda t: func(t, k), [-1, 1], N, error=True)
    return poly, err


_erfcx_y100_start = """

static {type_str}
erfcx_y100({type_str} y100)
{{
    {type_str} t;

    switch ((int) y100) {{
"""

_erfcx_y100_end = """
    }
    // We only get here if y = 1, i.e. |x| < 4*eps, in which case
    // erfcx(x) is numerically 1.
    return 1.0;
}
"""

# The values in the `_float_type_info` dictionary are:
#    :: numpy type
#    :: C literal suffix
#    :: bound after which to use the truncated continued fraction formula
#    :: Number of terms to use in the Chebyshev polynomials in erfcx_y100.
#       (The degree of the polynomials is N - 1.)
#
_float_type_info = {'float': (np.float32, 'f', 5.0, 4),
                    'double': (np.float64, '', 50.0, 7),
                    'long double': (np.longdouble, 'L', 100.0, 8)}


def generate_erfcx_y100(type_str):
    dtype, sfx, cfbound, N = _float_type_info[type_str]
    mpmath.mp.dps = int(1.35*np.finfo(dtype).precision)
    out = [_erfcx_y100_start.format(type_str=type_str)]
    errs = []
    for k in range(100):
        poly, err = getpoly(k, N)
        errs.append(err)
        out.append(f'    case {k}:')
        out.append(f"        t = 2.0{sfx}*y100 - {2*k+1};")
        s = f"{poly[0]}{sfx}"
        for c in poly[1:]:
            s = f"{c}{sfx} + \n               ({s})*t"
        out.append(f'        return {s};')
    out.append(_erfcx_y100_end)
    print(f"{type_str} max(errs): {max(errs)}")
    return '\n'.join(out), errs


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generate the main erfcx functions.  Each has the same structure as the
# erfcx function in Steven G. Johnson's implementation, but the coefficients
# and thresholds updated to appropriate values for each floating point type.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_erfcx_body = """
{type_str}
erfcx({type_str} x)
{{
    if (x > 0.0{sfx}) {{
        if (x > {cfbound}{sfx}) {{
            // Use the truncated continued fraction expansion.
            // See Abramowitz & Stegun 7.1.14 (combined with the definition in 7.1.2).
            // ispi = 1/sqrt(pi)
            const {type_str} ispi = 0.5641895835477562869480794515607725858{sfx};
            return ispi / (x + 0.5{sfx} / (x + 1.0{sfx} / (x + 1.5{sfx} / (x + 2.0{sfx}/x))));
        }}
        else {{
            return erfcx_y100(400.0{sfx}/(4.0{sfx} + x));
        }}
    }}
    else {{
       // x <= 0
       if (x < {lower_bound}{sfx}) {{
           return INFINITY;
       }} else {{
           // Use the identity erfcx(x) = 2*exp(x**2) - erfcx(-x)
           return 2.0{sfx}*std::exp(x*x) - erfcx_y100(400.0{sfx}/(4.0{sfx} - x));
       }}
    }}
}}
"""


def generate_erfcx(type_str):
    dtype, sfx, cfbound, N = _float_type_info[type_str]
    lower_bound = find_lower_bound(dtype)
    return _erfcx_body.format(type_str=type_str, sfx=sfx, cfbound=str(cfbound),
                              lower_bound=str(lower_bound))


_autogen_msg = """
//
// This code was generated automatically.  Do not edit!
//
"""

_preamble = """
//
// The scaled complementary error function erfcx(x) is defined as
//     erfcx(x) = erfc(x)*exp(x*x)
//
// This file has these implementations of erfcx:
//     float erfcx(float x)
//     double erfcx(double x)
//     long double erfcx(long double)
//

#include <cmath>
"""

if __name__ == "__main__":
    with open('erfcx_funcs.cxx', 'w') as f:
        f.write(_autogen_msg)
        f.write(_preamble)
        for typ, (dtype, sfx, cfbound, N) in _float_type_info.items():
            content, errs = generate_erfcx_y100(typ)
            f.write(content)
            content = generate_erfcx(typ)
            f.write(content)
            f.write('\n')
    with open('erfcx_funcs.h', 'w') as f:
        f.write(_autogen_msg)
        f.write('#ifndef ERFCX_H_\n')
        f.write('#define ERFCX_H_\n\n')
        for typ, (dtype, sfx, cfbound, N) in _float_type_info.items():
            f.write(f'{typ} erfcx({typ});\n')
        f.write('\n#endif\n')
