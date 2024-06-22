#include <math.h>

////////////////////////////////////////////////////////////////////////////
//
//  The code from here to the corresponding comment delimited by lines of
//  forward slashes below was taken from the Python source code file
//  Python-3.11.4/mathmodule.c.  The code has been modified for use by
//  the funcion loggamma1p().
//
////////////////////////////////////////////////////////////////////////////

/* Implementation of the real gamma function.  In extensive but non-exhaustive
   random tests, this function proved accurate to within <= 10 ulps across the
   entire float domain.  Note that accuracy may depend on the quality of the
   system math functions, the pow function in particular.  Special cases
   follow C99 annex F.  The parameters and method are tailored to platforms
   whose double format is the IEEE 754 binary64 format.

   Method: for x > 0.0 we use the Lanczos approximation with parameters N=13
   and g=6.024680040776729583740234375; these parameters are amongst those
   used by the Boost library.  Following Boost (again), we re-express the
   Lanczos sum as a rational function, and compute it that way.  The
   coefficients below were computed independently using MPFR, and have been
   double-checked against the coefficients in the Boost source code.

   For x < 0.0 we use the reflection formula.

   There's one minor tweak that deserves explanation: Lanczos' formula for
   Gamma(x) involves computing pow(x+g-0.5, x-0.5) / exp(x+g-0.5).  For many x
   values, x+g-0.5 can be represented exactly.  However, in cases where it
   can't be represented exactly the small error in x+g-0.5 can be magnified
   significantly by the pow and exp calls, especially for large x.  A cheap
   correction is to multiply by (1 + e*g/(x+g-0.5)), where e is the error
   involved in the computation of x+g-0.5 (that is, e = computed value of
   x+g-0.5 - exact value of x+g-0.5).  Here's the proof:

   Correction factor
   -----------------
   Write x+g-0.5 = y-e, where y is exactly representable as an IEEE 754
   double, and e is tiny.  Then:

     pow(x+g-0.5,x-0.5)/exp(x+g-0.5) = pow(y-e, x-0.5)/exp(y-e)
     = pow(y, x-0.5)/exp(y) * C,

   where the correction_factor C is given by

     C = pow(1-e/y, x-0.5) * exp(e)

   Since e is tiny, pow(1-e/y, x-0.5) ~ 1-(x-0.5)*e/y, and exp(x) ~ 1+e, so:

     C ~ (1-(x-0.5)*e/y) * (1+e) ~ 1 + e*(y-(x-0.5))/y

   But y-(x-0.5) = g+e, and g+e ~ g.  So we get C ~ 1 + e*g/y, and

     pow(x+g-0.5,x-0.5)/exp(x+g-0.5) ~ pow(y, x-0.5)/exp(y) * (1 + e*g/y),

   Note that for accuracy, when computing r*C it's better to do

     r + e*g/y*r;

   than

     r * (1 + e*g/y);

   since the addition in the latter throws away most of the bits of
   information in e*g/y.
*/

#define LANCZOS_N 13
static const double lanczos_g = 6.024680040776729583740234375;
// static const double lanczos_g_minus_half = 5.524680040776729583740234375;
static const double lanczos_num_coeffs[LANCZOS_N] = {
    23531376880.410759688572007674451636754734846804940,
    42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843,
    17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708,
    1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801,
    31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768,
    186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024,
    210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408
};

/* denominator is x*(x+1)*...*(x+LANCZOS_N-2) */
static const double lanczos_den_coeffs[LANCZOS_N] = {
    0.0, 39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0,
    13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0};

// /* gamma values for small positive integers, 1 though NGAMMA_INTEGRAL */
// #define NGAMMA_INTEGRAL 23
// static const double gamma_integral[NGAMMA_INTEGRAL] = {
//     1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
//     3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
//     1307674368000.0, 20922789888000.0, 355687428096000.0,
//     6402373705728000.0, 121645100408832000.0, 2432902008176640000.0,
//     51090942171709440000.0, 1124000727777607680000.0,
// };

/* Lanczos' sum L_g(x), for positive x */

static double
lanczos_sum(double x)
{
    double num = 0.0, den = 0.0;
    int i;

    /* evaluate the rational function lanczos_sum(x).  For large
       x, the obvious algorithm risks overflow, so we instead
       rescale the denominator and numerator of the rational
       function by x**(1-LANCZOS_N) and treat this as a
       rational function in 1/x.  This also reduces the error for
       larger x values.  The choice of cutoff point (5.0 below) is
       somewhat arbitrary; in tests, smaller cutoff values than
       this resulted in lower accuracy. */
    if (x < 5.0) {
        for (i = LANCZOS_N; --i >= 0; ) {
            num = num * x + lanczos_num_coeffs[i];
            den = den * x + lanczos_den_coeffs[i];
        }
    }
    else {
        for (i = 0; i < LANCZOS_N; i++) {
            num = num / x + lanczos_num_coeffs[i];
            den = den / x + lanczos_den_coeffs[i];
        }
    }
    return num/den;
}

/*
   lgamma:  natural log of the absolute value of the Gamma function.
   For large arguments, Lanczos' formula works extremely well here.

   This is an edited version of m_lgamma() from the Python 3.11.4
   version of Python-3.11.4/mathmodule.c.  Python-specific code has
   been modified or removed, code related to handling a negative
   input has been removed, and since this function is only used in
   loggamma1p(), some error checking has been removed.
*/

static double
m_lgamma(double x)
{
    double r;
    double absx;

    if (x == 0.0 || x == 1.0) {
        // log(gamma(1)) = log(gamma(2)) = 0.0
        return 0.0;
    }

    absx = fabs(x);
    // tiny arguments: lgamma(x) ~ -log(fabs(x)) for small x
    if (absx < 1e-20)
        return -log(absx);

    // Lanczos' formula.  We could save a fraction of a ulp in accuracy by
    // having a second set of numerator coefficients for lanczos_sum that
    // absorbed the exp(-lanczos_g) term, and throwing out the lanczos_g
    // subtraction below; it's probably not worth it.
    r = log(lanczos_sum(absx)) - lanczos_g;
    r += (absx - 0.5) * (log(absx + lanczos_g - 0.5) - 1);
    //if (Py_IS_INFINITY(r))
    //    errno = ERANGE;
    return r;
}

////////////////////////////////////////////////////////////////////////////
//
//  End of the code derived from the Python 3.11.4 mathmodule.c
//
////////////////////////////////////////////////////////////////////////////

//
// Coefficients of the (12, 10) Padé approximation to log(gamma(1+x)) at x=0.
//
// These coefficients can be derived with mpmath in Python:
//
//     from mpmath import mp
//
//     def loggamma1p(x):
//         return mp.log(mp.gamma(mp.one + x))
//
//     mp.dps = 100
//     ts = mp.taylor(loggamma1p, 0, 24)
//     p_coeff, q_coeff = mp.pade(ts, 12, 10)
//
#define PADE_NUMER_N 13
static const double
loggamma1p_small_x_p_coeff[PADE_NUMER_N] = {
    0.0, -0.5772156649015329, -2.198602239247181,
    -2.8835804898328345, -0.7093852391116942,
    2.054674619926225, 2.5151727627777807,
    1.3458863118876616, 0.38837050891168406,
    0.06011155167110235, 0.004451819276845639,
    0.00011582239270882403, 2.362492383650223e-07
};

#define PADE_DENOM_N 11
static const double
loggamma1p_small_x_q_coeff[PADE_DENOM_N] = {
    1.0, 5.23386570457449, 11.759169522860718,
    14.820042213972009, 11.488581652651515,
    5.650511133519242, 1.754785949617669,
    0.33151383879069596, 0.0351480730651527,
    0.0017788484304635968, 2.9167070790354156e-05
};

static double loggamma1p_small_x(double x)
{
    double num = 0.0, den = 0.0;

    for (int i = PADE_NUMER_N - 1; i >= 0; --i) {
        num = num*x + loggamma1p_small_x_p_coeff[i];
    }
    for (int i = PADE_DENOM_N - 1; i >= 0; --i) {
        den = den*x + loggamma1p_small_x_q_coeff[i];
    }
    return num/den;
}

double
loggamma1p(double x)
{
    if (isnan(x) || (x <= -1.0)) {
        return NAN;
    }
    if (isinf(x)) {
        return INFINITY;
    }
    if (x == 0.0 || x == 1.0) {
        return 0.0;
    }
    if ((x > -0.25) && (x < 0.9)) {
        return loggamma1p_small_x(x);
    }
    return m_lgamma(1.0 + x);
}
