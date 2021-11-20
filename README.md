ufunclab
========

Some NumPy `ufuncs`, and some related tools.

Requires at least Python 3.6.

The unit tests require pytest.

Links to reference material related to NumPy's C API for ufuncs
and gufuncs are given [below](#resources).

What's in ufunclab?
-------------------

*Element-wise ufuncs*

| Function                                | Description                                      |
| --------                                | -----------                                      |
| [`logfactorial`](#logfactorial)         | Log of the factorial of integers                 |
| [`issnan`](#issnan)                     | Like `isnan`, but for signaling nans only.       |
| [`deadzone`](#deadzone)                 | Deadzone function                                |
| [`expint1`](#expint1)                   | Exponential integral E₁ for real inputs          |
| [`logexpint1`](#logexpint1)             | Logarithm of the exponential integral E₁         |

*Generalized ufuncs*

| Function                                | Description                                           |
| --------                                | -----------                                           |
| [`first`](#first)                       | First value that matches a target comparison          |
| [`argfirst`](#argfirst)                 | Index of the first occurrence of a target comparison  |
| [`argmin`](#argmin)                     | Like `numpy.argmin`, but a gufunc                     |
| [`argmax`](#argmax)                     | Like `numpy.argmax`, but a gufunc                     |
| [`minmax`](#minmax)                     | Minimum and maximum                                   |
| [`argminmax`](#argminmax)               | Indices of the min and the max                        |
| [`min_argmin`](#min_argmin)             | Minimum value and its index                           |
| [`max_argmax`](#max_argmax)             | Maximum value and its index                           |
| [`searchsortedl`](#searchsortedl)       | Find position for given element in sorted seq.        |
| [`searchsortedr`](#searchsortedr)       | Find position for given element in sorted seq.        |
| [`peaktopeak`](#peaktopeak)             | Alternative to `numpy.ptp`                            |
| [`all_same`](#all_same)                 | Check all values are the same                         |
| [`gmean`](#gmean)                       | Geometric mean                                        |
| [`hmean`](#hmean)                       | Harmonic mean                                         |
| [`meanvar`](#meanvar)                   | Mean and variance                                     |
| [`mad`](#mad)                           | Mean absolute difference (MAD)                        |
| [`mad1`](#mad1)                         | Unbiased estimator of the MAD                         |
| [`rmad`](#rmad)                         | Relative mean absolute difference (RMAD)              |
| [`rmad1`](#rmad1)                       | RMAD based on unbiased MAD                            |
| [`vnorm`](#vnorm)                       | Vector norm                                           |
| [`cross2`](#cross2)                     | 2-d vector cross product (returns scalar)             |
| [`cross3`](#cross3)                     | 3-d vector cross product                              |
| [`backlash`](#backlash)                 | Backlash operator                                     |
| [`hysteresis_relay`](#hysteresis_relay) | Relay with hysteresis (Schmitt trigger)               |

*Other tools*

| Function                                | Description                                           |
| --------                                | -----------                                           |
| [`gendot`](#gendot)                     | Create a new gufunc that composes two ufuncs          |
| [`ufunc_inspector`](#ufunc_inspector)   | Display ufunc information                             |

-----

### `logfactorial`

`logfactorial` is a ufunc that computes the natural logarithm of the
factorial of the integer x.

For example,
```
In [47]: from ufunclab import logfactorial

In [48]: logfactorial([1, 10, 100, 1000])
Out[48]: array([   0.        ,   15.10441257,  363.73937556, 5912.12817849])
```

### `issnan`

`issnan` is an element-wise ufunc with a single input that acts like
the standard `isnan` function, but it returns True only for
[*signaling* nans](https://en.wikipedia.org/wiki/NaN#Signaling_NaN).

The current implementation only handles the floating point types `np.float16`,
`np.float32` and `np.float64`.

```
>>> import numpy as np
>>> from ufunclab import issnan
>>> x = np.array([12.5, 0.0, np.inf, 999.0, np.nan], dtype=np.float32)
```
Put a signaling nan in `x[1]`. (The nan in `x[4]` is a quiet nan, and
we'll leave it that way.)
```
>>> v = x.view(np.uint32)
>>> v[1] = 0b0111_1111_1000_0000_0000_0000_0000_0011
>>> x
array([ 12.5,   nan,   inf, 999. ,   nan], dtype=float32)
>>> np.isnan(x)
array([False,  True, False, False,  True])
```
Note that NumPy displays both quiet and signaling nans as just `nan`,
and `np.isnan(x)` returns True for both quiet and signaling nans (as
it should).

`issnan(x)` indicates which values are signaling nans:
```
>>> issnan(x)
array([False,  True, False, False, False])
```

### `deadzone`

`deadzone(x, low, high)` is ufunc with three inputs and one output.
It computes the "deadzone" response of a signal:

           { 0         if low <= x <= high
    f(x) = { x - low   if x < low
           { x - high  if x > high

The function is similar to the
[deadzone block](https://www.mathworks.com/help/simulink/slref/deadzone.html)
of Matlab's Simulink library.  The function is also known as
a *soft threshold*.

Here's a plot of `deadzone(x, -0.25, 0.5)`:

![Deadzone plot1](https://github.com/WarrenWeckesser/ufunclab/blob/main/examples/deadzone_demo1.png)

The script `deadzone_demo2.py` in the `examples` directory generates
the plot

![Deadzone plot2](https://github.com/WarrenWeckesser/ufunclab/blob/main/examples/deadzone_demo2.png)


### `expint1`

`expint1(x)` computes the exponential integral E₁ for the real input x.


### `logexpint1`

`logexpint1(x)` computes the logaritham of the exponential integral E₁ for the real input x.

`expint1(x)` underflows to 0 for sufficiently large x:

```
>>> from ufunclab import expint1, logexpint1
>>> expint1([650, 700, 750, 800])
array([7.85247922e-286, 1.40651877e-307, 0.00000000e+000, 0.00000000e+000])
```

`logexpint1` avoids the underflow by computing the logarithm of the value:

```
>>> logexpint1([650, 700, 750, 800])
array([-656.47850729, -706.55250586, -756.62140388, -806.68585939])
```

### `first`

`first` is a gufunc with signature `(i),(),(),()->()` that returns the first
value that matches a given comparison.  The function signature is
`first(x, op, target, otherwise)`, where `op` is one of the values in
`ufunclab.op` that specifies the comparison to be made. `otherwise` is the
value to be returned if no value in `x` satisfies the given relation with
`target`.

Find the first nonzero value in `a`:

```
>>> import numpy as np
>>> from ufunclab import first, op

>>> a = np.array([0, 0, 0, 0, 0, -0.5, 0, 1, 0.1])
>>> first(a, op.NE, 0.0, 0.0)
-0.5
```

Find the first value in each row of `b` that is less than 0.
If there is no such value, return 0:

```
>>> b = np.array([[10, 23, -10, 0, -9],
...               [18, 28, 42, 33, 71],
...               [17, 29, 16, 14, -7]], dtype=np.int8)
...
>>> first(b, op.LT, 0, 0)
array([-10,   0,  -7], dtype=int8)
```

### `argfirst`

`argfirst` is a gufunc (signature `(i),(),()->()`) that finds the index of
the first true value of a comparison of an array with a target value.  If no
value is found, -1 is return.  Some examples follow.

```
>>> import numpy as np
>>> from ufunclab import argfirst, op
```

Find the index of the first occurrence of 0 in `x`:

```
>>> x = np.array([10, 35, 19, 0, -1, 24, 0])
>>> argfirst(x, op.EQ, 0)
3
```

Find the index of the first nonzero value in `a`:

```
>>> a = np.array([0, 0, 0, 0, 0, -0.5, 0, 1, 0.1])
>>> argfirst(a, op.NE, 0.0)
5
```

`argfirst` is a gufunc, so it can handle higher-dimensional
array arguments, and among its gufunc-related parameters is
`axis`.  By default, the gufunc operates along the last axis.
For example, here we find the location of the first nonzero
element in each row of `b`:

```
>>> b = np.array([[0, 8, 0, 0], [0, 0, 0, 0], [0, 0, 9, 2]],
...              dtype=np.uint8)
>>> b
array([[0, 8, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 9, 2]])
>>> argfirst(b, op.NE, np.uint8(0))
array([ 1, -1,  2])
```

If we give the argument `axis=0`, we tell `argfirst` to
operate along the first axis, which in this case is the
columns:

```
>>> argfirst(b, op.NE, np.uint8(0), axis=0)
array([-1,  0,  2,  2])
```

### `argmin`

`argmin` is a `gufunc` with signature `(i)->()` that is similar to `numpy.argmin`.

```
>>> from ufunclab import argmin
>>> x = np.array([[11, 10, 10, 23, 31],
...               [19, 20, 21, 22, 22],
...               [16, 15, 16, 14, 14]])
>>> argmin(x, axis=1)  # same as argmin(x)
array([1, 0, 3])
>>> argmin(x, axis=0)
array([0, 0, 0, 2, 2])
```

### `argmax`

`argmax` is a `gufunc` with signature `(i)->()` that is similar to `numpy.argmax`.

```
>>> from ufunclab import argmax
>>> x = np.array([[11, 10, 10, 23, 31],
...               [19, 20, 21, 22, 22],
...               [16, 15, 16, 14, 14]])
>>> argmax(x, axis=1)  # same as argmax(x)
array([4, 3, 0])
>>> argmax(x, axis=0)
array([1, 1, 1, 0, 0])
```

### `minmax`

`minmax` is a `gufunc` (signature `(i)->(2)`) that simultaneously computes
the minimum and maximum of a NumPy array.

The function handles the standard integer and floating point types, and
object arrays. The function will not accept complex arrays, nor arrays with
the data types `datetime64` or `timedelta64`.  Also, the function does not
implement any special handling of `nan`, so the behavior of this function
with arrays containing `nan` is *undefined*.

For an input with more than one dimension, `minmax` is applied to the
last axis.  For example, if `a` has shape (L, M, N), then `minmax(a)` has
shape (L, M, 2).

```
>>> x = np.array([5, -10, -25, 99, 100, 10], dtype=np.int8)
>>> minmax(x)
array([-25, 100], dtype=int8)

>>> np.random.seed(12345)
>>> y = np.random.randint(-1000, 1000, size=(3, 3, 5)).astype(np.float32)
>>> y
array([[[-518.,  509.,  309., -871.,  444.],
        [ 449., -618.,  381., -454.,  565.],
        [-231.,  142.,  393.,  339., -346.]],

       [[-895.,  115., -241.,  398.,  232.],
        [-118., -287., -733.,  101.,  674.],
        [-919.,  746., -834., -737., -957.]],

       [[-769., -977.,   53.,  -48.,  463.],
        [ 311., -299., -647.,  883., -145.],
        [-964., -424., -613., -236.,  148.]]], dtype=float32)

>>> mm = minmax(y)
>>> mm
array([[[-871.,  509.],
        [-618.,  565.],
        [-346.,  393.]],

       [[-895.,  398.],
        [-733.,  674.],
        [-957.,  746.]],

       [[-977.,  463.],
        [-647.,  883.],
        [-964.,  148.]]], dtype=float32)

>>> mm.shape
(3, 3, 2)

>>> z = np.array(['foo', 'xyz', 'bar', 'abc', 'def'], dtype=object)
>>> minmax(z)
array(['abc', 'xyz'], dtype=object)

>>> from fractions import Fraction
>>> f = np.array([Fraction(1, 3), Fraction(3, 5),
...               Fraction(22, 7), Fraction(5, 2)], dtype=object)
>>> minmax(f)
array([Fraction(1, 3), Fraction(22, 7)], dtype=object)

```

### `argminmax`

`argminmax` is a `gufunc` (signature `(i)->(2)`) that simultaneously
computes the `argmin` and `argmax` of a NumPy array.

```
>>> y = np.array([[-518,  509,  309, -871,  444,  449, -618,  381],
...               [-454,  565, -231,  142,  393,  339, -346, -895],
...               [ 115, -241,  398,  232, -118, -287, -733,  101]],
...              dtype=np.float32)
>>> argminmax(y)
array([[3, 1],
       [7, 1],
       [6, 2]])
>>> argminmax(y, axes=[0, 0])
array([[0, 2, 1, 0, 2, 2, 2, 1],
       [2, 1, 2, 2, 0, 0, 1, 0]])
```

### `min_argmin`

`min_argmin` is a gufunc (signature `(i)->(),()`) that returns both
the extreme value and the index of the extreme value.

```
>>> x = np.array([[ 1, 10, 18, 17, 11],
...               [15, 11,  0,  4,  8],
...               [10, 10, 12, 11, 11]])
>>> min_argmin(x, axis=1)
(array([ 1,  0, 10]), array([0, 2, 0]))
```

### `max_argmax`

`max_argmax` is a gufunc (signature `(i)->(),()`) that returns both
the extreme value and the index of the extreme value.

```
>>> x = np.array([[ 1, 10, 18, 17, 11],
...               [15, 11,  0,  4,  8],
...               [10, 10, 12, 11, 11]])
>>> max_argmax(x, axis=1)
(array([18, 15, 12]), array([2, 0, 2]))

>>> from fractions import Fraction as F
>>> y = np.array([F(2, 3), F(3, 4), F(2, 7), F(2, 5)])
>>> max_argmax(y)
(Fraction(3, 4), 1)
```

### `searchsortedl`

`searchsortedl` is a gufunc with signature `(i),()->()`.  The function
is equivalent to `numpy.searchsorted` with `side='left'`, but as a gufunc,
it supports broadcasting of its arguments.  (Note that `searchsortedl`
does not provide the `sorter` parameter.)

```
>>> import numpy as np
>>> from ufunclab import searchsortedl
>>> searchsortedl([1, 1, 2, 3, 5, 8, 13, 21], [1, 4, 15, 99])
array([0, 4, 7, 8])
>>> arr = np.array([[1, 1, 2, 3, 5, 8, 13, 21],
...                 [1, 1, 1, 1, 2, 2, 10, 10]])
>>> searchsortedl(arr, [7, 8])
array([5, 6])
>>> searchsortedl(arr, [[2], [5]])
array([[2, 4],
       [4, 6]])
```

### `searchsortedr`

`searchsortedr` is a gufunc with signature `(i),()->()`.  The function
is equivalent to `numpy.searchsorted` with `side='right'`, but as a gufunc,
it supports broadcasting of its arguments.  (Note that `searchsortedr`
does not provide the `sorter` parameter.)

```
>>> import numpy as np
>>> from ufunclab import searchsortedl
>>> searchsortedr([1, 1, 2, 3, 5, 8, 13, 21], [1, 4, 15, 99])
array([2, 4, 7, 8])
>>> arr = np.array([[1, 1, 2, 3, 5, 8, 13, 21],
...                 [1, 1, 1, 1, 2, 2, 10, 10]])
>>> searchsortedr(arr, [7, 8])
array([5, 6])
>>> searchsortedr(arr, [[2], [5]])
array([[3, 6],
       [5, 6]])
```


### `peaktopeak`

`peaktopeak` is a `gufunc` (signature `(i)->()`) that computes the
peak-to-peak range of a NumPy array.  It is like the `ptp` method
of a NumPy array, but when the input is signed integers, the output
is an unsigned integer with the same bit width.

The function handles the standard integer and floating point types,
`datetime64`, `timedelta64`, and object arrays. The function does not
accept complex arrays.  Also, the function does not implement any special
handling of `nan`, so the behavior of this function with arrays containing
`nan` is undefined (i.e. it might not do what you want, and the behavior
might change in the next update of the software).

```
>>> x = np.array([85, 125, 0, -75, -50], dtype=np.int8)
>>> p = peaktopeak(x)
>>> p
200
>>> type(p)
numpy.uint8
```

Compare that to the `ptp` method, which returns a value with the
same data type as the input:

```
>>> q = x.ptp()
>>> q
-56
>>> type(q)
numpy.int8

```

`f` is an object array of `Fraction`s and has shape (2, 4).

```
>>> from fractions import Fraction
>>> f = np.array([[Fraction(1, 3), Fraction(3, 5),
...                Fraction(22, 7), Fraction(5, 2)],
...               [Fraction(-2, 9), Fraction(1, 3),
...                Fraction(2, 3), Fraction(5, 9)]], dtype=object)
>>> peaktopeak(x)
array([Fraction(59, 21), Fraction(8, 9)], dtype=object)

```

`dates` is an array of `datetime64`.

```
>>> dates = np.array([np.datetime64('2015-11-02T12:34:50'),
...                   np.datetime64('2016-03-01T16:00:00'),
...                   np.datetime64('2015-07-02T21:20:19'),
...                   np.datetime64('2016-05-01T19:25:00')])

>>> dates
array(['2015-11-02T12:34:50', '2016-03-01T16:00:00',
       '2015-07-02T21:20:19', '2016-05-01T19:25:00'],
      dtype='datetime64[s]')
>>> timespan = peaktopeak(dates)
>>> timespan
numpy.timedelta64(26258681,'s')
>>> timespan / np.timedelta64(1, 'D')  # Convert to number of days.
303.9199189814815
```

Casting works when the `out` argument is an array with dtype `timedelta64`.
For example,

```
>>> out = np.empty((), dtype='timedelta64[D]')
>>> peaktopeak(dates, out=out)
array(303, dtype='timedelta64[D]')

```


### `all_same`

`all_same` is a gufunc (signature `(i)->()`) that tests that all the
values in the array along the given axis are the same.

(Note: handling of `datetime64`, `timedelta64` and complex data types
are not implemented yet.)

```
>>> x = np.array([[3, 2, 2, 3, 2, 2, 3, 1, 3],
...               [1, 2, 2, 2, 2, 2, 3, 1, 1],
...               [2, 3, 3, 1, 2, 3, 3, 1, 2]])

>>> all_same(x, axis=0)
array([False, False, False, False,  True, False,  True,  True, False])

>>> all_same(x, axis=1)
array([False, False, False])
```

Object arrays are handled.

```
>>> a = np.array([[None, "foo", 99], [None, "bar", "abc"]])
>>> a
array([[None, 'foo', 99],
       [None, 'bar', 'abc']], dtype=object)

>>> all_same(a, axis=0)
array([ True, False, False])
```

### `gmean`

`gmean` is a gufunc (signature `(i)->()`) that computes the
[geometric mean](https://en.wikipedia.org/wiki/Geometric_mean).

For example,

```
In [25]: import numpy as np

In [26]: from ufunclab import gmean

In [27]: x = np.array([1, 2, 3, 5, 8], dtype=np.uint8)

In [28]: gmean(x)
Out[28]: 2.992555739477689

In [29]: y = np.arange(1, 16).reshape(3, 5)

In [30]: y
Out[30]:
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10],
       [11, 12, 13, 14, 15]])

In [31]: gmean(y, axis=1)
Out[31]: array([ 2.60517108,  7.87256685, 12.92252305])
```

### `hmean`

`hmean` is a gufunc (signature `(i)->()`) that computes the
[harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean).

For example,

```
In [25]: import numpy as np

In [26]: from ufunclab import hmean

In [27]: x = np.array([1, 2, 3, 5, 8], dtype=np.uint8)

In [28]: hmean(x)
Out[28]: 2.316602316602317

In [29]: y = np.arange(1, 16).reshape(3, 5)

In [30]: y
Out[30]:
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10],
       [11, 12, 13, 14, 15]])

In [31]: hmean(y, axis=1)
Out[31]: array([ 2.18978102,  7.74431469, 12.84486077])
```

### `meanvar`

`meanvar` is a gufunc (signature `(i),()->(2)`) that computes both
the mean and variance in one function call.

For example,

```
In [1]: import numpy as np

In [2]: from ufunclab import meanvar

In [3]: meanvar([1, 2, 4, 5], 0)  # Use ddof=0.
Out[3]: array([3. , 2.5])
```

Appy `meanvar` with `ddof=1` to the rows of a 2-d array.
The output has shape `(4, 2)`; the first column holds the
means, and the second column holds the variances.


```
In [4]: x = np.array([[1, 4, 4, 2, 1, 1, 2, 7],
   ...:               [0, 0, 9, 4, 1, 0, 0, 1],
   ...:               [8, 3, 3, 3, 3, 3, 3, 3],
   ...:               [5, 5, 5, 5, 5, 5, 5, 5]])

In [5]: meanvar(x, 1)  # Use ddof=1.
Out[5]:
array([[ 2.75 ,  4.5  ],
       [ 1.875, 10.125],
       [ 3.625,  3.125],
       [ 5.   ,  0.   ]])
```

Compare to the results of `numpy.mean` and `numpy.var`:

```
In [6]: np.mean(x, axis=1)
Out[6]: array([2.75 , 1.875, 3.625, 5.   ])

In [7]: np.var(x, ddof=1, axis=1)
Out[7]: array([ 4.5  , 10.125,  3.125,  0.   ])
```

### `mad`

`mad` computes the [mean absolute difference](https://en.wikipedia.org/wiki/Mean_absolute_difference)
of a 1-d array (gufunc signature is `(i)->()`).  `mad` is the standard
calculation (sum of the absolute differences divided by `n**2`), and `mad1`
is the unbiased estimator (sum of the absolute differences divided by
`n*(n-1)`).

For example,
```
In [15]: import numpy as np

In [16]: from ufunclab import mad

In [17]: x = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

In [18]: mad(x)
Out[18]: 2.6666666666666665

In [19]: y = np.linspace(0, 1, 21).reshape(3, 7)**2

In [20]: y
Out[20]:
array([[0.    , 0.0025, 0.01  , 0.0225, 0.04  , 0.0625, 0.09  ],
       [0.1225, 0.16  , 0.2025, 0.25  , 0.3025, 0.36  , 0.4225],
       [0.49  , 0.5625, 0.64  , 0.7225, 0.81  , 0.9025, 1.    ]])

In [21]: mad(y, axis=1)
Out[21]: array([0.03428571, 0.11428571, 0.19428571])
```

### `mad1`

`mad1` computes the [mean absolute difference](https://en.wikipedia.org/wiki/Mean_absolute_difference)
of a 1-d array (gufunc signature is `(i)->()`).  This version is
based on the unbiasd estimator of the mean absolute difference.
`mad` is the standard calculation (sum of the absolute differences
divided by `n**2`), and `mad1` is the unbiased estimator (sum of the
absolute differences divided by `n*(n-1)`).

For example,
```
In [1]: import numpy as np

In [2]: from ufunclab import mad1

In [3]: x = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

In [4]: mad1(x)
Out[4]: 3.2

In [5]: y = np.linspace(0, 1, 21).reshape(3, 7)**2

In [6]: y
Out[6]:
array([[0.    , 0.0025, 0.01  , 0.0225, 0.04  , 0.0625, 0.09  ],
       [0.1225, 0.16  , 0.2025, 0.25  , 0.3025, 0.36  , 0.4225],
       [0.49  , 0.5625, 0.64  , 0.7225, 0.81  , 0.9025, 1.    ]])

In [7]: mad1(y, axis=1)
Out[7]: array([0.04      , 0.13333333, 0.22666667])
```

### `rmad`

`rmad` computes the relative mean absolute difference (gufunc
signature is `(i)->()`).

`rmad` is the standard calculation and `rmad1` uses the unbiased
estimator of the mean absolute difference to compute the relative
mean absolute difference.

`rmad` is twice the [Gini coefficient](https://en.wikipedia.org/wiki/Gini_coefficient).

For example,
```
In [1]: import numpy as np

In [2]: from ufunclab import rmad

In [3]: x = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

In [4]: rmad(x)
Out[4]: 0.7999999999999999

In [5]: y = np.linspace(0, 1, 21).reshape(3, 7)**2

In [6]: y
Out[6]:
array([[0.    , 0.0025, 0.01  , 0.0225, 0.04  , 0.0625, 0.09  ],
       [0.1225, 0.16  , 0.2025, 0.25  , 0.3025, 0.36  , 0.4225],
       [0.49  , 0.5625, 0.64  , 0.7225, 0.81  , 0.9025, 1.    ]])

In [7]: rmad(y, axis=1)
Out[7]: array([1.05494505, 0.43956044, 0.26523647])
```

### `rmad1`

`rmad1` computes the relative mean absolute difference (gufunc
signature is `(i)->()`).

`rmad1` uses the unbiased estimator of the mean absolute difference
to compute the relative mean absolute difference.

For example,
```
In [1]: import numpy as np

In [2]: from ufunclab import rmad1

In [3]: x = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

In [4]: rmad1(x)
Out[4]: 0.96

In [5]: y = np.linspace(0, 1, 21).reshape(3, 7)**2

In [6]: y
Out[6]:
array([[0.    , 0.0025, 0.01  , 0.0225, 0.04  , 0.0625, 0.09  ],
       [0.1225, 0.16  , 0.2025, 0.25  , 0.3025, 0.36  , 0.4225],
       [0.49  , 0.5625, 0.64  , 0.7225, 0.81  , 0.9025, 1.    ]])

In [7]: rmad1(y, axis=1)
Out[7]: array([1.23076923, 0.51282051, 0.30944255])
```

### `vnorm`

`vnorm` computes the vector norm of 1D arrays.  It is a gufunc with
signatue `(i), () -> ()`.

For example, the 2-norm of [3, 4] is
```
In [28]: vnorm([3, 4], 2)
Out[28]: 5.0
```

Compute the p-norm of [3, 4] for several values of p:

```
In [29]: vnorm([3, 4], [1, 2, 3, np.inf])
Out[29]: array([7.        , 5.        , 4.49794145, 4.        ])
```

Compute the 2-norm of four 2-d vectors:

```
In [30]: vnorm([[3, 4], [5, 12], [0, 1], [1, 1]], 2)
Out[30]: array([ 5.        , 13.        ,  1.        ,  1.41421356])
```

For the same vectors, compute the p-norm for p = [1, 2, inf]:

```
In [31]: vnorm([[3, 4], [5, 12], [0, 1], [1, 1]], [[1], [2], [np.inf]])
Out[31]:
array([[ 7.        , 17.        ,  1.        ,  2.        ],
       [ 5.        , 13.        ,  1.        ,  1.41421356],
       [ 4.        , 12.        ,  1.        ,  1.        ]])
```

`vnorm` handles complex numbers. Here we compute the norm of `z`
with orders 1, 2, 3, and inf.  (Note that `abs(z)` is [2, 5, 0, 14].)

```
>>> z = np.array([-2j, 3+4j, 0, 14])
>>> vnorm(z, [1, 2, 3, np.inf])
array([21.        , 15.        , 14.22263137, 14.        ])
```

### `cross2`

`cross2(u, v)` is a gufunc with signature `(2),(2)->()`.  It computes
the 2-d cross product that returns a scalar.  That is, `cross2([u0, u1], [v0, v1])`
is `u0*v1 - u1*v0`.  The calculation is the same as that of `numpy.cross`,
but `cross2` is restricted to 2-d inputs.

For example,
```
In [1]: import numpy as np

In [2]: from ufunclab import cross2

In [3]: cross2([1, 2], [5, 3])
Out[3]: -7

In [4]: cross2([[1, 2], [6, 0]], [[5, 3], [2, 3]])
Out[4]: array([-7, 18])

In [5]: cross2([1j, 3], [-1j, 2+3j])
Out[5]: (-3+5j)
```

In the following, `a` and `b` are object arrays; `a` has shape (2,),
and `b` has shape (3, 2).  The result of ``cross2(a, b)`` has shape
(3,).

```
In [6]: from fractions import Fraction as F

In [7]: a = np.array([F(1, 3), F(2, 7)])

In [8]: b = np.array([[F(7, 4), F(6, 7)], [F(2, 5), F(-3, 7)], [1, F(1, 4)]])

In [9]: cross2(a, b)
Out[9]:
array([Fraction(-3, 14), Fraction(-9, 35), Fraction(-17, 84)],
      dtype=object)
```

### `cross3`

`cross3(u, v)` is a gufunc with signature `(3),(3)->(3)`.  It computes
the 3-d vector cross product (like `numpy.cross`, but specialized to the
case of 3-d vectors only).

For example,
```
In [1]: import numpy as np

In [2]: from ufunclab import cross3

In [3]: u = np.array([1, 2, 3])

In [4]: v = np.array([2, 2, -1])

In [5]: cross3(u, v)
Out[5]: array([-8,  7, -2])

In [6]: x = np.arange(15).reshape(5, 3)

In [7]: y = np.round(10*np.sin(np.linspace(0, 2, 6))).reshape(2, 1, 3)

In [8]: x
Out[8]:
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11],
       [12, 13, 14]])

In [9]: y
Out[9]:
array([[[ 0.,  4.,  7.]],

       [[ 9., 10.,  9.]]])

In [10]: cross3(x, y)
Out[10]:
array([[[ -1.,   0.,   0.],
        [  8., -21.,  12.],
        [ 17., -42.,  24.],
        [ 26., -63.,  36.],
        [ 35., -84.,  48.]],

       [[-11.,  18.,  -9.],
        [-14.,  18.,  -6.],
        [-17.,  18.,  -3.],
        [-20.,  18.,   0.],
        [-23.,  18.,   3.]]])
```

### `backlash`

`backlash(x, deadband, initial)`, a gufunc with signature `(i),(),()->(i)`,
computes the "backlash" response of a signal; see the Wikipedia article
[Backlash (engineering)](https://en.wikipedia.org/wiki/Backlash_(engineering)).
The function emulates the
[Backlash block](https://www.mathworks.com/help/simulink/slref/backlash.html)
of Matlab's Simulink library.

For example,

```
  In [52]: x = np.array([0, 0.5, 1, 1.1, 1.0, 1.5, 1.4, 1.2, 0.5])

  In [53]: deadband = 0.4

  In [54]: initial = 0

  In [55]: backlash(x, deadband, initial)
  Out[55]: array([0. , 0.3, 0.8, 0.9, 0.9, 1.3, 1.3, 1.3, 0.7])
```

The script `backlash_demo.py` in the `examples` directory generates
the plot

![Backlash plot](https://github.com/WarrenWeckesser/ufunclab/blob/main/examples/backlash_demo.png)


### `hysteresis_relay`

`hysteresis_relay(x, low_threshold, high_threshold, low_value, high_value, init)`
a gufunc with signature `(i),(),(),(),(),()->(i)`, passes `x` through a relay
with hysteresis (like a Schmitt trigger). The function is similar to the
[relay block](https://www.mathworks.com/help/simulink/slref/relay.html)
of Matlab's Simulink library.

The script `hysteresis_relay_demo.py` in the `examples` directory generates
the plot

![hysteresis_replay plot](https://github.com/WarrenWeckesser/ufunclab/blob/main/examples/hysteresis_relay_demo.png)

### `gendot`

`gendot` creates a new gufunc (with signature `(i),(i)->()`) that is
the composition of two ufuncs.  The first ufunc must be an element-wise
ufunc with two inputs and one output.  The second must be either another
element-wise ufunc with two inputs and one output, or a gufunc with
signature `(i)->()`.

The name `gendot` is from "generalized dot product".  The standard
dot product is the composition of element-wise multiplication and
reduction with addition.  The `prodfunc` and `sumfunc` arguments of
`gendot` take the place of multiplication and addition.

For example, to take the element-wise minimum of two 1-d arrays,
and then take the maximum of the result:

```
In [1]: import numpy as np

In [2]: from ufunclab import gendot

In [3]: minmaxdot = gendot(np.minimum, np.maximum)

In [4]: a = np.array([1.0, 2.5, 0.3, 1.9, 3.0, 1.8])

In [5]: b = np.array([0.5, 1.1, 0.9, 2.1, 0.3, 3.0])

In [6]: minmaxdot(a, b)
Out[6]: 1.9
```

`minmaxdot` is a gufunc with signature `(i),(i)->()`;  the type
signatures of the gufunc loop functions were derived by matching
the signatures of the ufunc loop functions for `np.minimum` and
`np.maximum`:

```
In [8]: minmaxdot.signature
Out[8]: '(i),(i)->()'

In [9]: print(minmaxdot.types)
['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
 'LL->L', 'qq->q', 'QQ->Q', 'ee->e', 'ff->f', 'dd->d', 'gg->g', 'FF->F',
 'DD->D', 'GG->G', 'mm->m', 'MM->M']
```

`gendot` is experimental, and might not be useful in many applications.
We could do the same calculation as `minmaxdot` with, for example,
`np.maximum.reduce(np.minimum(a, b))`, and in fact, the pure NumPy
version is faster than `minmaxdot(a, b)` for large (and even moderately
sized) 1-d arrays.  An advantage of the `gendot` gufunc is that it does
not create an intermediate array when broadcasting takes place.  For
example, with inputs `x` and `y` with shapes `(20, 10000000)` and
`(10, 1, 10000000)`, the equivalent of `minmaxdot(x, y)` can be computed
with `np.maximum.reduce(np.minimum(x, y), axis=-1)`, but `np.minimum(x, y)`
creates an array with shape `(10, 20, 10000000)`.  Computing the result
with `minmaxdot(x, y)` does not create the temporary intermediate array.

### `ufunc_inspector`

`ufunc_inspector(func)` prints information about a NumPy ufunc.

For example,

```
In [10]: import numpy as np

In [11]: from ufunclab import ufunc_inspector

In [12]: ufunc_inspector(np.hypot)
'hypot' is a ufunc.
nin = 2, nout = 1, ntypes = 5
loop types:
  0: ( 23,  23) ->  23  (ee->e)  PyUFunc_ee_e_As_ff_f
  1: ( 11,  11) ->  11  (ff->f)  PyUFunc_ff_f
  2: ( 12,  12) ->  12  (dd->d)  PyUFunc_dd_d
  3: ( 13,  13) ->  13  (gg->g)  PyUFunc_gg_g
  4: ( 17,  17) ->  17  (OO->O)  PyUFunc_OO_O_method
```
(The output will likely change as the code develops.)

```
In [16]: ufunc_inspector(np.sqrt)
'sqrt' is a ufunc.
nin = 1, nout = 1, ntypes = 10
loop types:
  0:   23 ->  23  (e->e)  PyUFunc_e_e_As_f_f
  1:   11 ->  11  (f->f)  not generic (or not in the checked generics)
  2:   12 ->  12  (d->d)  not generic (or not in the checked generics)
  3:   11 ->  11  (f->f)  PyUFunc_f_f
  4:   12 ->  12  (d->d)  PyUFunc_d_d
  5:   13 ->  13  (g->g)  PyUFunc_g_g
  6:   14 ->  14  (F->F)  PyUFunc_F_F
  7:   15 ->  15  (D->D)  PyUFunc_D_D
  8:   16 ->  16  (G->G)  PyUFunc_G_G
  9:   17 ->  17  (O->O)  PyUFunc_O_O_method

In [17]: ufunc_inspector(np.add)
'add' is a ufunc.
nin = 2, nout = 1, ntypes = 22
loop types:
  0: (  0,   0) ->   0  (??->?)  not generic (or not in the checked generics)
  1: (  1,   1) ->   1  (bb->b)  not generic (or not in the checked generics)
  2: (  2,   2) ->   2  (BB->B)  not generic (or not in the checked generics)
  3: (  3,   3) ->   3  (hh->h)  not generic (or not in the checked generics)
  4: (  4,   4) ->   4  (HH->H)  not generic (or not in the checked generics)
  5: (  5,   5) ->   5  (ii->i)  not generic (or not in the checked generics)
  6: (  6,   6) ->   6  (II->I)  not generic (or not in the checked generics)
  7: (  7,   7) ->   7  (ll->l)  not generic (or not in the checked generics)
  8: (  8,   8) ->   8  (LL->L)  not generic (or not in the checked generics)
  9: (  9,   9) ->   9  (qq->q)  not generic (or not in the checked generics)
  10: ( 10,  10) ->  10  (QQ->Q)  not generic (or not in the checked generics)
  11: ( 23,  23) ->  23  (ee->e)  not generic (or not in the checked generics)
  12: ( 11,  11) ->  11  (ff->f)  not generic (or not in the checked generics)
  13: ( 12,  12) ->  12  (dd->d)  not generic (or not in the checked generics)
  14: ( 13,  13) ->  13  (gg->g)  not generic (or not in the checked generics)
  15: ( 14,  14) ->  14  (FF->F)  not generic (or not in the checked generics)
  16: ( 15,  15) ->  15  (DD->D)  not generic (or not in the checked generics)
  17: ( 16,  16) ->  16  (GG->G)  not generic (or not in the checked generics)
  18: ( 21,  22) ->  22  (Mm->M)  not generic (or not in the checked generics)
  19: ( 22,  22) ->  22  (mm->m)  not generic (or not in the checked generics)
  20: ( 22,  21) ->  21  (mM->M)  not generic (or not in the checked generics)
  21: ( 17,  17) ->  17  (OO->O)  PyUFunc_OO_O
```


### Resources

Here's a collection of resources for learning about the C API for ufuncs.

* [Universal functions (ufunc)](https://numpy.org/devdocs/reference/ufuncs.html)
* [UFunc API](https://numpy.org/devdocs/reference/c-api/ufunc.html)
* [Generalized Universal Function API](https://numpy.org/devdocs/reference/c-api/generalized-ufuncs.html)
* [NEP 5 — Generalized Universal Functions](https://numpy.org/neps/nep-0005-generalized-ufuncs.html)
* [NEP 20 — Expansion of Generalized Universal Function Signatures](https://numpy.org/neps/nep-0020-gufunc-signature-enhancement.html)
* [Universal functions](https://numpy.org/devdocs/reference/internals.code-explanations.html#universal-functions),
  part of the [NumPy C Code Explanations](https://numpy.org/devdocs/reference/internals.code-explanations.html)
  * In particular, the section
    ["Function call"](https://numpy.org/devdocs/reference/internals.code-explanations.html#function-call)
    explains when the GIL is released.
* When implementing inner loops for many NumPy dtypes, the
  [NumPy distutils](https://numpy.org/doc/stable/reference/distutils_guide.html)
  [template preprocessor](https://numpy.org/doc/stable/reference/distutils_guide.html#conversion-of-src-files-using-templates)
  is a useful tool. (See the ["Other files"](https://numpy.org/doc/stable/reference/distutils_guide.html#other-files)
  section for the syntax that would be used in, say, a C file.)
* Some relevant NumPy source code, if you want to dive deep:
  * `PyUFuncObject` along with related C types and macros are defined in
   [`numpy/numpy/core/include/numpy/ufuncobject.h`](https://github.com/numpy/numpy/blob/7214ca4688545b432c45287195e2f46c5e418ce8/numpy/core/include/numpy/ufuncobject.h).
  * `PyUFunc_FromFuncAndData` and `PyUFunc_FromFuncAndDataAndSignatureAndIdentity`
    are defined in the file [`numpy/numpy/core/src/umath/ufunc_object.c`](https://github.com/numpy/numpy/blob/7214ca4688545b432c45287195e2f46c5e418ce8/numpy/core/src/umath/ufunc_object.c).
* Section of the [SciPy Lecture Notes](https://scipy-lectures.org/index.html) on ufuncs:
  * [2.2.2 Universal Functions](https://scipy-lectures.org/advanced/advanced_numpy/index.html#universal-functions)
* [Data Type API](https://numpy.org/doc/stable/reference/c-api/dtype.html) --
  a handy reference.
