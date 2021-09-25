ufunclab
========

Some NumPy `ufuncs`, and some related tools.

Requires at least Python 3.5.

The unit tests require pytest.

Links to reference material related to NumPy's C API for ufuncs
and gufuncs are given [below](#resources).

What's in ufunclab?
-------------------

`ufunclab` defines these functions:

| Function                                | Description                      |
| --------                                | -----------                      |
| [`logfactorial`](#logfactorial)         | Log of the factorial of integers |
| [`peaktopeak`](#peaktopeak)             | Alternative to `numpy.ptp`       |
| [`minmax`](#minmax)                     | Minimum and maximum              |
| [`argminmax`](#argminmax)               | Indices of the min and the max   |
| [`min_argmin`](#min_argmin)             | Minimum value and its index      |
| [`max_argmax`](#max_argmax)             | Maximum value and its index      |
| [`all_same`](#all_same)                 | Check all values are the same    |
| [`gmean`](#gmean)                       | Geometric mean                   |
| [`hmean`](#hmean)                       | Harmonic mean                    |
| [`mad`](#mad)                           | Mean absolute difference (MAD)   |
| [`mad1`](#mad1)                         | Unbiased estimator of the MAD    |
| [`rmad`](#rmad)                         | Relative mean absolute difference|
| [`rmad1`](#rmad1)                       | RMAD based on unbiased MAD       |
| [`vnorm`](#vnorm)                       | Vector norm                      |
| [`cross3`](#cross3)                     | 3-d vector cross product         |
| [`backlash`](#backlash)                 | Backlash operator                |
| [`deadzone`](#deadzone)                 | Deadzone operator                |
| [`hysteresis_relay`](#hysteresis_relay) | Relay with hysteresis            |
| [`ufunc_inspector`](#ufunc_inspector)   | Display ufunc information        |

Details follow.


### `logfactorial`

* Computes the natural logarithm of the factorial of the integer x.

* This is a fairly standard implementation of a NumPy ufunc.

### `peaktopeak`

* A `gufunc` that computes the peak-to-peak range of a NumPy array.
  It is like the `ptp` method of a NumPy array, but when the input
  is signed integers, the output is an unsigned integer with the
  same bit width.

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
  ...:                  np.datetime64('2016-03-01T16:00:00'),
  ...:                  np.datetime64('2015-07-02T21:20:19'),
  ...:                  np.datetime64('2016-05-01T19:25:00')])

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

### `minmax`

* A `gufunc` that simultaneously computes the minimum and maximum of a NumPy
  array.  (The ufunc signature is '(i)->(2)'.)

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

* A `gufunc` that simultaneously computes the `argmin` and `argmax` of a NumPy
  array.  (The ufunc signature is '(i)->(2)'.)

  ```
  >>> np.random.seed(12345)
  >>> y = np.random.randint(-1000, 1000, size=(3, 8)).astype(np.float32)
  >>> y
  array([[-518.,  509.,  309., -871.,  444.,  449., -618.,  381.],
         [-454.,  565., -231.,  142.,  393.,  339., -346., -895.],
         [ 115., -241.,  398.,  232., -118., -287., -733.,  101.]],
        dtype=float32)
  >>> argminmax(y)
  array([[3, 1],
         [7, 1],
         [6, 2]])
  ```

### `min_argmin`

* Returns both the extreme value and the index of the extreme
  value.  (The ufunc signature  is '(i)->(),()'.)

  ```
  >>> np.random.seed(123456)
  >>> x = np.random.randint(0, 20, size=(3, 5))
  >>> x
  array([[ 1, 10, 18, 17, 11],
         [15, 11,  0,  4,  8],
         [10, 10, 12, 11, 11]])
  >>> min_argmin(x, axis=1)
  (array([ 1,  0, 10]), array([0, 2, 0]))
  ```

### `max_argmax`

* Returns both the extreme value and the index of the extreme
  value.  (The ufunc signature is '(i)->(),()'.)

  ```
  >>> from fractions import Fraction as F
  >>> y = np.array([F(2, 3), F(3, 4), F(2, 7), F(2, 5)])
  >>> max_argmax(y)
  (Fraction(3, 4), 1)
  ```

### `all_same`

* Test that all the values in the array along the given axis are the same.
  (Note: handling of `datetime64`, `timedelta64` and complex data types
  are not implemented yet.  Also, `axis=None` is not accepted.)

  ```
  >>> rng = np.random.Generator(np.random.PCG64(8675309))
  >>> x = rng.integers(1, 4, size=(3, 9))
  >>> x
  array([[3, 2, 2, 3, 2, 2, 2, 3, 3],
         [1, 2, 2, 2, 2, 2, 2, 3, 1],
         [2, 3, 3, 1, 2, 3, 2, 1, 2]])

  >>> all_same(x, axis=0)
  array([False, False, False, False,  True, False,  True, False, False])

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

* Compute the geometric mean.

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

* Compute the harmonic mean.

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

### `mad`

* `mad` computes the mean absolute difference of a 1-d array
  (gufunc signature is `(i)->()`).  `mad` is the standard calculation
  (sum of the absolute differences divided by `n**2`), and `mad1` is
  the unbiased estimator (sum of the absolute differences divided by
  `n*(n-1)`).

### `mad1`

* `mad1` computes the mean absolute difference of a 1-d array
  (gufunc signature is `(i)->()`).  This version is based on the unbiasd
  estimator of the mean absolute difference. `mad` is the standard
  calculation (sum of the absolute differences divided by `n**2`), and
  `mad1` is the unbiased estimator (sum of the absolute differences
  divided by `n*(n-1)`).

### `rmad`

* `rmad` computes the relative mean absolute difference.
  `rmad` is the standard calculation and `rmad1` uses the unbiased
  estimator of the mean absolute difference to compute the relative
  mean absolute difference.

  `rmad` is twice the Gini coefficient.

### `rmad1`

* `rmad1` computes the relative mean absolute difference. `rmad1` uses
  the unbiased estimator of the mean absolute difference to compute the
  relative mean absolute difference.

### `vnorm`

* `vnorm` computes the vector norm of 1D arrays.  It is a gufunc with
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

### `cross3`

* `cross3(u, v)` is a gufunc with signature `(3),(3)->(3)`.  It computes
  the 3-d vector cross product (like `numpy.cross`, but specialized to the
  case of 3-d vectors only).

### `backlash`

* `backlash(x, deadband, initial)`, a gufunc with signature `(i),(),()->(i)`,
  computes the "backlash" response of a signal; see
  https://en.wikipedia.org/wiki/Backlash_(engineering).
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

### `deadzone`

* `deadzone(x, low, high)` is ufunc with three inputs and one output.
  It computes the "deadzone" response of a signal::

             { 0         if low <= x <= high
      f(x) = { x - low   if x < low
             { x - high  if x > high

  The function is similar to the
  [deadzone block](https://www.mathworks.com/help/simulink/slref/deadzone.html)
  of Matlab's Simulink library.  The function is also known as
  a *soft threshold*.

  The script `deadzone_demo.py` in the `examples` directory generates
  the plot

  ![Deadzone plot](https://github.com/WarrenWeckesser/ufunclab/blob/main/examples/deadzone_demo.png)

### `hysteresis_relay`

* `hysteresis_relay(x, low_threshold, high_threshold, low_value, high_value, init)`
  a gufunc with signature `(i),(),(),(),(),()->(i)`, passes `x` through a relay
  with hysteresis (like a Schmitt trigger). The function is similar to the
  [relay block](https://www.mathworks.com/help/simulink/slref/relay.html)
  of Matlab's Simulink library.

  The script `hysteresis_relay_demo.py` in the `examples` directory generates
  the plot

  ![hysteresis_replay plot](https://github.com/WarrenWeckesser/ufunclab/blob/main/examples/hysteresis_relay_demo.png)

### `ufunc_inspector`

* Print information about a NumPy ufunc.

  For example,

  ```
  In [10]: import numpy as np

  In [11]: from ufunclab import ufunc_inspector

  In [12]: ufunc_inspector(np.hypot)
  'hypot' is a ufunc.
  nin = 2, nout = 1, ntypes = 5
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
  nin = 1, nout = 1, ntypes = 11
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
