ufunclab
========

Some NumPy `ufuncs`, and some related tools.

Requires at least Python 3.5.

What's in ufunclab?
-------------------

`ufunclab` defines these functions:

* `logfactorial`
* `peaktopeak`
* `minmax`
* `argminmax`
* `min_argmin` and `max_argmax`
* `gmean` (geometric mean) and `hmean` (harmonic mean)
* `ufunc_inspector`

Details follow.


### `ufunclab.logfactorial`

* Computes the natural logarithm of the factorial of the integer x.

* This is a fairly standard implementation of a NumPy ufunc.

### `ufunclab.peaktopeak`

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

### `ufunclab.minmax`

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

### `ufunclab.argminmax`

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

### `ufunclab.min_argmin` and `ufunclab.max_argmax`

* These functions return both the extreme value and the index of the extreme
  value.  (The ufunc signature of these functions is '(i)->(),()'.)

  ```
  >>> np.random.seed(123456)
  >>> x = np.random.randint(0, 20, size=(3, 5))
  >>> x
  array([[ 1, 10, 18, 17, 11],
         [15, 11,  0,  4,  8],
         [10, 10, 12, 11, 11]])
  >>> min_argmin(x, axis=1)
  (array([ 1,  0, 10]), array([0, 2, 0]))

  >>> from fractions import Fraction as F
  >>> y = np.array([F(2, 3), F(3, 4), F(2, 7), F(2, 5)])
  >>> max_argmax(y)
  (Fraction(3, 4), 1)
  ```

### `ufunclab.gmean` and `ufunclab.hmean`

* These gufuncs compute the geometric and harmonic means, respectively.

  For example,

  ```
  In [25]: import numpy as np

  In [26]: from ufunclab import gmean, hmean

  In [27]: x = np.array([1, 2, 3, 5, 8], dtype=np.uint8)

  In [28]: hmean(x)
  Out[28]: 2.316602316602317

  In [29]: gmean(x)
  Out[29]: 2.992555739477689

  In [30]: np.mean(x)
  Out[30]: 3.8

  In [31]: y = np.arange(1, 16).reshape(3, 5)

  In [32]: y
  Out[32]:
  array([[ 1,  2,  3,  4,  5],
         [ 6,  7,  8,  9, 10],
         [11, 12, 13, 14, 15]])

  In [33]: hmean(y, axis=1)
  Out[33]: array([ 2.18978102,  7.74431469, 12.84486077])

  In [34]: gmean(y, axis=1)
  Out[34]: array([ 2.60517108,  7.87256685, 12.92252305])

  In [35]: np.mean(y, axis=1)
  Out[35]: array([ 3.,  8., 13.])
  ```

### `ufunclab.ufunc_inspector`

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

Resources for learning about the C API for ufuncs
------------------------------------------------
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
  [NumPy distutils](https://docs.scipy.org/doc/numpy/reference/distutils_guide.html)
  [template preprocessor](https://docs.scipy.org/doc/numpy/reference/distutils_guide.html#conversion-of-src-files-using-templates)
  is a useful tool. (See the ["Other files"](https://docs.scipy.org/doc/numpy/reference/distutils_guide.html#other-files)
  section for the syntax that would be used in, say, a C file.)
* Some relevant NumPy source code, if you want to dive deep:
  * `PyUFuncObject` along with related C types and macros are defined in
   [`numpy/numpy/core/include/numpy/ufuncobject.h`](https://github.com/numpy/numpy/blob/7214ca4688545b432c45287195e2f46c5e418ce8/numpy/core/include/numpy/ufuncobject.h).
  * `PyUFunc_FromFuncAndData` and `PyUFunc_FromFuncAndDataAndSignatureAndIdentity`
    are defined in the file [`numpy/numpy/core/src/umath/ufunc_object.c`](https://github.com/numpy/numpy/blob/7214ca4688545b432c45287195e2f46c5e418ce8/numpy/core/src/umath/ufunc_object.c).
* [Data Type API](https://docs.scipy.org/doc/numpy/reference/c-api.dtype.html) --
  a handy reference.
