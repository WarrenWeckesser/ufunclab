ufunclab
========

Some NumPy `ufuncs`, and some related tools.

Requires at least Python 3.5.

What's in ufunclab?
-------------------

### `ufunclab.logfactorial`

* Computes the natural logarithm of the factorial of the integer x.

* This is a fairly standard implementation of a NumPy ufunc.

### `ufunclab.peaktopeak`

* A `gufunc` that computes the peak-to-peak range of a NumPy array.
  It is like the `ptp` method of a NumPy array, but when the input
  is signed integers, the output is an unsigned integer with the
  same bit width.

  Only the standard integer and floating point types are implemented;
  the function will not accept complex arrays, nor arrays with the data
  types `datetime64`, `timedelta64` or `object`.


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

### `ufunclab.minmax`

* A `gufunc` that simultaneously computes the minimum and maximum of a NumPy
  array.  (The ufunc signature is '(i)->(2)'.)

  Only the standard integer and floating point types are implemented;
  the function will not accept complex arrays, nor arrays with the data
  types `datetime64`, `timedelta64` or `object`.

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
  Found type dd->d
  Let's try calling the inner loop function.
  x =   3.000000, y =   4.000000, z =   5.000000

  Found PyUFunc_dd_d
  Let's try calling ufunc->data[i]
  x =   3.000000, y =   4.000000, z =   5.000000
  ```
  The output will certainly change as the code develops.  The output
  starting with `Found type...` is part of an experiment, and will
  eventually be removed.

  ```
  In [16]: ufunc_inspector(np.sqrt)                                                                                                 
  'sqrt' is a ufunc.
  nin = 1, nout = 1, ntypes = 11
    0:   23 ->  23  (e->e)  PyUFunc_e_e_As_f_f
    1:   11 ->  11  (f->f)  not generic (or not in the checked generics)
    2:   12 ->  12  (d->d)  not generic (or not in the checked generics)
    3:   23 ->  23  (e->e)  PyUFunc_e_e_As_f_f
    4:   11 ->  11  (f->f)  PyUFunc_f_f
    5:   12 ->  12  (d->d)  PyUFunc_d_d
    6:   13 ->  13  (g->g)  PyUFunc_g_g
    7:   14 ->  14  (F->F)  PyUFunc_F_F
    8:   15 ->  15  (D->D)  PyUFunc_D_D
    9:   16 ->  16  (G->G)  PyUFunc_G_G
   10:   17 ->  17  (O->O)  PyUFunc_O_O_method

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
  Found type dd->d
  Let's try calling the inner loop function.
  x =   3.000000, y =   4.000000, z =   7.000000
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
