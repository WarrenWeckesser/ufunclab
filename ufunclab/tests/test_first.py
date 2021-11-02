import pytest
from fractions import Fraction as F
from decimal import Decimal
import numpy as np
from numpy.testing import assert_equal
from ufunclab import first, argfirst, op


@pytest.mark.parametrize('x, cmp, target, otherwise, loc',
                         [(np.array([10, 35, 19, 0, -1, 24, 0]),
                           op.EQ, 0, 0, 3),
                          (np.array([0, 0, 0, 0, 0, -0.5, 0, 1, 0.1]),
                           op.NE, 0.0, 0.0, 5),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.GT, np.int8(3), 0, 4),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.GE, np.int8(3), 0, 2),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.LT, np.int8(0), 0, 0),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.LE, np.int8(-3), 0, -1),
                          (np.array([6, 9, 4, 4, 3, 1, 7], dtype=np.int8),
                           op.LE, np.int8(3), 0, 4),
                          ])
def test_first_basic(x, cmp, target, otherwise, loc):
    expected = x[loc] if loc != -1 else otherwise
    assert first(x, cmp, target, otherwise) == expected


def test_first_2d():
    b = np.array([[0, 8, 0, 0], [0, 0, 0, 0], [0, 0, 9, 2]],
                 dtype=np.uint8)
    f1 = first(b, op.NE, np.uint8(0), np.uint8(0))
    assert f1.dtype == b.dtype
    assert_equal(f1, [8, 0, 9])
    f0 = first(b, op.NE, np.uint8(0), np.uint8(0), axis=0)
    assert f0.dtype == b.dtype
    assert_equal(f0, [0, 8, 9, 2])


def test_first_broadcasting_target():
    a = np.array([[4.5, 2.0, 0.0, 8.0],
                  [1.0, 0.0, 1.5, 3.6],
                  [2.0, 2.0, 3.0, 3.0]])
    # Find the first occurrences of 0.0 and 1.0 in each row.
    v = first(a, op.LE, np.array([[0.0], [1.0]]), np.nan)
    assert_equal(v, [[0.0, 0.0, np.nan], [0.0, 1.0, np.nan]])


@pytest.mark.parametrize('op, target, otherwise, expected',
                         [(op.EQ, 0, 0, 0),
                          (op.EQ, None, "not found", "not found"),
                          (op.EQ, F(9, 10), 0, F(9, 10)),
                          (op.LT, 0, 0, F(-1, 3)),
                          (op.LT, -5.0, 0, 0),
                          (op.NE, Decimal('0.5'), 0, F(9, 10))])
def test_firstobject(op, target, otherwise, expected):
    a = np.array([F(1, 2), F(9, 10), F(-1, 3), 0, 1.5], dtype=object)
    v = first(a, op, target, otherwise)
    assert v == expected


@pytest.mark.parametrize('x, cmp, target, loc',
                         [(np.array([10, 35, 19, 0, -1, 24, 0]),
                           op.EQ, 0, 3),
                          (np.array([0, 0, 0, 0, 0, -0.5, 0, 1, 0.1]),
                           op.NE, 0.0, 5),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.GT, np.int8(3), 4),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.GE, np.int8(3), 2),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.LT, np.int8(0), 0),
                          (np.array([-1, 1, 3, 3, 9, 1, 2], dtype=np.int8),
                           op.LE, np.int8(-3), -1),
                          (np.array([6, 9, 4, 4, 3, 1, 7], dtype=np.int8),
                           op.LE, np.int8(3), 4),
                          ])
def test_argfirst_basic(x, cmp, target, loc):
    assert argfirst(x, cmp, target) == loc


def test_argfirst_2d():
    b = np.array([[0, 8, 0, 0], [0, 0, 0, 0], [0, 0, 9, 2]],
                 dtype=np.uint8)
    i = argfirst(b, op.NE, np.uint8(0))
    assert_equal(i, [1, -1, 2])
    j = argfirst(b, op.NE, np.uint8(0), axis=0)
    assert_equal(j, [-1, 0, 2, 2])


def test_argfirst_broadcasting_target():
    a = np.array([[4.5, 2.0, 0.0, 8.0],
                  [1.5, 0.0, 1.0, 3.6],
                  [2.0, 2.0, 3.0, 3.0]])
    # Find the first occurrences of 0.0 and 1.0 in each row.
    i = argfirst(a, op.EQ, np.array([[0.0], [1.0]]))
    assert_equal(i, [[2, 1, -1], [-1, 2, -1]])


def test_broadcasting_op():
    a = np.array([[4.5, 2.0, 0.0, 8.0],
                  [1.5, 0.0, 1.0, 3.6],
                  [2.0, 2.0, 3.0, 3.0]])
    # In each row, find the first occurrence of the a value less
    # than 2, and the first occurrence of the value greater than 2.
    i = argfirst(a, [[op.LT], [op.GT]], 2.0)
    assert_equal(i, [[2, 0, -1], [0, 3, 2]])


@pytest.mark.parametrize('op, target, loc', [(op.EQ, 0, 3),
                                             (op.EQ, None, -1),
                                             (op.EQ, F(9, 10), 1),
                                             (op.LT, 0, 2),
                                             (op.LT, -5.0, -1),
                                             (op.NE, Decimal('0.5'), 1)])
def test_argfirstobject(op, target, loc):
    a = np.array([F(1, 2), F(9, 10), F(-1, 3), 0, 1.5], dtype=object)
    i = argfirst(a, op, target)
    assert i == loc


def test_object_bad_op():
    a = np.array([F(1, 2), F(9, 10)], dtype=object)
    i = argfirst(a, 6, 0.5)
    assert i == -1
