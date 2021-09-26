import pytest
import numpy as np
from numpy.testing import assert_equal
from ufunclab import findfirst, op


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
def test_basic(x, cmp, target, loc):
    assert findfirst(x, cmp, target) == loc


def test_2d():
    b = np.array([[0, 8, 0, 0], [0, 0, 0, 0], [0, 0, 9, 2]],
                 dtype=np.uint8)
    i = findfirst(b, op.NE, np.uint8(0))
    assert_equal(i, [1, -1, 2])
    j = findfirst(b, op.NE, np.uint8(0), axis=0)
    assert_equal(j, [-1, 0, 2, 2])


def test_broadcasting_target():
    a = np.array([[4.5, 2.0, 0.0, 8.0],
                  [1.5, 0.0, 1.0, 3.6],
                  [2.0, 2.0, 3.0, 3.0]])
    # Find the first occurrences of 0.0 and 1.0 in each row.
    i = findfirst(a, op.EQ, np.array([[0.0], [1.0]]))
    assert_equal(i, [[2, 1, -1], [-1, 2, -1]])


def test_broadcasting_op():
    a = np.array([[4.5, 2.0, 0.0, 8.0],
                  [1.5, 0.0, 1.0, 3.6],
                  [2.0, 2.0, 3.0, 3.0]])
    # In each row, find the first occurrence of the a value less
    # than 2, and the first occurrence of the value greater than 2.
    i = findfirst(a, [[op.LT], [op.GT]], 2.0)
    assert_equal(i, [[2, 0, -1], [0, 3, 2]])
