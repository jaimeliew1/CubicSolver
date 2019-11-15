import CubicSolver
import numpy as np


def test_class0():
    x1, x2 = CubicSolver.solve(0, 1, 2, -3)
    np.testing.assert_almost_equal(x1, 1)
    np.testing.assert_almost_equal(x2, -3)


def test_class1():
    x1, x2, x3 = CubicSolver.solve(1, -5, 19, -15)
    np.testing.assert_almost_equal(x1, 1)
    np.testing.assert_almost_equal(x2, 2 + 1j*np.sqrt(11))
    np.testing.assert_almost_equal(x3, 2 - 1j*np.sqrt(11))


def test_class2():
    x1, x2, x3 = CubicSolver.solve(1, -11, 35, -49)
    np.testing.assert_almost_equal(x1, 7)
    np.testing.assert_almost_equal(x2, 2 - 1j*np.sqrt(3))
    np.testing.assert_almost_equal(x3, 2 + 1j*np.sqrt(3))


def test_class3():
    x1, x2, x3 = CubicSolver.solve(1, 1, -11, 45)
    np.testing.assert_almost_equal(x1, -5)
    np.testing.assert_almost_equal(x2, 2 + 1j*np.sqrt(5))
    np.testing.assert_almost_equal(x3, 2 - 1j*np.sqrt(5))



def test_class4():
    x1, x2, x3 = CubicSolver.solve(1, 6, 11, 6)

    np.testing.assert_almost_equal(x1, -1)
    np.testing.assert_almost_equal(x2, -3)
    np.testing.assert_almost_equal(x3, -2)
