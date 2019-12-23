import CubicSolver
import numpy as np

# def test_class0():
#     cubic =  CubicSolver.Cubic(0, 1, 2, -3)
#     x1, x2, x3 = cubic.solve()
#     np.testing.assert_almost_equal(x1, 1)
#     np.testing.assert_almost_equal(x2, -3)


def test_class1():
    cubic =  CubicSolver.Cubic(1, -5, 19, -15)
    x1, x2, x3 = cubic.solve()
    np.testing.assert_almost_equal(x1, 1)
    np.testing.assert_almost_equal(x2, 2 + 1j * np.sqrt(11))
    np.testing.assert_almost_equal(x3, 2 - 1j * np.sqrt(11))


def test_class2():
    cubic =  CubicSolver.Cubic(1, -11, 35, -49)
    x1, x2, x3 = cubic.solve()
    np.testing.assert_almost_equal(x1, 7)
    np.testing.assert_almost_equal(x2, 2 - 1j * np.sqrt(3))
    np.testing.assert_almost_equal(x3, 2 + 1j * np.sqrt(3))


def test_class3():
    cubic =  CubicSolver.Cubic(1, 1, -11, 45)
    x1, x2, x3 = cubic.solve()
    np.testing.assert_almost_equal(x1, -5)
    np.testing.assert_almost_equal(x2, 2 + 1j * np.sqrt(5))
    np.testing.assert_almost_equal(x3, 2 - 1j * np.sqrt(5))


def test_class4():
    cubic = CubicSolver.Cubic(1, 6, 11, 6)
    x1, x2, x3 = cubic.solve()

    np.testing.assert_almost_equal(x1, -1)
    np.testing.assert_almost_equal(x2, -3)
    np.testing.assert_almost_equal(x3, -2)



def test_1000_random_cubics():
    N = 1000
    for i in range(N):
        coeffs = np.random.randint(-100, 100, 4)
        if coeffs[0] == 0:
            continue
        cubic = CubicSolver.Cubic(*coeffs)
        soln = cubic.solve()

        for x in soln:
            val = np.polyval(coeffs, x)
            np.testing.assert_almost_equal(val, 0)
