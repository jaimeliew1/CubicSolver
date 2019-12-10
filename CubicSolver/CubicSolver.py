import numpy as np
import cmath


def classify(a, b, c, d):
    '''
    Classify the cubic into one of four categories:
    - category 0: a = 0 (quadratic)
    - category 1: delta^2 < 0 (monotonic, one real root)
    - category 2: yn/h < -1 (one real root, 2 turning points)
    - category 3: yn/h > 1 (one real root, 2 turning points)
    - category 4: |yn/h| <= 1 (3 real roots)
    '''
    if a == 0:
        return 0

    delta_sqr = (b**2 - 3 * a * c) / (9 * a**2)

    if delta_sqr < 0:
        return 1

    x_n = -b / (3 * a)
    y_n = a * x_n**3 + b * x_n**2 + c * x_n + d
    delta = np.sqrt(delta_sqr)
    h = 2 * a * delta**3

    if y_n / h < -1:
        return 2
    elif y_n / h > 1:
        return 3
    elif abs(y_n / h) <= 1:
        return 4
    else:
        raise(ValueError('Cubic cannot be classified.'))


def solve_class0(a, b, c, d):
    x1 = (-c + cmath.sqrt(c**2 - 4 * b * d)) / (2 * b)
    x2 = (-c - cmath.sqrt(c**2 - 4 * b * d)) / (2 * b)

    return [x1, x2]


def solve_class1(a, b, c, d):
    delta = np.sqrt(-(b**2 - 3 * a * c) / (9 * a**2))
    x_n = -b / (3 * a)
    y_n = a * x_n**3 + b * x_n**2 + c * x_n + d
    h = 2 * a * delta**3

    Z = 1 / 3 * np.arcsinh(y_n / h)
    alpha = x_n - 2 * delta * np.sinh(Z)
    beta = x_n + delta * (np.sinh(Z) + 1j * np.sqrt(3) * np.cosh(Z))
    gamma = x_n + delta * (np.sinh(Z) - 1j * np.sqrt(3) * np.cosh(Z))

    return [alpha, beta, gamma]


def solve_class2(a, b, c, d):
    delta = np.sqrt((b**2 - 3 * a * c) / (9 * a**2))
    x_n = -b / (3 * a)
    y_n = a * x_n**3 + b * x_n**2 + c * x_n + d
    h = 2 * a * delta**3

    Z = 1 / 3 * np.arccosh(-y_n / h)
    alpha = x_n + 2 * delta * np.cosh(Z)
    beta = x_n - delta * (np.cosh(Z) + 1j * np.sqrt(3) * np.sinh(Z))
    gamma = x_n - delta * (np.cosh(Z) - 1j * np.sqrt(3) * np.sinh(Z))

    return [alpha, beta, gamma]


def solve_class3(a, b, c, d):
    delta = np.sqrt((b**2 - 3 * a * c) / (9 * a**2))
    x_n = -b / (3 * a)
    y_n = a * x_n**3 + b * x_n**2 + c * x_n + d
    h = 2 * a * delta**3

    Z = 1 / 3 * np.arccosh(y_n / h)
    alpha = x_n - 2 * delta * np.cosh(Z)
    beta = x_n + delta * (np.cosh(Z) + 1j * np.sqrt(3) * np.sinh(Z))
    gamma = x_n + delta * (np.cosh(Z) - 1j * np.sqrt(3) * np.sinh(Z))

    return [alpha, beta, gamma]


def solve_class4(a, b, c, d):
    delta = np.sqrt((b**2 - 3 * a * c) / (9 * a**2))
    x_n = -b / (3 * a)
    y_n = a * x_n**3 + b * x_n**2 + c * x_n + d
    h = 2 * a * delta**3

    Z = 1 / 3 * np.arccos(-y_n / h)
    alpha = x_n + 2 * delta * np.cos(Z)
    beta = x_n + 2 * delta * np.cos(Z + 2 / 3 * np.pi)
    gamma = x_n + 2 * delta * np.cos(Z - 2 / 3 * np.pi)

    return [alpha, beta, gamma]


solver = [solve_class0, solve_class1, solve_class2, solve_class3, solve_class4]


def solve(a, b, c, d):
    Class = classify(a, b, c, d)
    X = solver[Class](a, b, c, d)
    return X


if __name__ == '__main__':
    print(solve(0, 1, 2, -3))
    print(solve(1, -5, 19, -15))
