import numpy as np
import cmath


def depress(a, b, c, d):
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    return p, q


class Cubic(object):


    def __init__(self, *coefs):
        assert len(coefs) == 4
        assert coefs[0] != 0
        self.solver = {
            1: self.solve_class1,
            2: self.solve_class2,
            3: self.solve_class3,
            4: self.solve_class4,
        }


        self.coefs = coefs
        self.p, self.q = depress(*coefs)

    @property
    def x_n(self):
        a, b, c, d = self.coefs
        return -b / (3 * a)

    @property
    def y_n(self):
        return self.q

    @property
    def delta(self):
        return cmath.sqrt(-self.p / 3)

    @property
    def h(self):
        return 2 / 3 * self.p * cmath.sqrt(-self.p) / cmath.sqrt(3)

    @property
    def descriminant(self):
        return (self.q / 2)**2 + (self.p / 3)**3

    @property
    def descriminant2(self):
        out = self.y_n / self.h
        if np.isnan(np.imag(out)):
            out = np.real(out)
        return out

    def classify(self):
        '''
        Classify the cubic into one of four categories:
        - category 0: a = 0 (quadratic)
        - category 1: delta^2 < 0 (monotonic, one real root)
        - category 2: yn/h < -1 (one real root, 2 turning points)
        - category 3: yn/h > 1 (one real root, 2 turning points)
        - category 4: |yn/h| <= 1 (3 real roots)
        '''

        if np.iscomplex(self.descriminant2):
            return 1
        elif np.real(self.descriminant2) < -1:
            return 2
        elif np.real(self.descriminant2) > 1:
            return 3
        elif abs(self.descriminant2) <= 1:
            return 4
        else:
            raise (ValueError('Cubic cannot be classified.'))

    def solve(self):
        the_class = self.classify()
        roots = self.solver[the_class]()
        return roots

    def solve_class1(self):
        Z = 1 / 3 * np.arcsinh(np.imag(-self.descriminant2))
        alpha = self.x_n - 2 * np.imag(self.delta) * np.sinh(Z)
        beta = self.x_n + np.imag(self.delta) * (np.sinh(Z) +
                                        1j * np.sqrt(3) * np.cosh(Z))
        gamma = self.x_n + np.imag(self.delta) * (np.sinh(Z) -
                                         1j * np.sqrt(3) * np.cosh(Z))
        return [alpha, beta, gamma]

    def solve_class3(self):
        Z = 1 / 3 * np.arccosh(self.descriminant2)
        alpha = self.x_n + 2 * self.delta * np.cosh(Z)
        beta = self.x_n - self.delta * (np.cosh(Z) +
                                        1j * np.sqrt(3) * np.sinh(Z))
        gamma = self.x_n - self.delta * (np.cosh(Z) -
                                         1j * np.sqrt(3) * np.sinh(Z))

        return [alpha, beta, gamma]

    def solve_class2(self):
        Z = 1 / 3 * np.arccosh(-self.descriminant2)
        alpha = self.x_n - 2 * self.delta * np.cosh(Z)
        beta = self.x_n + self.delta * (np.cosh(Z) +
                                        1j * np.sqrt(3) * np.sinh(Z))
        gamma = self.x_n + self.delta * (np.cosh(Z) -
                                         1j * np.sqrt(3) * np.sinh(Z))


        return [alpha, beta, gamma]

    def solve_class4(self):
        Z = 1 / 3 * np.arccos(self.descriminant2)
        alpha = self.x_n + 2 * self.delta * np.cos(Z)
        beta = self.x_n + 2 * self.delta * np.cos(Z + 2 / 3 * np.pi)
        gamma = self.x_n + 2 * self.delta * np.cos(Z - 2 / 3 * np.pi)

        return [alpha, beta, gamma]
