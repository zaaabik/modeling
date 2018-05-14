from collections import defaultdict

import numpy as np


class DiscreteRand(object):
    def __init__(self, probability, values):
        self.values = values
        self.probs = [0]
        self.original_probs = probability
        cur = 0
        for i in probability:
            cur += i
            self.probs.append(cur)

    def get_val(self, val):
        for i in range(0, len(self.probs) - 1):
            if self.probs[i] <= val < self.probs[i + 1]:
                return self.values[i]

    def math_expect(self):
        m = 0.
        for x, y in zip(self.values, self.original_probs):
            m += x * y
        return m

    def dispersion(self):
        m = self.math_expect()
        d = 0
        for x, y in zip(self.values, self.original_probs):
            d += (x ** 2) * y
        return d - m ** 2

    @staticmethod
    def math_expect_empirical(values):
        return np.average(values)

    @staticmethod
    def dispersion_empirical(values):
        math_expect = DiscreteRand.math_expect_empirical(values)
        array = (values - math_expect) ** 2
        return DiscreteRand.math_expect_empirical(array)

    @staticmethod
    def show_histogram(array):
        len = array.shape[0]
        hist = defaultdict(float)
        for i in array:
            hist[i] += 1 / len
        return hist
