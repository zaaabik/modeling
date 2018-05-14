from collections import defaultdict

import numpy as np


class LaggedFibonacciGenerator(object):
    def __init__(self):
        self.data_array = np.random.randint(0, 2 ** 32 - 1, dtype='uint32', size=55)

    def rand(self):
        last = (int(self.data_array[0]) + self.data_array[30]) % (2 ** 32)
        self.data_array = np.delete(self.data_array, 0)
        self.data_array = np.append(self.data_array, last)
        return last / 2 ** 32

    def rand_array(self, size):
        res = np.zeros(size, dtype='float')
        for i in range(0, size):
            res[i] = self.rand()
        return res

    @staticmethod
    def math_expect(array):
        sum = np.sum(array)
        return sum / array.size

    @staticmethod
    def dispersion(array):
        math_expect = LaggedFibonacciGenerator.math_expect(array)
        array = (array - math_expect) ** 2
        return LaggedFibonacciGenerator.math_expect(array)

    @staticmethod
    def test(array):
        freq = defaultdict(int)
        for i in range(array.shape[0]):
            freq[array[i]] += 1
        return freq

    @staticmethod
    def show_histo(array):
        size = len(array)
        hist = defaultdict(float)
        for i in array:
            hist[i] += 1 / size
        return hist

    @staticmethod
    def probability_distribution(array):
        delta = 0.1
        a = int(np.floor(min(array)))
        b = int(np.ceil(max(array)))
        y = [0] * (b * 10)
        x = np.arange(0, b, delta)
        for i in x:
            l = i
            r = i + delta
            y[int(i * 10)] = ((l < array) & (array < r)).sum()
        return x, y

    @staticmethod
    def probability_distribution_negative(array):
        delta = 0.1
        x = np.arange(-15, 15, delta)
        y = defaultdict(float)
        l = len(list(x))
        for i in x:
            l = i
            r = i + delta
            y[i] = ((l < array) & (array < r)).sum()
        return y.keys(), y.values()

    @staticmethod
    def correlation(a, b):
        exp_val_a = LaggedFibonacciGenerator.math_expect(a)
        exp_val_b = LaggedFibonacciGenerator.math_expect(b)
        res_top = 0
        res_bot1 = 0
        res_bot2 = 0
        for x, y in zip(np.nditer(a), np.nditer(b)):
            res_top += (x - exp_val_a) * (y - exp_val_b)
            res_bot1 += ((x - exp_val_a) ** 2)
            res_bot2 += ((y - exp_val_b) ** 2)

        return res_top / np.sqrt(res_bot1 * res_bot2)

    @staticmethod
    def auto_correlation(array, s, t):
        a = array[s:]
        b = array[:-s]
        crl = []
        for i in t:
            crl.append(np.math.fabs(LaggedFibonacciGenerator.correlation(a[:i], b[:i])))
        return crl
