import optparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import DiscreteRand as d
import generator


def main():
    p = optparse.OptionParser()
    p.add_option('--probability', '-p')
    p.add_option('--value', '-v')

    options, arguments = p.parse_args()
    probability = list(map(float, options.probability.split(',')))
    value = list(map(float, options.value.split(',')))

    rand = d.DiscreteRand(probability, value)
    print(rand.math_expect())
    print(rand.dispersion())

    rand_value = []
    g = generator.LaggedFibonacciGenerator()
    rand_array = g.rand_array(500)
    for i in range(0, 500):
        rand_value.append(rand.get_val(rand_array[i]))

    print('empirical m', rand.math_expect_empirical(rand_value))
    print('empirical d', rand.dispersion_empirical(rand_value))

    hist = g.show_histo(rand_value)
    a = np.array(value)

    plt.bar(hist.keys(), hist.values(), width=1, label="Bar 1")
    emp = mpatches.Patch(color='blue', label='empirical')
    teor = mpatches.Patch(color='red', label='theoretical')
    plt.legend(handles=[emp, teor])
    b = plt.bar(a + 1, probability, color='r', width=1, label="Bar 1")
    plt.show()


if __name__ == '__main__':
    main()
