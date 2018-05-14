import matplotlib.pyplot as plt
import optparse

import generator

p = optparse.OptionParser()
p.add_option('--len', '-l')
p.add_option('--cor_len', '-c')
options, arguments = p.parse_args()

g = generator.LaggedFibonacciGenerator()
randArray = g.rand_array(int(options.len))

print("m = ", g.math_expect(randArray))
print("d = ", g.dispersion(randArray))
freq = g.test(randArray)

for k, v in freq.items():
    if v != 1:
        print(v)

d = g.show_histo(randArray)

x, y = g.probability_distribution(randArray)
plt.plot(x, y)
plt.title("probability distribution")
plt.show()
t = range(10, int(options.cor_len), 20)
crl3 = g.auto_correlation(randArray, 3, t)
crl5 = g.auto_correlation(randArray, 5, t)
crl10 = g.auto_correlation(randArray, 10, t)
plt.plot(list(t), crl3, 'g', color='g')
plt.plot(list(t), crl5, 'b', color='b')
plt.plot(list(t), crl10, 'r', color='r')
plt.title("correlation")
plt.show()
