import optparse

import matplotlib.pyplot as plt
import numpy as np

import generator

p = optparse.OptionParser()
p.add_option('--len', '-l')
p.add_option('--k', '-k')
p.add_option('--lambda', '-a')
p.add_option('--d_g', '-d')
p.add_option('--m_g', '-m')
options, arguments = p.parse_args()

g = generator.LaggedFibonacciGenerator()
randArray = g.rand_array(int(options.len))

# exp
l = 0.5
exp = (-1 / l) * np.log(randArray)
m = g.math_expect(exp)
d = g.dispersion(exp)
print('\n')
print('exp')
print('lambda ', l)
print('m = ', m)
print('d = ', d)
print('expected m = ', 1 / l)
print('expected d = ', 1 / (l ** 2))
x, y = g.probability_distribution(exp)
plt.bar(x, y)
plt.title('exp.png')
plt.show()

# Uniform
a = 3
b = 15
un = a + (b - a) * randArray
x, y = g.probability_distribution(un)
m = g.math_expect(un)
d = g.dispersion(un)
print('\n')
print('uniform')
print('a ', a)
print('b ', b)
print('m = ', m)
print('d = ', d)
print('expected m = ', a + (b - a) / 2)
print('expected d = ', ((b - a) ** 2) / 12)
plt.bar(x, y)
plt.title('uniform')
plt.show()

# erlang
er = np.zeros((int(options.len)), np.float64)
k = int(options.k)
for i in range(0, k):
    g = generator.LaggedFibonacciGenerator()
    array = (-1 / l) * np.log(g.rand_array(int(options.len)))
    er += array

x, y = g.probability_distribution(er)
m = g.math_expect(er)
d = g.dispersion(er)
print('\n')
print('erlang')
print('k = ', k)
print('lambda = ', l)
print('m = ', m)
print('d = ', d)
print('expected m = ', k / l)
print('expected d = ', k / l ** 2)
plt.bar(x, y)
plt.title('erlang.png')
plt.show()

# normal
randArray2 = g.rand_array(int(options.len))
normal = np.sqrt(-2 * np.log(randArray2)) * np.cos(2 * np.pi * randArray)
x, y = g.probability_distribution_negative(normal)
m = g.math_expect(normal)
d = g.dispersion(normal)
print('\n')
print('normal')
print('m = ', m)
print('d = ', d)
print('expected m = ', 0)
print('expected d = ', 1)
plt.plot(x, y)
plt.title('normal.png')
plt.show()

# gauss
m_exp = 5
d_exp = 2
x = randArray
gauss = normal * d_exp + m_exp
m = g.math_expect(gauss)
d = g.dispersion(gauss)
print('\n')
print('gauss')
print('m = ', m)
print('d = ', d)
print('expected m = ', m_exp)
print('expected d = ', d_exp ** 2)
x, y = g.probability_distribution_negative(gauss)
plt.plot(x, y)
plt.title('gauss.png')
plt.show()
