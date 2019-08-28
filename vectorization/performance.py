import numpy
import time

a = numpy.random.rand(1000000)
b = numpy.random.rand(1000000)

start = time.time()
c = numpy.dot(a, b)
end = time.time()

print(c)
print('Vectorized version: ', str(1000 * (end - start)), 'ms')

c = 0
start = time.time()
for i in range(1000000):
    c += a[i] * b[i]
end = time.time()

print(c)
print('For loop: ', str(1000 * (end - start)), 'ms')
