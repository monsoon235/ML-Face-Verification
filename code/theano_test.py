import theano.tensor as T
import theano
import numpy as np

floatX = 'float32'

theano.config.floatX = 'float32'

x = T.matrix()
y = T.matrix()
print(x.broadcastable)
print(y.broadcastable)
z = x + y
# z.eval({x: [[1, 2]], y: [[2, 6], [6, 2]]})

MyType1 = T.TensorType(dtype=floatX, broadcastable=(False, True, False, False, False))
MyType2 = T.TensorType(dtype=floatX, broadcastable=(True, False, True, True, True))

t1 = MyType1('t1')
t2 = MyType2('t2')

f = theano.function([t1, t2], T.sum(t1 + t2, axis=(2,)))

a = np.ones((10, 1, 3, 3, 3), dtype=np.float32)
b = np.ones((1, 10, 1, 1, 1), dtype=np.float32)

print(type(f(a, b)))
print(f(a, b))

print()

if __name__ == '__main__':
    pass

# r = T.row()
# print(r.broadcastable)
# # (True, False)
# mtr = T.matrix()
# print(mtr.broadcastable)
# # (False, False)
# f_row = theano.function([r, mtr], [r + mtr])
# R = np.arange(3).reshape(1, 3)
# print(R)
# # array([[0, 1, 2]])
# M = np.arange(9).reshape(3, 3)
# print(M)
# # array([[0, 1, 2],
# #        [3, 4, 5],
# #        [6, 7, 8]])
# f_row(R, M)
# # [array([[  0.,   2.,   4.],
# #        [  3.,   5.,   7.],
# #        [  6.,   8.,  10.]])]
# c = T.col()
# print(c.broadcastable)
# # (False, True)
# f_col = theano.function([c, mtr], [c + mtr])
# C = np.arange(3).reshape(3, 1)
# print(C)
# # array([[0],
# #        [1],
# #        [2]])
# M = np.arange(9).reshape(3, 3)
# f_col(C, M)
# # [array([[  0.,   1.,   2.],
# #        [  4.,   5.,   6.],
# #        [  8.,   9.,  10.]])]
