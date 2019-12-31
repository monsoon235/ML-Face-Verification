from typing import Tuple
import numpy as np
import time
import torch

# import theano.tensor as T
# import theano

floatX = np.float32
theano_type = 'float32'
torch_type = torch.float32


class FullConnection:
    batch: int
    height: int
    width: int
    channel: int
    dim: int
    learning_rate: float

    # params: np.ndarray  # （dim, height, width, channel)
    # biases: np.ndarray  # (dim, )
    params: torch.Tensor
    biases: torch.Tensor
    in_data: np.ndarray
    out_data: np.ndarray
    grad_params: torch.Tensor
    grad_biases: torch.Tensor

    # forward_func: theano.compile.Function
    # gradient_func: theano.compile.function

    def __init__(self, input_shape, dim, learning_rate=0.0001) -> None:
        self.batch, self.height, self.width, self.channel = input_shape
        self.dim = dim
        self.learning_rate = learning_rate

        # self.params = np.random.randn(dim, self.height, self.width, self.channel).astype(floatX)
        # self.biases = np.random.randn(dim).astype(floatX)

        self.params = torch.randn((dim, self.height, self.width, self.channel), dtype=torch_type) / 100
        self.biases = torch.randn((dim,), dtype=torch_type) / 100

        # type1 = T.TensorType(dtype=theano_type, broadcastable=(False, True, False, False, False))
        # type2 = T.TensorType(dtype=theano_type, broadcastable=(True, False, False, False, False))
        # type3 = T.TensorType(dtype=theano_type, broadcastable=(True, False))
        # in_data_theano = type1('in_data_theano')
        # params_theano = type2('params_theano')
        # biases_theano = type3('biases_theano')
        # self.forward_func = theano.function(
        #     [in_data_theano, params_theano, biases_theano],
        #     T.sum(in_data_theano * params_theano, axis=(2, 3, 4)) + biases_theano
        # )
        # type4 = T.TensorType(dtype=theano_type, broadcastable=(False, False, True, True, True))
        # # type5 = T.TensorType(dtype=theano_type, broadcastable=(False, True, False, False, False))
        # # type5 = T.TensorType(dtype=theano_type, broadcastable=(False, False))
        # eta_theano = type4('eta_theano')
        # # in_data_theano_grad=type5('in_data_theano_grad')
        # # biases_theano_grad=type(
        # self.gradient_func = theano.function(
        #     [eta_theano, in_data_theano],
        #     T.sum(eta_theano * in_data_theano, axis=0)
        # )

    def forward(self, in_data: np.ndarray) -> np.ndarray:
        assert in_data.shape == (self.batch, self.height, self.width, self.channel)
        self.in_data = in_data

        # self.out_data = np.tensordot(
        #     in_data, self.params, axes=[(1, 2, 3), (1, 2, 3)]
        # ) + self.biases.reshape((1, self.dim))
        # self.out_data = np.multiply(
        #     in_data.reshape((self.batch, 1, self.height, self.width, self.channel)),
        #     self.params.reshape((1, self.dim, self.height, self.width, self.channel))
        # ).sum(axis=(2, 3, 4)) + self.biases.reshape((1, self.dim))
        # self.out_data = self.forward_func(
        #     in_data.reshape((self.batch, 1, self.height, self.width, self.channel)),
        #     self.params.reshape((1, self.dim, self.height, self.width, self.channel)),
        #     self.biases.reshape((1, self.dim))
        # )
        self.out_data = torch.add(
            torch.mul(
                torch.tensor(in_data.reshape((self.batch, 1, self.height, self.width, self.channel)), dtype=torch_type),
                self.params.reshape((1, self.dim, self.height, self.width, self.channel))
            ).sum(dim=(2, 3, 4)),
            self.biases.reshape((1, self.dim))
        ).numpy()

        assert self.out_data.shape == (self.batch, self.dim)
        return self.out_data

    def gradient(self, eta: np.ndarray) -> np.ndarray:
        assert eta.shape == (self.batch, self.dim)

        # 本次更新的梯度
        # todo 性能瓶颈
        # start = time.time()
        # self.grad_params = np.multiply(
        #     eta.reshape((self.batch, self.dim, 1, 1, 1)),
        #     self.in_data.reshape((self.batch, 1, self.height, self.width, self.channel))
        # ).sum(axis=0)
        # self.grad_params = self.gradient_func(
        #     eta.reshape((self.batch, self.dim, 1, 1, 1)),
        #     self.in_data.reshape((self.batch, 1, self.height, self.width, self.channel))
        # )
        self.grad_biases = torch.tensor(eta, dtype=torch_type).sum(dim=0)
        # print(time.time() - start)
        self.grad_params = torch.mul(
            torch.tensor(eta.reshape((self.batch, self.dim, 1, 1, 1)), dtype=torch_type),
            torch.tensor(self.in_data.reshape((self.batch, 1, self.height, self.width, self.channel)), dtype=torch_type)
        ).sum(dim=0)

        # 下一步传递的梯度
        # start = time.time()
        next_eta = np.tensordot(eta, self.params, axes=1)
        # print(time.time() - start)
        return next_eta

    def backward(self):
        print('====fc params', self.params.mean())
        print('====fc biases', self.biases.mean())
        print('====fc params grad', self.grad_params.mean())
        print('====fc biases grad', self.grad_biases.mean())
        self.params -= self.learning_rate * self.grad_params
        self.biases -= self.learning_rate * self.grad_biases


# class FullConnection:
#     batch: int
#     height: int
#     width: int
#     channel: int
#     dim: int
#     learning_rate: float
#
#     param: np.ndarray  # （dim, height*width*channel+1)
#     extend_in_data: np.ndarray  # 方便计算
#     out_data: np.ndarray
#
#     def __init__(self, input_shape, dim, learning_rate=0.00001) -> None:
#         self.batch, self.height, self.width, self.channel = input_shape
#         self.dim = dim
#         self.learning_rate = learning_rate
#         num = self.height * self.width * self.channel
#         self.param = np.zeros(shape=(dim, num + 1), dtype=floatX)
#         self.extend_in_data = np.empty(shape=(self.batch, num + 1), dtype=floatX)
#         self.extend_in_data[:, -1] = 1
#         self.out_data = np.empty(shape=(self.batch, self.dim), dtype=floatX)
#
#     def forward(self, in_data: np.ndarray) -> np.ndarray:
#         assert in_data.shape == (self.batch, self.height, self.width, self.channel)
#         num = self.height * self.width * self.channel
#
#         # 扩展了 1 的输入数据
#         self.extend_in_data[:, :-1] = in_data.reshape((self.batch, num))
#
#         # start = time.time()
#         self.out_data = np.multiply(
#             self.extend_in_data.reshape((self.batch, 1, num + 1)),
#             self.param.reshape((1, self.dim, num + 1))
#         ).sum(axis=2)
#         # print(time.time() - start)
#
#         return self.out_data
#
#     def gradient(self, eta: np.ndarray) -> np.ndarray:
#         assert eta.shape == (self.batch, self.dim)
#
#         # 本次更新的梯度
#         # todo 性能瓶颈
#         num = self.height * self.width * self.channel
#         # start = time.time()
#         self.grad = np.multiply(
#             eta.reshape((self.batch, self.dim, 1)),
#             self.extend_in_data.reshape((self.batch, 1, num + 1))
#         ).sum(axis=0)
#         # print(time.time() - start)
#
#         # 下一步传递的梯度
#         # start = time.time()
#         next_eta = np.dot(eta, self.param[:, :-1])
#         next_eta = next_eta.reshape((self.batch, self.height, self.width, self.channel))
#         # print(time.time() - start)
#         return next_eta
#
#     def backward(self):
#         self.param -= self.learning_rate * self.grad


class Tanh:
    batch: int
    height: int
    width: int
    channel: int
    in_data: np.ndarray
    output: np.ndarray

    def __init__(self, input_shape) -> None:
        self.batch, self.height, self.width, self.channel = input_shape

    def forward(self, in_data: np.ndarray) -> np.ndarray:
        assert in_data.shape == (self.batch, self.height, self.width, self.channel)
        self.in_data = in_data
        self.output = np.tanh(in_data) / 2 + 0.5
        return self.output

    def gradient(self, eta: np.ndarray) -> np.ndarray:
        assert eta.shape == (self.batch, self.height, self.width, self.channel)
        return eta * (1 - np.tanh(self.in_data) ** 2) / 2


class Sigmoid:
    batch: int
    height: int
    width: int
    channel: int
    in_data: np.ndarray
    output: np.ndarray

    coe = 1

    def __init__(self, input_shape) -> None:
        self.batch, self.height, self.width, self.channel = input_shape

    def forward(self, in_data: np.ndarray) -> np.ndarray:
        assert in_data.shape == (self.batch, self.height, self.width, self.channel)
        print(in_data.max())
        self.in_data = in_data.astype(dtype=np.float64)
        tmp = np.exp(-self.in_data / self.coe)
        self.output = 1 / (1 + tmp)
        # self.output = self.output.astype(floatX)
        self.output[self.output >= 0.999999999] = 0.999999999  # 截断概率
        self.output[self.output <= 0.000000001] = 0.000000001
        return self.output

    def gradient(self, eta: np.ndarray) -> np.ndarray:
        assert eta.shape == (self.batch, self.height, self.width, self.channel)
        tmp = np.exp(-self.in_data / self.coe)
        grad = (tmp / (1 + tmp) ** 2) / self.coe
        grad = grad.astype(floatX)
        return eta * grad


def tune1():
    batch = 10
    fc = FullConnection((batch, 29, 29, 60), 1000, 0.002 / batch)
    in_data = np.random.randn(batch, 29, 29, 60).astype(np.float32)
    out_true = np.random.randn(batch, 1000).astype(np.float32)
    for _ in range(1000):
        start = time.time()
        y_pred = fc.forward(in_data)
        eta = y_pred - out_true
        print('eta=', eta.sum())
        fc.gradient(eta)
        fc.backward()
        print(time.time() - start)
        print()


def tune2():
    batch = 100
    fc = FullConnection((batch, 1000, 1, 1), 1, 0.05 / batch)
    in_data = np.random.randn(batch, 1000, 1, 1).astype(np.float32)
    out_true = np.random.randn(batch, 1).astype(np.float32)
    for _ in range(1000):
        start = time.time()
        y_pred = fc.forward(in_data)
        eta = y_pred - out_true
        print('eta=', eta.sum())
        fc.gradient(eta)
        fc.backward()
        print(time.time() - start)
        print()


if __name__ == '__main__':
    tune1()
    # tune2()
