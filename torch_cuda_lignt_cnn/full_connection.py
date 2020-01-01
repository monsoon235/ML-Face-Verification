import math

import torch
from torch import Tensor as T
import numpy as np
import other
import os

floatX = torch.float32
device = 'cuda'


class FullConnection:
    batch: int
    dim_in: int
    dim_out: int
    learning_rate: float

    params: T
    biases: T
    in_data: T

    def __init__(self, dim_in, dim_out, learning_rate, activate_func: str = None) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.learning_rate = learning_rate
        # xaiver
        coe = math.sqrt(6) / math.sqrt(dim_in + dim_out)
        print('fc coe', coe)
        self.params = torch.tensor(
            np.random.uniform(-coe, coe, (dim_out, dim_in)),
            dtype=floatX, device=device
        )
        self.biases = torch.tensor(
            np.random.uniform(-coe, coe, (dim_out,)),
            dtype=floatX, device=device
        )
        # self.params = torch.randn((dim_out, dim_in), dtype=floatX, device=device)
        # self.biases = torch.randn((dim_out,), dtype=floatX, device=device)

        if activate_func == 'relu':
            self.activation = other.Relu()
        elif activate_func == 'mfm':
            self.activation = other.MFM_fc()
        elif activate_func == 'softmax':
            assert False
        else:
            self.activation = None

    def forward(self, in_data: torch.testing) -> T:
        assert in_data.is_cuda
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, self.dim_in)
        self.in_data = in_data
        out_data = torch.add(
            torch.mul(
                in_data.reshape((self.batch, 1, self.dim_in)),
                self.params.reshape((1, self.dim_out, self.dim_in))
            ).sum(dim=2),
            self.biases.reshape((1, self.dim_out))
        )
        assert out_data.shape == (self.batch, self.dim_out)
        if self.activation is not None:
            out_data = self.activation.forward(out_data)
        return out_data

    def backward(self, eta: T) -> T:
        assert eta.is_cuda
        if self.activation is not None:
            eta = self.activation.backward(eta)
        assert eta.shape == (self.batch, self.dim_out)
        # 本次更新的梯度
        grad_biases = eta.sum(dim=0)
        grad_params = torch.mul(
            eta.reshape((self.batch, self.dim_out, 1)),
            self.in_data.reshape((self.batch, 1, self.dim_in))
        ).sum(dim=0)
        # 下一步传递的梯度
        next_eta = torch.tensordot(eta, self.params, dims=1)
        assert next_eta.shape == (self.batch, self.dim_in)
        self.params -= self.learning_rate * grad_params
        self.biases -= self.learning_rate * grad_biases
        return next_eta

    def save(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.params, os.path.join(folder_path, 'params.bin'))
        torch.save(self.biases, os.path.join(folder_path, 'biases.bin'))

    def load(self, folder_path: str):
        self.params = torch.load(os.path.join(folder_path, 'params.bin'))
        self.biases = torch.load(os.path.join(folder_path, 'biases.bin'))


if __name__ == '__main__':
    b = 400
    x = torch.randn((b, 1000), dtype=floatX)
    fc1 = FullConnection(1000, 500, learning_rate=5e-12)
    x1 = fc1.forward(x)
    fc2 = FullConnection(500, 100, learning_rate=5e-12)
    x2 = fc2.forward(x1)
    fc3 = FullConnection(100, 1, learning_rate=5e-12)
    for i in range(10):
        pred = fc3.forward(fc2.forward(fc1.forward(x)))
        dy = pred
        print(f'mean dy = {dy.abs().mean()}')
        fc1.backward(fc2.backward(fc3.backward(dy)))
