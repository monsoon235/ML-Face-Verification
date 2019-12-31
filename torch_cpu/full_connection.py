import torch
import other

floatX = torch.float32


class FullConnection:
    batch: int
    dim_in: int
    dim_out: int
    learning_rate: float

    params: torch.Tensor
    biases: torch.Tensor
    in_data: torch.Tensor

    def __init__(self, dim_in, dim_out, learning_rate, activate_func: str = None) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.learning_rate = learning_rate
        self.params = torch.randn((dim_out, dim_in), dtype=floatX) / 500
        self.biases = torch.randn((dim_out,), dtype=floatX) / 500

        if activate_func == 'relu':
            self.activation = other.Relu()
        elif activate_func == 'sigmoid':
            self.activation = other.Sigmoid(100)
        elif activate_func == 'tanh':
            self.activation = other.Tanh(100)
        elif activate_func == 'softmax':
            self.activation = other.Softmax()
        else:
            self.activation = None

    def forward(self, in_data: torch.testing) -> torch.Tensor:
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

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.shape == (self.batch, self.dim_out)
        if self.activation is not None:
            eta = self.activation.backward(eta)
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


if __name__ == '__main__':
    b = 400
    x = torch.randn((b, 10000), dtype=floatX)
    fc1 = FullConnection(x.shape, 5000, learning_rate=5e-12)
    x1 = fc1.forward(x)
    fc2 = FullConnection(x1.shape, 1000, learning_rate=5e-12)
    x2 = fc2.forward(x1)
    fc3 = FullConnection(x2.shape, 1, learning_rate=5e-12)
    for i in range(10):
        pred = fc3.forward(fc2.forward(fc1.forward(x)))
        dy = pred
        print(f'mean dy = {dy.abs().mean()}')
        fc1.backward(fc2.backward(fc3.backward(dy)))
