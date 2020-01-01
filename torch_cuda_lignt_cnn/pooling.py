import torch
import time
import other

floatX = torch.float32


# 通过测试
class Pooling:
    batch: int
    height: int
    width: int
    channel: int
    pool_size: int

    mask: torch.Tensor

    # 仅支持 stride=pool_size
    def __init__(self, input_shape, pool_size, activate_func: str = None) -> None:
        self.height, self.width, self.channel = input_shape
        self.pool_size = pool_size
        assert self.height % pool_size == 0
        assert self.width % pool_size == 0

        if activate_func == 'relu':
            self.activation = other.Relu()
        elif activate_func == 'sigmoid':
            self.activation = other.Sigmoid(100)
        elif activate_func == 'tanh':
            self.activation = other.Tanh(100)
        else:
            self.activation = None

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        assert in_data.is_cuda
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, self.height, self.width, self.channel)
        grid = in_data.reshape(
            (self.batch,
             self.height // self.pool_size, self.pool_size,
             self.width // self.pool_size, self.pool_size,
             self.channel)
        )
        max1, indices1 = grid.max(dim=2)
        max2, indices2 = max1.max(dim=3)
        output = max2
        self.mask = grid == output.reshape((self.batch,
                                            self.height // self.pool_size, 1,
                                            self.width // self.pool_size, 1,
                                            self.channel))
        if self.activation is not None:
            output = self.activation.forward(output)
        return output

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.is_cuda
        if self.activation is not None:
            eta = self.activation.backward(eta)
        assert eta.shape == (self.batch, self.height // self.pool_size, self.width // self.pool_size, self.channel)
        next_eta = torch.mul(
            eta.reshape((self.batch,
                         self.height // self.pool_size, 1,
                         self.width // self.pool_size, 1,
                         self.channel)),
            self.mask
        ).reshape((self.batch, self.height, self.width, self.channel))
        return next_eta


if __name__ == '__main__':
    b = 100
    pool = Pooling((b, 200, 200, 20), pool_size=2)
    x = torch.randn((b, 200, 200, 20))
    dy = torch.zeros((b, 100, 100, 20))
    for _ in range(50):
        pool.forward(x)
        pool.backward(dy)
    # x = torch.arange(2 * 4 * 4 * 2).reshape((2, 4, 4, 2))
    # pooling = Pooling(x.shape, pool_size=2)
    # print(x)
    # print(pooling.forward(x))
