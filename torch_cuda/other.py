import torch

floatX = torch.float32


class Relu:
    mask: torch.Tensor

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        assert in_data.is_cuda
        self.mask = in_data > 0
        return in_data * self.mask

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.is_cuda
        return eta * self.mask


class Softmax:
    batch: int
    # x: torch.Tensor
    y: torch.Tensor

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        assert in_data.is_cuda
        assert len(in_data.shape) == 2
        self.batch = in_data.shape[0]
        # self.x = in_data
        e = torch.exp(in_data - in_data.max(dim=1, keepdim=True)[0])
        base = e.sum(dim=1, keepdim=True)
        self.y = e / base
        return self.y

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.is_cuda
        n = eta.shape[1]
        assert eta.shape == (self.batch, n)
        cross = -self.y.reshape((self.batch, n, 1)) * self.y.reshape((self.batch, 1, n))
        for b in range(self.batch):
            cross[b] += torch.diag(self.y[b]).cuda()
        next_eta = torch.empty_like(eta)
        for b in range(self.batch):
            next_eta[b] = (cross[b] @ eta[b].reshape((n, 1))).reshape((n,))
        return next_eta

# class Sigmoid:
#     out_data: torch.Tensor
#     coe: float
#
#     def __init__(self, coe) -> None:
#         assert False
#         self.coe = coe
#
#     def forward(self, in_data: torch.Tensor) -> torch.Tensor:
#         self.out_data = 1 / (1 + torch.exp(-in_data / self.coe))
#         return self.out_data
#
#     def backward(self, eta: torch.Tensor) -> torch.Tensor:
#         return eta * self.out_data * (1 - self.out_data) / self.coe
#
#
# class Tanh:
#     out_data: torch.Tensor
#     coe: float
#
#     def __init__(self, coe) -> None:
#         assert False
#         self.coe = coe
#
#     def forward(self, in_data: torch.Tensor) -> torch.Tensor:
#         self.out_data = torch.tanh(in_data)
#         return self.out_data
#
#     def backward(self, eta: torch.Tensor) -> torch.Tensor:
#         return eta * (1 - self.out_data ** 2) / self.coe
