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


class MFM:
    mask: torch.Tensor

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        assert in_data.is_cuda
        b, h, w, c = in_data.shape
        assert c % 2 == 0
        s = in_data.stride()
        new_stride = (s[0], s[1], s[2], s[3], s[3] * c // 2)
        data = in_data.as_strided(size=(b, h, w, c // 2, 2), stride=new_stride)
        out = data.max(dim=4)[0]
        # 检验
        # assert (out == torch.max(in_data[:, :, :, :c // 2], in_data[:, :, :, c // 2:])).all()
        self.mask = data == out.reshape((b, h, w, c // 2, 1))
        return out

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.is_cuda
        b, h, w, c2 = eta.shape
        c = c2 * 2
        s = eta.stride()
        new_stride = (s[0], s[1], s[2], s[3], 0)
        eta = eta.as_strided(size=(b, h, w, c // 2, 2), stride=new_stride)
        next_eta = eta * self.mask
        next_eta = torch.cat((next_eta[:, :, :, :, 0], next_eta[:, :, :, :, 1]), dim=3)
        assert next_eta.shape == (b, h, w, c)
        return next_eta


if __name__ == '__main__':
    mfm = MFM()
    data = torch.randn((2, 2, 2, 2), device='cuda', dtype=floatX)
    res = mfm.forward(data)
    print(res)
    eta = mfm.backward(res)
    print(eta)

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
