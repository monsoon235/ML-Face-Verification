import torch
import numpy as np
from PIL import Image
import time

print(torch.cuda.is_available())

x = torch.empty((10000, 1000, 1), dtype=torch.float32)
y = torch.empty((1, 1000, 1000), dtype=torch.float32)

print(x.stride())
print(np.empty((10000, 1000, 1), dtype=np.float32).strides)

start = time.time()
(x * y).sum(dim=1)
print(time.time() - start)

a = np.empty((10000, 1000, 1), dtype=np.float32)
b = np.empty((1, 1000, 1000), dtype=np.float32)

torch.tensor

start = time.time()
(a * b).sum(axis=1)
print(time.time() - start)

if __name__ == '__main__':
    pass
