from typing import Tuple

from PIL import Image
import numpy as np
import torch
import os
import gc
import glob

floatX = torch.float32

img_size = 119


def clip(images: torch.Tensor) -> torch.Tensor:
    return images[:, :, 3:119 + 3, 3:119 + 3, :]


def get_images() -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.empty(size=(3200, 2, 125, 125, 3), dtype=floatX)
    labels = torch.tensor(data=(1,) * 1600 + (0,) * 1600, dtype=torch.int)

    path = '../dataset/match pairs'
    for i, pair in enumerate(os.listdir(path)):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            image_data = Image.open(os.path.join(path, pair, img), 'r').resize((125, 125), Image.ANTIALIAS)
            images[i, j] = torch.tensor(np.array(image_data, dtype=np.float32) / 255)
    path = '../dataset/mismatch pairs'
    for i, pair in enumerate(os.listdir(path)):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            image_data = Image.open(os.path.join(path, pair, img), 'r').resize((125, 125), Image.ANTIALIAS)
            images[i + 1600, j] = torch.tensor(np.array(image_data, dtype=np.float32) / 255)

    # 图片镜像, 数据集x4
    f1 = images.clone()
    f1[:, 0] = f1[:, 0].flip(dims=(2,))
    f2 = images.clone()
    f2[:, 1] = f2[:, 1].flip(dims=(2,))
    f12 = images.flip(dims=(3,))
    images = torch.cat((images, f1, f2, f12), dim=0)
    labels = torch.cat((labels,) * 4, dim=0)

    # 交换图片顺序, 数据集x2
    swap = torch.empty_like(images)
    swap[:, 0] = images[:, 1]
    swap[:, 1] = images[:, 0]
    images = torch.cat((images, swap), dim=0)
    labels = torch.cat((labels,) * 2, dim=0)

    print(images.shape)

    # 剪裁
    images = clip(images)

    # 打乱顺序
    index = np.arange(images.shape[0])
    np.random.shuffle(index)
    images = images[index]
    labels = labels[index]

    # 交换图片顺序, 图片进行镜像等操作, 数据集变为原来 8 倍

    gc.collect()
    return images, labels


if __name__ == '__main__':
    get_images()
