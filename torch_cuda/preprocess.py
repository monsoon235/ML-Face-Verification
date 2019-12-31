from typing import Tuple

from PIL import Image
import numpy as np
import torch
import os

floatX = torch.float32

img_size = 119


def mirror_and_swap(imgs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    # 图片镜像, 数据集x4
    f1 = imgs.clone()
    f1[:, 0] = f1[:, 0].flip(dims=(2,))
    f2 = imgs.clone()
    f2[:, 1] = f2[:, 1].flip(dims=(2,))
    f12 = imgs.flip(dims=(3,))
    imgs = torch.cat((imgs, f1, f2, f12), dim=0)
    labels = torch.cat((labels,) * 4, dim=0)

    # 交换图片顺序, 数据集x2
    swap = torch.empty_like(imgs)
    swap[:, 0] = imgs[:, 1]
    swap[:, 1] = imgs[:, 0]
    imgs = torch.cat((imgs, swap), dim=0)
    labels = torch.cat((labels,) * 2, dim=0)

    return imgs, labels


def confuse(imgs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    index = np.arange(imgs.shape[0])
    np.random.shuffle(index)
    return imgs[index], labels[index]


def get_images() -> Tuple[torch.Tensor, ...]:
    imgs = torch.empty(size=(3200, 2, 125, 125, 3), dtype=floatX)
    labels = torch.tensor(data=(1,) * 1600 + (0,) * 1600, dtype=torch.int)

    path = '../dataset/match pairs'
    for i, pair in enumerate(os.listdir(path)):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            image_data = Image.open(os.path.join(path, pair, img), 'r').resize((125, 125), Image.ANTIALIAS)
            imgs[i, j] = torch.tensor(np.array(image_data, dtype=np.float32) / 255)
    path = '../dataset/mismatch pairs'
    for i, pair in enumerate(os.listdir(path)):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            image_data = Image.open(os.path.join(path, pair, img), 'r').resize((125, 125), Image.ANTIALIAS)
            imgs[i + 1600, j] = torch.tensor(np.array(image_data, dtype=np.float32) / 255)

    # 剪裁
    imgs = imgs[:, :, 3:119 + 3, 3:119 + 3, :]
    # 打乱
    imgs, labels = confuse(imgs, labels)
    # 划分训练集与测试集
    n = imgs.shape[0]
    split = 4 * n // 5
    train_imgs = imgs[:split]
    train_labels = labels[:split]
    test_imgs = imgs[split:]
    test_labels = labels[split:]
    # 数据集扩充
    train_imgs, train_labels = mirror_and_swap(train_imgs, train_labels)
    test_imgs, test_labels = mirror_and_swap(test_imgs, test_labels)
    # 再次打乱
    train_imgs, train_labels = confuse(train_imgs, train_labels)
    test_imgs, test_labels = confuse(test_imgs, test_labels)

    return train_imgs, train_labels, test_imgs, test_labels


if __name__ == '__main__':
    get_images()
