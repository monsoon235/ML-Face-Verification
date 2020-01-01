from typing import Tuple

from PIL import Image
import numpy as np
import torch
from torch import Tensor as T
import os
import math

floatX = torch.float32

clip_size = 119


def mirror_and_swap(imgs: T, labels: T) -> Tuple[T, ...]:
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


def confuse(imgs: T, labels: T) -> Tuple[T, ...]:
    index = np.arange(imgs.shape[0])
    np.random.shuffle(index)
    return imgs[index], labels[index]


def clip_and_align(img: Image) -> Image:
    return img.resize((119, 119), Image.ANTIALIAS)


def get_images() -> Tuple[T, ...]:
    # if os.path.exists('../saved_data/train_imgs.bin') and os.path.exists('../saved_data/train_labels.bin') \
    #         and os.path.exists('../saved_data/test_imgs.bin') and os.path.exists('../saved_data/test_labels.bin'):
    #     train_imgs = torch.load('../saved_data/train_imgs.bin')
    #     train_labels = torch.load('../saved_data/train_labels.bin')
    #     test_imgs = torch.load('../saved_data/test_imgs.bin')
    #     test_labels = torch.load('../saved_data/test_labels.bin')
    #     print('data loaded')
    #     n = train_imgs.shape[0]
    #     m = test_imgs.shape[0]
    #     assert train_imgs.shape == (n, 2, clip_size, clip_size, 3)
    #     assert test_imgs.shape == (m, 2, clip_size, clip_size, 3)
    #     assert train_labels.shape == (n,)
    #     assert test_labels.shape == (m,)
    #     return train_imgs, train_labels, test_imgs, test_labels

    imgs = torch.empty(size=(3200, 2, clip_size, clip_size, 3), dtype=floatX)
    labels = torch.empty(size=(3200,), dtype=torch.int)

    index = 0

    path = '../dataset/match pairs'
    for pair in os.listdir(path):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            img = clip_and_align(Image.open(os.path.join(path, pair, img), 'r'))
            if img is None:
                index -= 1
                break
            imgs[index, j] = torch.tensor(np.array(img, dtype=np.float32) / 255)
            labels[index] = 1
        index += 1
    path = '../dataset/mismatch pairs'
    for pair in os.listdir(path):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            img = clip_and_align(Image.open(os.path.join(path, pair, img), 'r'))
            if img is None:
                index -= 1
                break
            imgs[index, j] = torch.tensor(np.array(img, dtype=np.float32) / 255)
            labels[index] = 0
        index += 1
    # 去除劣质图片
    print(index)
    imgs = imgs[:index]
    labels = labels[:index]
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
    train_imgs, train_labels, test_imgs, test_labels = get_images()
    torch.save(train_imgs, '/home/user/yjh/train_imgs.bin')
    torch.save(train_labels, '/home/user/yjh/train_labels.bin')
    torch.save(test_imgs, '/home/user/yjh/test_imgs.bin')
    torch.save(test_labels, '/home/user/yjh/test_labels.bin')
