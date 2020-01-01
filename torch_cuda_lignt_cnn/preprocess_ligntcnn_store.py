from typing import Tuple

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import Tensor as T
import os
import math
import face_recognition

floatX = torch.float32

clip_size = 130


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
    try:
        # 识别关键点
        landmarks = face_recognition.face_landmarks(np.array(img), model='small')[0]
        print(landmarks)
        l_eye = landmarks['left_eye']
        r_eye = landmarks['right_eye']
        l_mean = np.mean(l_eye, axis=0).astype(int)
        r_mean = np.mean(r_eye, axis=0).astype(int)
        dy = r_mean[1] - l_mean[1]
        dx = r_mean[0] - l_mean[0]
        # 计算旋转角度
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 旋转
        img = img.rotate(angle, expand=True)
        # 再次识别
        landmarks = face_recognition.face_landmarks(np.array(img), model='small')[0]
        # 剪裁, 以鼻子为中心
        nose = landmarks['nose_tip']
        center = np.mean(nose, axis=0).astype(int)
        center[0] = max(center[0], clip_size // 2)
        center[0] = min(center[0], img.size[0] - clip_size // 2)
        center[1] = max(center[1], clip_size // 2)
        center[1] = min(center[1], img.size[1] - clip_size // 2)
        assert clip_size % 2 == 0
        left = center[0] - clip_size // 2
        right = center[0] + clip_size // 2
        down = center[1] + clip_size // 2
        top = center[1] - clip_size // 2
        img = img.crop((left, top, right, down))
        assert img.size == (clip_size, clip_size)
        return img
    except:
        # 部分图片过于模糊,排除
        return None


name_imgs: dict


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
    global name_imgs
    name_imgs = {}

    index = 0
    path = '../dataset/match pairs'
    for pair in os.listdir(path):
        print(index)
        for j, file in enumerate(os.listdir(os.path.join(path, pair))):
            img = clip_and_align(Image.open(os.path.join(path, pair, file)))
            if img is None:
                index -= 1
                break
            imgs[index, j] = torch.tensor(np.array(img, dtype=np.float32) / 255)
            labels[index] = 1
            name = file[:file.rfind('_')]
            name_imgs[name] = name_imgs.get(name, [])
            name_imgs[name].append(imgs[index, j])
        index += 1
    path = '../dataset/mismatch pairs'
    for pair in os.listdir(path):
        print(index)
        for j, file in enumerate(os.listdir(os.path.join(path, pair))):
            img = clip_and_align(Image.open(os.path.join(path, pair, file)))
            if img is None:
                index -= 1
                break
            imgs[index, j] = torch.tensor(np.array(img, dtype=np.float32) / 255)
            labels[index] = 0
            name = file[:file.rfind('_')]
            name_imgs[name] = name_imgs.get(name, [])
            name_imgs[name].append(imgs[index, j])
        index += 1
    # 去除劣质图片
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


def get_classifier_imgs() -> Tuple[int, T, T]:
    # 重新组织分类用数据
    all_num = sum(len(name_imgs[k]) for k in name_imgs)
    class_num = len(name_imgs)
    imgs = torch.empty(size=(all_num, clip_size, clip_size, 3), dtype=floatX)
    labels = torch.empty(size=(all_num,), dtype=torch.int)
    index = 0
    for label, name in enumerate(name_imgs):
        imgs_by_name = name_imgs[name]
        n = len(imgs_by_name)
        labels[index:index + n] = label
        for i in range(n):
            imgs[index + i] = imgs_by_name[i]
        index += n
    # 数据集扩充
    imgs = torch.cat((imgs, imgs.flip(dims=(2,))), dim=0)
    labels = torch.cat((labels, labels), dim=0)
    # 打乱
    imgs, labels = confuse(imgs, labels)
    return class_num, imgs, labels


if __name__ == '__main__':
    train_imgs, train_labels, test_imgs, test_labels = get_images()
    torch.save(train_imgs, '/home/user/yjh/fv/train_imgs.bin')
    torch.save(train_labels, '/home/user/yjh/fv/train_labels.bin')
    torch.save(test_imgs, '/home/user/yjh/fv/test_imgs.bin')
    torch.save(test_labels, '/home/user/yjh/fv/test_labels.bin')

    class_num, imgs, labels = get_classifier_imgs()
    torch.save(imgs, '/home/user/yjh/cl/imgs.bin')
    torch.save(labels, '/home/user/yjh/cl/labels.bin')
