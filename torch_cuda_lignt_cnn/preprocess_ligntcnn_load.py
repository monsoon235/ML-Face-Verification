from typing import Tuple

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import Tensor as T
import os
import math

floatX = torch.float32

clip_size = 130


def get_images() -> Tuple[T, ...]:
    train_imgs = torch.load('../saved_data/fv/train_imgs.bin')
    train_labels = torch.load('../saved_data/fv/train_labels.bin')
    test_imgs = torch.load('../saved_data/fv/test_imgs.bin')
    test_labels = torch.load('../saved_data/fv/test_labels.bin')
    print('fv data loaded')
    n = train_imgs.shape[0]
    m = test_imgs.shape[0]
    assert train_imgs.shape == (n, 2, clip_size, clip_size, 3)
    assert test_imgs.shape == (m, 2, clip_size, clip_size, 3)
    assert train_labels.shape == (n,)
    assert test_labels.shape == (m,)
    return train_imgs, train_labels, test_imgs, test_labels


def get_classifier_imgs() -> Tuple[int, T, T]:
    imgs = torch.load('../lightcnn_data/cl/imgs.bin')
    labels: T = torch.load('../lightcnn_data/cl/labels.bin')
    print('cl data loaded')
    n = imgs.shape[0]
    assert imgs.shape == (n, clip_size, clip_size, 3)
    assert labels.shape == (n,)
    class_num = labels.unique().shape[0]
    return class_num, imgs, labels
