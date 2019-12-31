from PIL import Image
import numpy as np
import os

floatX = np.float32


def get_img():
    # 返回 (3200,2,247,247,3) 的数据
    # 和 (3200,) 的 label
    data = np.empty(shape=(3200, 2, 247, 247, 3), dtype=floatX)
    label = np.empty(shape=(3200,), dtype=int)
    path = '../dataset/match pairs'
    for i, pair in enumerate(os.listdir(path)):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            label[i] = 1
            image_data = Image.open(os.path.join(path, pair, img), 'r')
            data[i, j] = np.array(image_data)[1:248, 1:248, :] / 255
            # print(data[i, j])
    path = '../dataset/mismatch pairs'
    for i, pair in enumerate(os.listdir(path)):
        for j, img in enumerate(os.listdir(os.path.join(path, pair))):
            label[i + 1600] = 0
            image_data = Image.open(os.path.join(path, pair, img), 'r')
            data[i + 1600, j] = np.array(image_data)[1:248, 1:248, :] / 255
            # print(data[i, j])
    # 打乱顺序
    index = np.arange(3200)
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label


if __name__ == '__main__':
    get_img()
