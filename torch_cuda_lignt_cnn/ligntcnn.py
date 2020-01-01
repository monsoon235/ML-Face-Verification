import os
import time
from typing import Tuple

import torch
from torch import Tensor as T
import convolution, full_connection, other, pooling, preprocess_ligntcnn_load

floatX = torch.float32
device = 'cuda'
torch.cuda.set_device(2)
torch.set_num_threads(72)

max_batch = 40
class_num: int  # 样本类别数
learning_rate = 0.05 / max_batch


class LightCNN:
    conv1: convolution.Convolution
    conv2a: convolution.Convolution
    conv2: convolution.Convolution
    conv3a: convolution.Convolution
    conv3: convolution.Convolution
    conv4a: convolution.Convolution
    conv4: convolution.Convolution
    conv5a: convolution.Convolution
    conv5: convolution.Convolution
    pool1: pooling.Pooling
    pool2: pooling.Pooling
    pool3: pooling.Pooling
    pool4: pooling.Pooling
    fc1: full_connection.FullConnection

    def __init__(self) -> None:
        self.conv1 = convolution.Convolution((130, 130, 3), out_channel=96, kernel_size=(5, 5), stride=(1, 1),
                                             learning_rate=learning_rate, padding=(1, 1), activate_func='mfm')
        self.conv2a = convolution.Convolution((64, 64, 48), out_channel=96, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv2 = convolution.Convolution((64, 64, 48), out_channel=192, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, padding=(1, 1), activate_func='mfm')
        self.conv3a = convolution.Convolution((32, 32, 96), out_channel=192, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv3 = convolution.Convolution((32, 32, 96), out_channel=384, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, padding=(1, 1), activate_func='mfm')
        self.conv4a = convolution.Convolution((16, 16, 192), out_channel=384, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv4 = convolution.Convolution((16, 16, 192), out_channel=256, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, padding=(1, 1), activate_func='mfm')
        self.conv5a = convolution.Convolution((16, 16, 128), out_channel=256, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv5 = convolution.Convolution((16, 16, 128), out_channel=256, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, padding=(1, 1), activate_func='mfm')
        self.pool1 = pooling.Pooling((128, 128, 48), pool_size=2)
        self.pool2 = pooling.Pooling((64, 64, 96), pool_size=2)
        self.pool3 = pooling.Pooling((32, 32, 192), pool_size=2)
        self.pool4 = pooling.Pooling((16, 16, 128), pool_size=2)
        self.fc1 = full_connection.FullConnection(8 * 8 * 128, 512, learning_rate=learning_rate, activate_func='mfm')

        self.fc_out = full_connection.FullConnection(256, class_num, learning_rate=learning_rate)

    def forward(self, in_data: T) -> T:
        assert in_data.is_cuda
        pool4_out = self.pool4.forward(self.conv5.forward(self.conv5a.forward(
            self.conv4.forward(self.conv4a.forward(
                self.pool3.forward(self.conv3.forward(self.conv3a.forward(
                    self.pool2.forward(self.conv2.forward(self.conv2a.forward(
                        self.pool1.forward(self.conv1.forward(
                            in_data
                        ))
                    )))
                )))
            ))
        )))
        assert pool4_out.shape[1:] == (8, 8, 128)
        out = self.fc1.forward(pool4_out.flatten(start_dim=1, end_dim=-1))
        return out

    def backward(self, eta: T):
        assert eta.is_cuda
        eta_fc1 = self.fc1.backward(eta)
        b = eta_fc1.shape[0]
        assert eta_fc1.shape == (b, 8 * 8 * 128)
        self.conv1.backward(self.pool1.backward(
            self.conv2a.backward(self.conv2.backward(self.pool2.backward(
                self.conv3a.backward(self.conv3.backward(self.pool3.backward(
                    self.conv4a.backward(self.conv4.backward(
                        self.conv5a.backward(self.conv5.backward(self.pool4.backward(
                            eta_fc1.reshape((b, 8, 8, 128))
                        )))
                    ))
                )))
            )))
        ))

    fc_out: full_connection.FullConnection

    def forward_classifier(self, in_data: T) -> T:
        assert in_data.is_cuda
        return self.fc_out.forward(self.forward(in_data))

    def backward_classifier(self, eta: T):
        assert eta.is_cuda
        return self.backward(self.fc_out.backward(eta))

    def get_labels(self, proba: T) -> T:
        return proba.argmax(dim=1)

    # def get_pred_proba(self, imgs: T) -> T:
    #     n = imgs.shape[0]
    #     if n % max_batch == 0:
    #         batch_num = n // max_batch
    #     else:
    #         batch_num = n // max_batch + 1
    #     all_pred_proba = torch.empty((n, class_num), dtype=floatX)
    #     for b in range(batch_num):
    #         start = b * max_batch
    #         end = (b + 1) * max_batch
    #         batch_imgs = imgs[start:end]
    #         out = self.forward_classifier(batch_imgs.cuda())
    #         batch_pred_proba = self.softmax(out)
    #         all_pred_proba[start:end] = batch_pred_proba
    #     return all_pred_proba

    def loss(self, true_labels: T, pred_proba: T) -> float:
        loss = 0
        for i, l in enumerate(true_labels):
            d = -pred_proba[i, l].log()
            if torch.isinf(d):
                d = 100
            loss += d
        return loss

    def softmax(self, in_data: T) -> T:
        assert in_data.is_cuda
        in_data = in_data - in_data.max(dim=1)[0].reshape((in_data.shape[0], 1))
        e = torch.exp(in_data)
        return e / e.sum(dim=1).reshape((in_data.shape[0], 1))

    def softmax_eta(self, true_labels: T, pred_proba: T) -> T:
        eta = pred_proba.clone()
        for i, l in enumerate(true_labels):
            eta[i, l] -= 1
        return eta

    def train_for_classifier(self, train_imgs: T, train_labels: T, iteration_limit):
        n = train_imgs.shape[0]
        if n % max_batch == 0:
            batch_num = n // max_batch
        else:
            batch_num = n // max_batch + 1

        all_pred_proba = torch.empty((n, class_num), dtype=floatX)
        for i in range(iteration_limit):
            print(f'========> iteration {i}')
            for b in range(batch_num):
                start = b * max_batch
                end = (b + 1) * max_batch
                batch_imgs = train_imgs[start:end]
                batch_labels = train_labels[start:end]

                time_start = time.time()

                out = self.forward_classifier(batch_imgs.cuda())
                batch_pred_proba = self.softmax(out)
                all_pred_proba[start:end] = batch_pred_proba

                loss = self.loss(batch_labels, batch_pred_proba)
                eta = self.softmax_eta(batch_labels, batch_pred_proba)
                print('eta mean =', eta.abs().mean())
                self.backward_classifier(eta.cuda())

                time_end = time.time()

                print(f'\tbatch {b}, loss = {loss}, time = {time_end - time_start}')

            # 无测试集
            print(f'iteration correctness = {(train_labels == self.get_labels(all_pred_proba)).type(floatX).mean()}')

    def save(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.conv1.save(os.path.join(folder_path, 'conv1'))
        self.conv2a.save(os.path.join(folder_path, 'conv2a'))
        self.conv2.save(os.path.join(folder_path, 'conv2'))
        self.conv3a.save(os.path.join(folder_path, 'conv3a'))
        self.conv3.save(os.path.join(folder_path, 'conv3'))
        self.conv4a.save(os.path.join(folder_path, 'conv4a'))
        self.conv4.save(os.path.join(folder_path, 'conv4'))
        self.conv5a.save(os.path.join(folder_path, 'conv5a'))
        self.conv5.save(os.path.join(folder_path, 'conv5'))
        self.fc1.save(os.path.join(folder_path, 'fc1'))
        self.fc_out.save(os.path.join(folder_path, 'fc_out'))

    def load(self, folder_path: str):
        self.conv1.load(os.path.join(folder_path, 'conv1'))
        self.conv2a.load(os.path.join(folder_path, 'conv2a'))
        self.conv2.load(os.path.join(folder_path, 'conv2'))
        self.conv3a.load(os.path.join(folder_path, 'conv3a'))
        self.conv3.load(os.path.join(folder_path, 'conv3'))
        self.conv4a.load(os.path.join(folder_path, 'conv4a'))
        self.conv4.load(os.path.join(folder_path, 'conv4'))
        self.conv5a.load(os.path.join(folder_path, 'conv5a'))
        self.conv5.load(os.path.join(folder_path, 'conv5'))
        self.fc1.load(os.path.join(folder_path, 'fc1'))
        self.fc_out.load(os.path.join(folder_path, 'fc_out'))


if __name__ == '__main__':
    # 训练多分类
    class_num, imgs, labels = preprocess_ligntcnn_load.get_classifier_imgs()
    lightcnn = LightCNN()
    lightcnn.train_for_classifier(imgs, labels, 300)
    lightcnn.save('../saved_lightcnn')
