from typing import List, Tuple

from sklearn.metrics import f1_score

import torch
import time
import os

import convolution, full_connection, pooling, preprocess, other

floatX = torch.float32
device = 'cuda'
torch.set_num_threads(72)

max_batch = 200
img_size = 119
deepid_dim = 1000
cross_dim = 500

# 1e-3 发散
# 2e-4 发散

# 1e-5 迭代 300 即可

conv1_learning_rate = 1e-5 / max_batch
conv2_learning_rate = 1e-5 / max_batch
conv3_learning_rate = 1e-5 / max_batch
conv4_learning_rate = 1e-5 / max_batch
fc1_learning_rate = 1e-5 / max_batch
fc2_learning_rate = 1e-5 / max_batch
fc_cross_learning_rate = 1e-5 / max_batch
fc_softmax_learning_rate = 1e-5 / max_batch

ks1 = 4
ks2 = 3
ks3 = 3
ks4 = 2

ps1 = 2
ps2 = 2
ps3 = 2

ci = 3
oc1 = 20
oc2 = 40
oc3 = 60
oc4 = 80

stride1 = 1
stride2 = 1
stride3 = 1
stride4 = 1

test_imgs: torch.Tensor
test_labels: torch.Tensor


class DeepID:
    batch: int
    conv1: convolution.Convolution
    conv2: convolution.Convolution
    conv3: convolution.Convolution
    conv4: convolution.Convolution
    pool1: pooling.Pooling
    pool2: pooling.Pooling
    pool3: pooling.Pooling
    fc1: full_connection.FullConnection
    fc2: full_connection.FullConnection

    def __init__(self) -> None:
        size = img_size
        self.conv1 = convolution.Convolution(input_shape=(size, size, ci), out_channel=oc1,
                                             kernel_size=(ks1, ks1), stride=(stride1, stride1),
                                             learning_rate=conv1_learning_rate, activate_func='relu')
        assert (size - ks1) % stride1 == 0
        size = (size - ks1) // stride1 + 1
        self.pool1 = pooling.Pooling(input_shape=(size, size, oc1), pool_size=ps1)
        assert size % 2 == 0
        size //= 2
        self.conv2 = convolution.Convolution(input_shape=(size, size, oc1), out_channel=oc2,
                                             kernel_size=(ks2, ks2), stride=(stride2, stride2),
                                             learning_rate=conv2_learning_rate, activate_func='relu')
        assert (size - ks2) % stride2 == 0
        size = (size - ks2) // stride2 + 1
        self.pool2 = pooling.Pooling(input_shape=(size, size, oc2), pool_size=ps2)
        assert size % 2 == 0
        size //= 2
        self.conv3 = convolution.Convolution(input_shape=(size, size, oc2), out_channel=oc3,
                                             kernel_size=(ks3, ks3), stride=(stride3, stride3),
                                             learning_rate=conv3_learning_rate, activate_func='relu')
        assert (size - ks3) % stride3 == 0
        size = (size - ks3) // stride3 + 1
        self.pool3 = pooling.Pooling(input_shape=(size, size, oc3), pool_size=ps3)
        assert size % 2 == 0
        size //= 2
        self.conv4 = convolution.Convolution(input_shape=(size, size, oc3), out_channel=oc4,
                                             kernel_size=(ks4, ks4), stride=(stride4, stride4),
                                             learning_rate=conv4_learning_rate, activate_func='relu')
        assert (size - ks4) % stride4 == 0
        size4 = (size - ks4) // stride4 + 1
        self.fc1 = full_connection.FullConnection(dim_in=size * size * oc3,
                                                  dim_out=deepid_dim // 2, learning_rate=fc1_learning_rate,
                                                  activate_func='relu')
        self.fc2 = full_connection.FullConnection(dim_in=size4 * size4 * oc4,
                                                  dim_out=deepid_dim // 2, learning_rate=fc2_learning_rate,
                                                  activate_func='relu')

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        assert in_data.is_cuda
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, img_size, img_size, 3)
        out3 = self.pool3.forward(self.conv3.forward(
            self.pool2.forward(self.conv2.forward(
                self.pool1.forward(self.conv1.forward(
                    in_data
                ))))))
        out4 = self.conv4.forward(out3)
        out3 = out3.flatten(start_dim=1, end_dim=-1)
        out4 = out4.flatten(start_dim=1, end_dim=-1)

        id = torch.empty((self.batch, deepid_dim), dtype=floatX, device=device)
        id[:, :deepid_dim // 2] = self.fc1.forward(out3)
        id[:, deepid_dim // 2:] = self.fc2.forward(out4)
        return id

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.is_cuda
        assert eta.shape == (self.batch, deepid_dim)
        shape3 = (self.pool3.batch, self.pool3.height // self.pool3.pool_size,
                  self.pool3.width // self.pool3.pool_size, oc3)
        shape4 = (self.conv4.batch, self.conv4.out_h, self.conv4.out_w, self.conv4.out_channel)
        eta3 = self.fc1.backward(eta[:, :deepid_dim // 2]).reshape(shape3)
        eta4 = self.fc2.backward(eta[:, deepid_dim // 2:])
        eta4 = self.conv4.backward(eta4.reshape(shape4))
        next_eta = self.conv1.backward(self.pool1.backward(
            self.conv2.backward(self.pool2.backward(
                self.conv3.backward(self.pool3.backward(
                    eta3 + eta4
                ))))))
        return next_eta

    def save(self, folder_path: str):
        self.conv1.save(os.path.join(folder_path, 'conv1'))
        self.conv2.save(os.path.join(folder_path, 'conv2'))
        self.conv3.save(os.path.join(folder_path, 'conv3'))
        self.conv4.save(os.path.join(folder_path, 'conv4'))
        self.pool1.save(os.path.join(folder_path, 'pool1'))
        self.pool2.save(os.path.join(folder_path, 'pool2'))
        self.pool3.save(os.path.join(folder_path, 'pool3'))
        self.fc1.save(os.path.join(folder_path, 'fc1'))
        self.fc2.save(os.path.join(folder_path, 'fc2'))

    def load(self, folder_path: str):
        self.conv1.load(os.path.join(folder_path, 'conv1'))
        self.conv2.load(os.path.join(folder_path, 'conv2'))
        self.conv3.load(os.path.join(folder_path, 'conv3'))
        self.conv4.load(os.path.join(folder_path, 'conv4'))
        self.pool1.load(os.path.join(folder_path, 'pool1'))
        self.pool2.load(os.path.join(folder_path, 'pool2'))
        self.pool3.load(os.path.join(folder_path, 'pool3'))
        self.fc1.load(os.path.join(folder_path, 'fc1'))
        self.fc2.load(os.path.join(folder_path, 'fc2'))


class FaceVerification:
    batch: int
    deepid1: DeepID
    deepid2: DeepID
    fc_cross: full_connection.FullConnection
    fc_softmax: full_connection.FullConnection

    def __init__(self) -> None:
        self.deepid1 = DeepID()
        self.deepid2 = DeepID()
        self.fc_cross = full_connection.FullConnection(dim_in=deepid_dim * 2, dim_out=cross_dim,
                                                       learning_rate=fc_cross_learning_rate, activate_func='relu')
        self.fc_softmax = full_connection.FullConnection(dim_in=cross_dim, dim_out=2,
                                                         learning_rate=fc_softmax_learning_rate,
                                                         activate_func='softmax')

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        assert in_data.is_cuda
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, 2, img_size, img_size, 3)
        id1 = self.deepid1.forward(in_data[:, 0])
        id2 = self.deepid2.forward(in_data[:, 1])
        id_cross = self.fc_cross.forward(torch.cat((id1, id2), dim=1))
        softmax = self.fc_softmax.forward(id_cross)
        return softmax

    def backward(self, eta: torch.Tensor):
        assert eta.is_cuda
        assert eta.shape == (self.batch, 2)
        eta_cross = self.fc_softmax.backward(eta)
        eta_id = self.fc_cross.backward(eta_cross)
        self.deepid1.backward(eta_id[:, :deepid_dim])
        self.deepid2.backward(eta_id[:, deepid_dim:])

    def get_labels(self, pred_proba: torch.Tensor) -> torch.Tensor:
        return pred_proba.argmax(dim=1)

    def get_loss_and_eta(self, pred_proba: torch.Tensor, true_labels: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # TODO check
        true_onehot = torch.zeros(pred_proba.shape)
        for b in range(pred_proba.shape[0]):
            true_onehot[b, true_labels[b]] = 1
        loss = -true_onehot * torch.log(pred_proba)
        eta = -true_onehot / pred_proba
        assert not torch.isnan(eta).any()
        assert not torch.isinf(eta).any()
        return loss.sum(), eta

    def get_pred_proba(self, imgs: torch.Tensor) -> torch.Tensor:
        all_batch = imgs.shape[0]
        assert imgs.shape == (all_batch, 2, img_size, img_size, 3)
        if all_batch % max_batch == 0:
            batch_num = all_batch // max_batch
        else:
            batch_num = all_batch // max_batch + 1
        pred_proba = torch.empty((all_batch, 2))
        for b in range(batch_num):
            batch_imgs = imgs[b * max_batch:(b + 1) * max_batch].cuda()
            pred_proba[b * max_batch:(b + 1) * max_batch] = self.forward(batch_imgs).cpu()
        return pred_proba

    def train(self, imgs: torch.Tensor, labes: torch.Tensor, iteration_limit):
        n = imgs.shape[0]
        assert imgs.shape == (n, 2, img_size, img_size, 3)
        assert labes.shape == (n,)
        if n % max_batch == 0:
            batch_num = n // max_batch
        else:
            batch_num = n // max_batch + 1

        train_pred_proba_all = torch.empty((n, 2))  # 分 batch 迭代时所有输出的训练集 proba
        for i in range(iteration_limit):
            print(f'========> iteration {i}')
            time_all = 0
            for j in range(batch_num):
                print(f'\t====> batch {j}')
                start = time.time()
                index_start = j * max_batch
                index_end = (j + 1) * max_batch
                batch_imgs = imgs[index_start:index_end].cuda()
                batch_labels = labes[index_start:index_end]

                train_pred_proba_all[index_start:index_end] = self.forward(batch_imgs).cpu()  # 前向传播
                loss, eta = self.get_loss_and_eta(train_pred_proba_all[index_start:index_end], batch_labels)
                self.backward(eta.cuda())  # 反向传播

                time_cost = time.time() - start
                time_all += time_cost
                print(f'\tbatch loss = {loss},\ttime = {time_cost} s')

            loss_all, _ = self.get_loss_and_eta(train_pred_proba_all, train_labels)  # 所有训练集的 loss
            test_pred_proba = self.get_pred_proba(test_imgs)
            f1_all = f1_score(test_labels, self.get_labels(test_pred_proba))  # 测试集的 F1
            print(f'iteration loss = {loss_all},\ttime = {time_all}')
            print(f'iteration F1 = {f1_all}')
            print()

    def save(self, folder_path):
        self.deepid1.save(os.path.join(folder_path, 'deepid1'))
        self.deepid2.save(os.path.join(folder_path, 'deepid2'))
        self.fc_cross.save(os.path.join(folder_path, 'fc_cross'))
        self.fc_softmax.save(os.path.join(folder_path, 'fc_softmax'))

    def load(self, folder_path):
        self.deepid1.load(os.path.join(folder_path, 'deepid1'))
        self.deepid2.load(os.path.join(folder_path, 'deepid2'))
        self.fc_cross.load(os.path.join(folder_path, 'fc_cross'))
        self.fc_softmax.load(os.path.join(folder_path, 'fc_softmax'))


if __name__ == '__main__':
    train_imgs, train_labels, test_imgs, test_labels = preprocess.get_images()
    fv = FaceVerification()
    fv.train(train_imgs, train_labels, 300)
    fv.save('../saved_model_0.7')
