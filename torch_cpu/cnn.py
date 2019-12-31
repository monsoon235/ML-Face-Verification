from typing import List, Tuple

from sklearn.metrics import f1_score

import torch
import time

import convolution, full_connection, pooling, preprocess, other

floatX = torch.float32
torch.set_num_threads(64)

max_batch = 3000
img_size = 119
deepid_dim = 1000
cross_dim = 500

conv1_learning_rate = 2e-7 / max_batch
conv2_learning_rate = 2e-7 / max_batch
conv3_learning_rate = 2e-7 / max_batch
conv4_learning_rate = 2e-7 / max_batch
fc1_learning_rate = 2e-7 / max_batch
fc2_learning_rate = 2e-7 / max_batch
fc_cross_learning_rate = 2e-7 / max_batch
fc_softmax_learning_rate = 2e-7 / max_batch

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

        id = torch.empty((self.batch, deepid_dim), dtype=floatX)
        id[:, :deepid_dim // 2] = self.fc1.forward(out3)
        id[:, deepid_dim // 2:] = self.fc2.forward(out4)
        return id

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
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
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, 2, img_size, img_size, 3)
        id1 = self.deepid1.forward(in_data[:, 0])
        id2 = self.deepid2.forward(in_data[:, 1])
        id_cross = self.fc_cross.forward(torch.cat((id1, id2), dim=1))
        softmax = self.fc_softmax.forward(id_cross)
        return softmax

    def backward(self, eta: torch.Tensor):
        assert eta.shape == (self.batch, 2)
        eta_cross = self.fc_softmax.backward(eta)
        eta_id = self.fc_cross.backward(eta_cross)
        self.deepid1.backward(eta_id[:, :deepid_dim])
        self.deepid2.backward(eta_id[:, deepid_dim:])

    def get_labels(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.argmax(dim=1)

    def get_loss_and_dy(self, pred: torch.Tensor, true_labels: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        true_onehot = torch.zeros(pred.shape)
        for b in range(pred.shape[0]):
            true_onehot[b, true_labels[b]] = 1
        loss = -true_onehot * torch.log(pred)
        # print(pred)
        # print(true_onehot)
        # print(loss)
        dy = -true_onehot / pred
        assert not torch.isnan(dy).any()
        assert not torch.isinf(dy).any()
        return loss.sum(), dy

    def train(self, imgs: torch.Tensor, labes: torch.Tensor, iteration_limit):
        all_batch = imgs.shape[0]
        assert imgs.shape == (all_batch, 2, img_size, img_size, 3)
        assert labes.shape == (all_batch,)
        if all_batch % max_batch == 0:
            batch_num = all_batch // max_batch
        else:
            batch_num = all_batch // max_batch + 1
        print('start iteration....')
        for i in range(iteration_limit):
            print(f'---------> iteration {i}')
            for j in range(batch_num):
                print(f'\t-----> batch {j}')
                start = time.time()
                batch_data = imgs[j * max_batch:(j + 1) * max_batch]
                batch_labels = labes[j * max_batch:(j + 1) * max_batch]
                pred = self.forward(batch_data)
                loss, dy = self.get_loss_and_dy(pred, batch_labels)
                self.backward(dy)
                print(f'\tloss = {loss}, time cost = {time.time() - start} s')
                pred = self.forward(test_imgs)
                print(f'\tF1 = {f1_score(test_labels, self.get_labels(pred))}')


if __name__ == '__main__':
    imgs, labels = preprocess.get_images()

    n = imgs.shape[0]
    split = n - max_batch
    test_imgs = imgs[split:]
    test_labels = labels[split:]
    train_imgs = imgs[:split]
    train_labels = labels[:split]

    fv = FaceVerification()
    fv.train(train_imgs, train_labels, 1000)
