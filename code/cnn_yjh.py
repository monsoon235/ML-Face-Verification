import yjh, cnn, losses, preprocess
import numpy as np
import time

from sklearn.metrics import f1_score

# import theano.tensor as T
# import theano

floatX = np.float32

batch_size = 400

learning_rate_conv1 = 0.01 / batch_size
learning_rate_conv2 = 0.01 / batch_size
learning_rate_conv3 = 0.01 / batch_size
learning_rate_fc = 1 / batch_size
learning_rate_fc_output = 1 / batch_size

face_size = 247

feature_vector_dim = 500


class CNN:
    conv11: cnn.Convolution
    conv12: cnn.Convolution
    conv13: cnn.Convolution
    conv21: cnn.Convolution
    conv22: cnn.Convolution
    conv23: cnn.Convolution

    relu11: cnn.Relu
    relu12: cnn.Relu
    relu13: cnn.Relu
    relu21: cnn.Relu
    relu22: cnn.Relu
    relu23: cnn.Relu

    max11: cnn.Pooling
    max12: cnn.Pooling
    max13: cnn.Pooling
    max21: cnn.Pooling
    max22: cnn.Pooling
    max23: cnn.Pooling

    fc1: yjh.FullConnection
    fc2: yjh.FullConnection

    relu_fc1: cnn.Relu
    relu_fc2: cnn.Relu

    fc_last: yjh.FullConnection

    output_tanh: yjh.Tanh
    output_sigmoid: yjh.Sigmoid

    batch_size: int

    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        conv1_input = (batch_size, face_size, face_size, 3)
        assert (face_size - 3) % 2 == 0
        max1_input = (batch_size, face_size - 3, face_size - 3, 20)

        conv2_input = (batch_size, (face_size - 3) // 2, (face_size - 3) // 2, 20)
        assert ((face_size - 3) // 2 - 2) % 2 == 0
        max2_input = (batch_size, (face_size - 3) // 2 - 2, (face_size - 3) // 2 - 2, 40)

        conv3_input = (batch_size, ((face_size - 3) // 2 - 2) // 2, ((face_size - 3) // 2 - 2) // 2, 40)
        assert (((face_size - 3) // 2 - 2) // 2 - 2) % 2 == 0
        max3_input = (batch_size, ((face_size - 3) // 2 - 2) // 2 - 2, ((face_size - 3) // 2 - 2) // 2 - 2, 60)

        fc_input = (
            batch_size, (((face_size - 3) // 2 - 2) // 2 - 2) // 2, (((face_size - 3) // 2 - 2) // 2 - 2) // 2, 60)

        self.conv11 = cnn.Convolution(input_shape=conv1_input, out_channels=20, kernel_size=4, stride=1,
                                      learning_rate=learning_rate_conv1)
        self.conv12 = cnn.Convolution(input_shape=conv2_input, out_channels=40, kernel_size=3, stride=1,
                                      learning_rate=learning_rate_conv2)
        self.conv13 = cnn.Convolution(input_shape=conv3_input, out_channels=60, kernel_size=3, stride=1,
                                      learning_rate=learning_rate_conv3)
        self.conv21 = cnn.Convolution(input_shape=conv1_input, out_channels=20, kernel_size=4, stride=1,
                                      learning_rate=learning_rate_conv1)
        self.conv22 = cnn.Convolution(input_shape=conv2_input, out_channels=40, kernel_size=3, stride=1,
                                      learning_rate=learning_rate_conv2)
        self.conv23 = cnn.Convolution(input_shape=conv3_input, out_channels=60, kernel_size=3, stride=1,
                                      learning_rate=learning_rate_conv3)

        self.relu11 = cnn.Relu(max1_input)
        self.relu12 = cnn.Relu(max2_input)
        self.relu13 = cnn.Relu(max3_input)
        self.relu21 = cnn.Relu(max1_input)
        self.relu22 = cnn.Relu(max2_input)
        self.relu23 = cnn.Relu(max3_input)

        self.max11 = cnn.Pooling(input_shape=max1_input, pool_size=2, stride=2)
        self.max12 = cnn.Pooling(input_shape=max2_input, pool_size=2, stride=2)
        self.max13 = cnn.Pooling(input_shape=max3_input, pool_size=2, stride=2)
        self.max21 = cnn.Pooling(input_shape=max1_input, pool_size=2, stride=2)
        self.max22 = cnn.Pooling(input_shape=max2_input, pool_size=2, stride=2)
        self.max23 = cnn.Pooling(input_shape=max3_input, pool_size=2, stride=2)

        self.fc1 = yjh.FullConnection(input_shape=fc_input, dim=feature_vector_dim, learning_rate=learning_rate_fc)
        self.fc2 = yjh.FullConnection(input_shape=fc_input, dim=feature_vector_dim, learning_rate=learning_rate_fc)

        self.relu_fc1 = cnn.Relu(fc_input)
        self.relu_fc2 = cnn.Relu(fc_input)

        self.fc_last = yjh.FullConnection(input_shape=(batch_size, feature_vector_dim, 1, 1), dim=1,
                                          learning_rate=learning_rate_fc_output)
        # self.output_tanh = yjh.Tanh((batch_size, 1, 1, 1))
        self.output_sigmoid = yjh.Sigmoid((batch_size, 1, 1, 1))

    def forward(self, data: np.ndarray) -> np.ndarray:
        data1 = data[:, 0, :, :, :]
        data2 = data[:, 1, :, :, :]

        # print('data1=', data1)
        # print('data2=', data2)

        # feature = np.empty(shape=(self.batch_size, 1000, 1, 2), dtype=floatX)

        conv1 = self.conv11.forward(data1)
        relu1 = self.relu11.forward(conv1)
        pool1 = self.max11.forward(relu1)

        conv2 = self.conv12.forward(pool1)
        relu2 = self.relu12.forward(conv2)
        pool2 = self.max12.forward(relu2)

        conv3 = self.conv13.forward(pool2)
        relu3 = self.relu13.forward(conv3)
        pool3 = self.max13.forward(relu3)

        fc1 = self.fc1.forward(pool3)
        # feature[:, :, 0, 0] = self.relu_fc1.forward(fc1)

        conv1 = self.conv21.forward(data2)
        relu1 = self.relu21.forward(conv1)
        pool1 = self.max21.forward(relu1)

        conv2 = self.conv22.forward(pool1)
        relu2 = self.relu22.forward(conv2)
        pool2 = self.max22.forward(relu2)

        conv3 = self.conv23.forward(pool2)
        relu3 = self.relu23.forward(conv3)
        pool3 = self.max23.forward(relu3)

        fc2 = self.fc2.forward(pool3)
        # feature[:, :, 0, 1] = self.relu_fc1.forward(fc2)

        value = self.fc_last.forward((fc1 - fc2).reshape((self.batch_size, feature_vector_dim, 1, 1)))
        out = self.output_sigmoid.forward(value.reshape((self.batch_size, 1, 1, 1)))

        # print('sigmoid out=', out.reshape((self.batch_size,)))

        return out.reshape((self.batch_size,))

        # output = self.output_tanh.forward(value.reshape((self.batch_size, 1, 1, 1)))
        #
        # output = output.reshape((self.batch_size,))
        # s = np.empty((self.batch_size,), dtype=int)
        # s[output >= 0] = 1
        # s[output < 0] = 0
        # return s

    def back(self, dy: np.ndarray):
        assert dy.shape == (self.batch_size,)

        # print('dy=', dy)
        # grad_tanh = self.output_tanh.gradient(loss.reshape((self.batch_size, 1, 1, 1)))
        # print('grad tanh=', grad_tanh)

        grad_sigmoid = self.output_sigmoid.gradient(dy.reshape((self.batch_size, 1, 1, 1)))
        # print('grad sigmoid =', grad_sigmoid)

        grad_fc_last = self.fc_last.gradient(grad_sigmoid.reshape((self.batch_size, 1)))
        self.fc_last.backward()

        grad_fc1 = self.fc1.gradient(grad_fc_last.reshape((self.batch_size, feature_vector_dim)))
        self.fc1.backward()

        grad_max3 = self.max13.gradient(grad_fc1)
        grad_relu3 = self.relu13.gradient(grad_max3)
        grad_conv3 = self.conv13.gradient(grad_relu3)
        # print(grad_conv3)
        self.conv13.backward()

        grad_max2 = self.max12.gradient(grad_conv3)
        grad_relu2 = self.relu12.gradient(grad_max2)
        grad_conv2 = self.conv12.gradient(grad_relu2)
        # print(grad_conv2)
        self.conv12.backward()

        grad_max1 = self.max11.gradient(grad_conv2)
        grad_relu1 = self.relu11.gradient(grad_max1)
        grad_conv1 = self.conv11.gradient(grad_relu1)
        self.conv11.backward()

        grad_fc2 = self.fc2.gradient(-grad_fc_last.reshape((self.batch_size, feature_vector_dim)))
        self.fc2.backward()

        grad_max3 = self.max23.gradient(grad_fc2)
        grad_relu3 = self.relu23.gradient(grad_max3)
        grad_conv3 = self.conv23.gradient(grad_relu3)
        self.conv23.backward()

        grad_max2 = self.max22.gradient(grad_conv3)
        grad_relu2 = self.relu22.gradient(grad_max2)
        grad_conv2 = self.conv22.gradient(grad_relu2)
        self.conv22.backward()

        grad_max1 = self.max21.gradient(grad_conv2)
        grad_relu1 = self.relu21.gradient(grad_max1)
        grad_conv1 = self.conv21.gradient(grad_relu1)
        self.conv21.backward()

    def get_batches(self, data: np.ndarray, label: np.ndarray):
        n = data.shape[0]
        if n % self.batch_size == 0:
            data_batches = np.empty(shape=(n // self.batch_size, self.batch_size) + data.shape[1:], dtype=floatX)
            label_batches = np.empty(shape=(n // self.batch_size, self.batch_size), dtype=int)
        else:
            data_batches = np.empty(shape=(n // self.batch_size + 1, self.batch_size) + data.shape[1:], dtype=floatX)
            label_batches = np.empty(shape=(n // self.batch_size + 1, self.batch_size), dtype=int)
        for i in range(n // self.batch_size):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            data_batches[i] = data[start:end]
            label_batches[i] = label[start:end]
        if n % self.batch_size != 0:
            start = n * self.batch_size
            rest = n - start
            data_batches[-1, :rest] = data[start:]
            label_batches[-1, :rest] = label[start:]
        return data_batches, label_batches

    def get_label(self, output: np.ndarray):
        label = np.empty(shape=output.shape, dtype=int)
        label[output >= 0.5] = 1
        label[output < 0.5] = 0
        return label

    def train(self, data: np.ndarray, label: np.ndarray, iteration: int):
        data_batches, label_batches = self.get_batches(data, label)
        pred_all = np.empty((data_batches.shape[0] * self.batch_size,), dtype=floatX)
        for iter in range(iteration):
            print('=========================')
            print('iteration ', iter)
            for i in range(data_batches.shape[0]):
                start = time.time()
                pred = self.forward(data_batches[i])
                pred_all[i * self.batch_size:(i + 1) * self.batch_size] = pred
                loss, dy = losses.cross_entropy_error(pred, label_batches[i])
                print(f'\tbatch {i}, loss = {loss}')
                print(f'\tcorrectness = {1 - np.abs(self.get_label(pred) - label_batches[i]).mean()}')
                print(f'\tf1 = {f1_score(label_batches[i], self.get_label(pred))}')
                self.back(dy)
                print(f'\tbatch time = {time.time() - start} s')
            print(f'== all loss = {losses.cross_entropy_error(pred_all[:data.shape[0]], label)[0]}')

    # # 五折交叉验证
    # # todo 交换两张图片的顺序进行训练保证网络对称性
    # n = len(label)
    # assert data.shape == (n, 2, 247, 247, 3)
    # assert label.shape == (n,)
    # index = np.empty(shape=(n,), dtype=int)
    # for i in range(5):
    #     index[i * n // 5:(i + 1) * n // 5] = i
    # np.random.shuffle(index)

    def predict(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        label = np.empty(shape=(n,), dtype=int)
        # out = np.empty(shape=(n,), dtype=int)
        for i in range(n // self.batch_size):
            out = self.forward(data[i * self.batch_size:(i + 1) * self.batch_size])
            label[i * self.batch_size:(i + 1) * self.batch_size] = self.get_label(out)
        if n % self.batch_size != 0:
            start = n // self.batch_size * self.batch_size
            rest = n - n // self.batch_size * self.batch_size
            rest_batch = np.empty((self.batch_size, 2, 247, 247, 3), dtype=floatX)
            rest_batch[:rest] = data[start:]
            out = self.forward(rest_batch)
            label[start:] = self.get_label(out)
        return label


if __name__ == '__main__':
    # data, label = preprocess.get_img()
    # data = data[:batch_size]
    # label = label[:batch_size]
    data = np.empty(shape=(batch_size, 2, 247, 247, 3), dtype=floatX)
    label = np.empty(shape=(batch_size,), dtype=int)
    data[:batch_size // 2, 0] = np.random.randn(batch_size // 2, 247, 247, 3)
    data[:batch_size // 2, 1] = data[:batch_size // 2, 0]
    label[:batch_size // 2] = 1
    data[batch_size // 2:] = np.random.randn(batch_size // 2, 2, 247, 247, 3)
    label[batch_size // 2:] = 0
    cnn = CNN(batch_size=batch_size)
    cnn.train(data, label, 1000)
