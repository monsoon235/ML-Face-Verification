import torch
from . import convolution, full_connection, other, pooling, preprocess

floatX = torch.float32
device = 'cuda'

batch = 20

learning_rate = 0.01 / batch


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
                                             learning_rate=learning_rate, activate_func='mfm')
        self.conv2a = convolution.Convolution((64, 64, 48), out_channel=96, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv2 = convolution.Convolution((64, 64, 48), out_channel=192, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, activate_func='mfm')
        self.conv3a = convolution.Convolution((32, 32, 96), out_channel=192, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv3 = convolution.Convolution((32, 32, 96), out_channel=384, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, activate_func='mfm')
        self.conv4a = convolution.Convolution((16, 16, 192), out_channel=384, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv4 = convolution.Convolution((16, 16, 192), out_channel=256, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, activate_func='mfm')
        self.conv5a = convolution.Convolution((16, 16, 128), out_channel=256, kernel_size=(1, 1), stride=(1, 1),
                                              learning_rate=learning_rate, activate_func='mfm')
        self.conv5 = convolution.Convolution((16, 16, 128), out_channel=256, kernel_size=(3, 3), stride=(1, 1),
                                             learning_rate=learning_rate, activate_func='mfm')
        self.pool1 = pooling.Pooling((128, 128, 48), pool_size=2)
        self.pool2 = pooling.Pooling((64, 64, 96), pool_size=2)
        self.pool3 = pooling.Pooling((32, 32, 192), pool_size=2)
        self.pool4 = pooling.Pooling((16, 16, 128), pool_size=2)
        self.fc1 = full_connection.FullConnection(8 * 8 * 128, 512, learning_rate=learning_rate, activate_func='mfm')

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
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

    def backward(self, eta: torch.Tensor):
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
