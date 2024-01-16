from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0')

class OrientationNet(nn.Module):
    def __init__(
        self,
        dendrite=1,
        init_w_mul=0.01,
        init_w_add=0.2,
        init_q=0,
        pad=0,
        k=10,
    ):
        super(OrientationNet, self).__init__()
        self.frontconv = FrontConv(
            pad=pad
        )
        # self.dconvOnOff = FrontConvOnOffResponse(
        #     pad=pad
        # )
        self.dconvSynaps = DConvSynaps(
            dendrite=dendrite,
            init_w_mul=init_w_mul,
            init_w_add=init_w_add,
            init_q=init_q,
            k=k
        )
        self.dconvDend = DConvDend()
        self.dconvMenb = DConvMenb()
        self.dconvSoma = DConvSoma()
        self.calcOutput = CalcOutput()

    # @profile
    def forward(self, x):
        x = self.frontconv(x)
        # x = self.dconvOnOff(x)
        x = self.dconvSynaps(x)
        x = self.dconvDend(x)
        x = self.dconvMenb(x)
        x = self.dconvSoma(x)
        x = self.calcOutput(x)

        return x


class FrontConv(nn.Module):
    def __init__(
        self,
        input_dim=((1, 32, 32)),
        output_dim=((900, 18)),
        filter_size=3,
        pad=0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_size = filter_size
        self.pad = pad
        image_w = input_dim[2]
        self.activate = nn.Sigmoid()

    def forward(self, x):
        # im2col
        x = nn.Unfold(kernel_size=(self.filter_size, self.filter_size), stride=(
            1, 1), padding=self.pad, dilation=(1, 1))(x)


#on off response
        x_center = torch.cat(
        [x[:, 4, ...].unsqueeze(0)for _ in range(9)], dim=0)
        x_center = x_center.permute(1, 0, 2)
        #print(x.size)
        #x_center = x_center.permute(1,0,2,3)
        x1 = torch.isclose(x, x_center, rtol=0, atol=3)#atol表示onoff的阈值
        x1 = x1.float()
        #print(x1.size)
        x1[:, 4, :] = 1 
        x2 = x#.permute(1, 0, 2) 
        x2 = x2.float()
        x2a = x2.size(0)
        x2b = x2.size(1)
        x2c = x2.size(2)
        x = torch.zeros((x2a, 2 * x2b, x2c))
        x[:, 0:9, :] = x1
        x[:, 9:18, :] = x2 
        #print(x.shape)
        x.to(device)


        return x




class DConvSynaps(nn.Module):
    def __init__(
        self,
        dendrite=1,
        init_w_mul=0.01,
        init_w_add=0.2,
        init_q=0,
        k=10
    ):
        self.dendrite = dendrite
        super().__init__()
        self.W = nn.Parameter(
            torch.Tensor(
                init_w_mul * 
                #np.ones((self.dendrite, 18, 4))
                np.abs(np.random.randn(self.dendrite, 18, 4))
                +init_w_add #
            ))
        self.q = nn.Parameter(
            torch.Tensor(
                init_w_mul * 
                #np.ones((self.dendrite, 18, 4))
                np.abs(np.random.randn(self.dendrite, 18, 4))
                +init_q #.random.randn
            ))
        self.activation = nn.Sigmoid()

        self.k = k

    def forward(self, x):
        x_width = x.shape[2]
        W = self.W.expand(x.shape[0], x_width, self.dendrite,
                          18, 4)
        q = self.q.expand(x.shape[0], x_width, self.dendrite,
                          18, 4)
        x = torch.cat([x.unsqueeze(0) for _ in range(4)], dim=0)
        x = x.unsqueeze(0)
        x = x.permute(2, 4, 0, 3, 1)
        return self.activation((x.to(device) * W - q) * self.k)


class LeNet(nn.Module):
    def __init__(self,inputChannel=6, cls_num=8):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(inputChannel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 128)  # 4*4 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, cls_num)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x



class DConvDend(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return torch.prod(x, 3)


class DConvMenb(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, 2)


class DConvSoma(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation((x - 0.5) * 10)


class CalcOutput(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, 1)
