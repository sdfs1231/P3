import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # block 1
        self.conv1 = nn.Conv2d(3,8,5,2,0)
        self.bn_1 = nn.BatchNorm2d(8)
        # block 2

        self.conv2_1 = nn.Conv2d(8,16,3,1,0)
        self.conv2_2 = nn.Conv2d(16,16,3,1,0)
        self.bn_2 = nn.BatchNorm2d(16)

        # block 3
        self.conv3_1 = nn.Conv2d(16,24,3,1,0)
        self.conv3_2 = nn.Conv2d(24,24,3,1,0)
        self.bn_3 = nn.BatchNorm2d(24)
        # block 4
        self.conv4_1 = nn.Conv2d(24,40,3,1,1)
        self.conv4_2 = nn.Conv2d(40,80,3,1,1)
        self.bn4 = nn.BatchNorm2d(80)

        self.ip1 = nn.Linear(4*4*80,256)
        self.ip1bn = nn.BatchNorm1d(256)
        self.ip2 = nn.Linear(256,256)
        self.ip2bn = nn.BatchNorm1d(256)
        self.ip3 = nn.Linear(256,42)

        self.prelu_1 = nn.PReLU()
        self.prelu_2_1 = nn.PReLU()
        self.prelu_2_2 = nn.PReLU()
        self.prelu_3_1 = nn.PReLU()
        self.prelu_3_2 = nn.PReLU()
        self.prelu_4_1 = nn.PReLU()
        self.prelu_4_2 = nn.PReLU()
        self.prelu_ip1 = nn.PReLU()
        self.prelu_ip2 = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.max_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        x = self.max_pool(self.prelu_1(self.bn_1(self.conv1(x))))

        x = self.prelu_2_1(self.conv2_1(x))
        x = self.prelu_2_2(self.conv2_2(x))
        x = self.max_pool(x)

        x = self.prelu_3_1(self.conv3_1(x))
        x = self.prelu_3_2(self.conv3_2(x))
        x = self.max_pool(x)

        x = self.prelu_4_1(self.conv4_1(x))
        ip3 = self.prelu_4_2(self.conv4_2(x))

        ip3 = ip3.view(-1,4*4*80)

        ip3 = self.prelu_ip1(self.ip1bn(self.ip1(ip3)))

        ip3 = self.prelu_ip2(self.ip2bn(self.ip2(ip3)))

        ip3 = self.ip3(ip3)

        return ip3




