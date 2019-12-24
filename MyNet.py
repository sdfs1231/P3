import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import cv2
from data import get_train_test_set

class Net(nn.Module):
    def __init__(self):
        super(Net).__init__()
        # block 1
        self.conv1 = nn.Conv2d(1,8,5,2,0)
        # block 2
        self.conv2_1 = nn.Conv2d(8,16,3,1,0)
        self.conv2_2 = nn.Conv2d(16,16,3,1,0)
        # block 3
        self.con3_1 = nn.Conv2d(16,24,3,1,0)
        self.con3_2 = nn.Conv2d(24,24,3,1,0)
        # block 4
        self.con4_1 = nn.Conv2d(24,40,3,1,1)
        self.con4_2 = nn.Conv2d(40,80,3,1,1)

        self.ip1 = nn.Linear(4*4*80,128)
        self.ip2 = nn.Linear(128,128)
        self.ip3 = nn.Linear(128,42)

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

    def forward(self, x):
        x = self.ave_pool(self.prelu_1(self.conv1(x)))

        x = self.prelu_2_1(self.conv2_1(x))
        x = self.prelu_2_2(self.conv2_2(x))
        x = self.ave_pool(x)

        x = self.prelu_3_1(self.conv3_1(x))
        x = self.prelu_3_2(self.conv3_2(x))
        x = self.ave_pool(x)

        x = self.prelu_4_1(self.conv4_1(x))
        x = self.prelu_4_2(self.conv4_2(x))

        ip3 = x.view(-1,4*4*80)

        ip3 = self.prelu_ip1(self.ip1(ip3))

        ip3 = self.prelu_ip2(self.ip2(ip3))

        ip3 = self.ip3(ip3)

        return ip3

def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('*'*3+'batch-size',type=int,default=64,metavar='N',
                        help='input batch size for training (default:64)')

    parser.add_argument('*' * 3 + 'test-batch-size' , type=int, default=64, metavar='N',
                        help='input test batch size for training (default:64)')

    parser.add_argument('*' * 3 + 'epochs' , type=int, default=100, metavar='N',
                        help='number of epochs for training (default:100)')

    parser.add_argument('*' * 3 + 'lr' , type=float, default=0.001, metavar='N',
                        help='learning rate for training (default:0.001)')

    parser.add_argument('*' * 3 + 'momentum' , type=float, default=0.5, metavar='N',
                        help='SGD momentum for training (default:0.5)')

    parser.add_argument('*' * 3 + 'no-cuda' , action='store_true',default=False,
                        help='disable CUDA training')

    parser.add_argument('*' * 3 + 'seed' , type=int, default=1, metavar='S',
                        help='random seed (default:1)')

    parser.add_argument('*' * 3 + 'log-interval' , type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status(default:20)')

    parser.add_argument('*' * 3 + 'save-model' , action='strore_true', default=True,
                        help='save the current model')

    parser.add_argument('*' * 3 + 'save-directory' , type=str,action='trained_models',
                        help='learn models are saving here')

    parser.add_argument('*' * 3 + 'phase' , type=str, default=True,
                        help='training,predicting or finetuning')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    #for multi GPUs
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print("===> Loading Datasets")
    train_set , test_set = get_train_test_set()
    train_loader = torch.utils.data.Dataloader(train_set,batch_size=args.batch_size,shuffle=True)
    train_loader = torch.utils.data.Dataloader(test_set, batch_size=args.batch_size, shuffle=True)

    print("===> Building Model")
    model = Net().to(device)

    criterion_pts = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)
    





