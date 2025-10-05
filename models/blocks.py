import torch 
import torch.nn as nn
import numpy as np

class MaxPoolArgMax(nn.Module):
    def __init__(self,stride_x, stride_y, padding_x, padding_y,kernel_size):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size,stride = (stride_x,stride_y),padding = (padding_x,padding_y),return_indices=True)
    def forward(self,x):
        return self.pool(x)

class Conv(nn.Module):
    def __init__(self,stride_x, stride_y, padding_x, padding_y,kernel_size,in_channels,out_channels,n_layers):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Conv2d(in_channels=in_channels if i==0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=(stride_x,stride_y),
                        padding=(padding_x,padding_y)
                    ) 
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.convolve = nn.Sequential(*layers)
    def forward(self,x):
        return self.convolve(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=3,out_channels=64,n_layers=2)
        self.maxpool1 = MaxPoolArgMax(stride_x=2,stride_y=2,padding_x=0,padding_y=0,kernel_size=2)
        self.conv2 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=64,out_channels=128,n_layers=2)
        self.maxpool2 = MaxPoolArgMax(stride_x=2,stride_y=2,padding_x=0,padding_y=0,kernel_size=2)
        self.conv3 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=128,out_channels=256,n_layers=3)
        self.maxpool3 = MaxPoolArgMax(stride_x=2,stride_y=2,padding_x=0,padding_y=0,kernel_size=2)
        self.conv4 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=256,out_channels=512,n_layers=3)
        self.maxpool4 = MaxPoolArgMax(stride_x=2,stride_y=2,padding_x=0,padding_y=0,kernel_size=2)
        self.conv5 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=512,out_channels=512,n_layers=3)
        self.maxpool5 = MaxPoolArgMax(stride_x=2,stride_y=2,padding_x=0,padding_y=0,kernel_size=2)
    def forward(self,x):
        x1 = self.conv1(x)
        x,idx_1 = self.maxpool1(x1)
        s1 = x1.size()
        x2 = self.conv2(x)
        x,idx_2 = self.maxpool2(x2)
        s2 = x2.size()
        x3 = self.conv3(x)
        x,idx_3 = self.maxpool3(x3)
        s3 = x3.size()
        x4 = self.conv4(x)
        x,idx_4 = self.maxpool4(x4)
        s4 = x4.size()
        x5 = self.conv5(x)
        x,idx_5 = self.maxpool5(x5)
        s5 = x5.size()
        indices = [idx_1,idx_2,idx_3,idx_4,idx_5]
        sizes = [s1,s2,s3,s4,s5]
        return x, indices, sizes
    
class Uppool(nn.Module):
    def __init__(self):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2,stride=2)
    def forward(self,x,indices,output_size=None):
        return self.unpool(x,indices,output_size=output_size)

class Decoder(nn.Module):
    def __init__(self,n_classes):
        super().__init__()

        self.uppool1 = Uppool()
        self.uppool2 = Uppool()
        self.uppool3 = Uppool()
        self.uppool4 = Uppool()
        self.uppool5 = Uppool()

        self.conv1 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=512,out_channels=512,n_layers=3)
        self.conv2 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=512,out_channels=256,n_layers=3)
        self.conv3 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=256,out_channels=128,n_layers=3)
        self.conv4 = Conv(stride_x=1,stride_y=1,padding_x=3,padding_y=3,kernel_size=7,in_channels=128,out_channels=64,n_layers=2)
        self.conv5 = nn.Conv2d(64, n_classes, kernel_size=7, stride=1, padding=3)
    def forward(self,x,indices,sizes):
        x = self.conv1(self.uppool1(x,indices[-1],sizes[-1]))
        x = self.conv2(self.uppool2(x,indices[-2],sizes[-2]))
        x = self.conv3(self.uppool3(x,indices[-3],sizes[-3]))
        x = self.conv4(self.uppool4(x,indices[-4],sizes[-4]))
        x = self.conv5(self.uppool5(x,indices[-5],sizes[-5]))
        return x
    
