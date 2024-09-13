import torch.nn as nn
import torch.nn.functional as F
import torch

class IntermediateBlock(nn.Module):
    def __init__(self, in_channels, num_convs, out_channels):
        super(IntermediateBlock, self).__init__()
        self.conv_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)) for _ in range(num_convs)])
        self.fc = nn.Linear(in_channels, num_convs)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        m = torch.mean(x, dim=[2, 3])
        a = F.softmax(self.fc(m), dim=1)
        conv_outputs = [conv_layer(x) for conv_layer in self.conv_layers]
        stacked_outputs = torch.stack(conv_outputs, dim=1)
        a_expanded = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = (stacked_outputs * a_expanded).sum(dim=1)
        x = self.dropout(x)
        return x

class OutputBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutputBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.block1 = IntermediateBlock(in_channels=3, num_convs=3, out_channels=64)
        self.block2 = IntermediateBlock(in_channels=64, num_convs=3, out_channels=128)
        self.block3 = IntermediateBlock(in_channels=128, num_convs=3, out_channels=256)
        self.block4 = IntermediateBlock(in_channels=256, num_convs=3, out_channels=512)
        self.output_block = OutputBlock(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output_block(x)
        return x
