from import_sets import *

def _validate(modelOutput, labels):
    maxvalues, maxindices = torch.max(modelOutput.data, 1)

    count = 0

    for i in range(0, labels.squeeze(1).size(0)):

        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1

    return count

class CBR(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(CBR, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.LeakyReLU(0.1)
        # self.drop = nn.Dropout3d(0.4)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.drop(x)
        return x

class Block3(nn.Module):
    def __init__(self, in_planes):
        super(Block3, self).__init__()
        self.block1 = CBR(in_planes, in_planes)
        self.block2 = CBR(in_planes, in_planes)
        self.block3 = CBR(2*in_planes, in_planes)

    def forward(self, input):
        b1 = self.block1(input)
        b2 = self.block2(b1)
        b3_in = torch.cat([b1, b2], 1)
        b3 = self.block3(b3_in)
        shortcut = input
        b3 += shortcut
        return b3

class Block4(nn.Module):
    def __init__(self, in_planes):
        super(Block4, self).__init__()
        self.block1 = CBR(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.block2 = CBR(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.block3 = CBR(2*in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.block4 = CBR(3*in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.block5 = CBR(4*in_planes, in_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        b1 = self.block1(input)
        b2 = self.block2(b1)

        b3_in = torch.cat([b1, b2], 1)
        b3 = self.block3(b3_in)

        b4_in = torch.cat([b1, b2, b3], 1)
        b4 = self.block4(b4_in)

        b5_in = torch.cat([b1, b2, b3, b4],1)
        b5 = self.block5(b5_in)
        shortcut = input
        b5 += shortcut
        return b5

class Bottleneck(nn.Module):
    def __init__(self, in_planes):
        super(Bottleneck, self).__init__()
        # self.block = CBR(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.block = Block4(in_planes)
        # self.block3 = nn.Sequential(CBR(in_planes, 2*in_planes),
        #                             CBR(2*in_planes, 2*in_planes),
        #                             CBR(2*in_planes, in_planes))
        # self.block3 = Block3(in_planes)
        # self.block4 = nn.Sequential(CBR(in_planes, 2*in_planes),
        #                             CBR(2*in_planes, 4*in_planes),
        #                             CBR(4*in_planes, 4*in_planes),
        #                             CBR(4*in_planes, 2*in_planes),
        #                             CBR(2*in_planes, in_planes))
        # self.block4 = Block4(in_planes)

    def forward(self, input):
        # x1 = input
        # x2 = self.block2(input)
        # x3 = self.block3(input)
        # x4 = self.block4(input)
        # output = x1 + x2 + x3 + x4
        # return output
        x = self.block(input)
        return x

class LipNet(nn.Module):
    def __init__(self):
        super(LipNet, self).__init__()
        self.l1 = nn.Sequential(CBR(3, 32, stride=(2, 2, 2)),
                                Block3(32),
                                # Block4(16),
                                CBR(32, 32),
                                # CBR(16, 16)
                                )
        self.l2 = nn.Sequential(CBR(32, 64, stride=(2, 2, 2)),
                                # Bottleneck(32),
                                Block3(64),
                                CBR(64, 64),
                                # CBR(32, 32)
                                )
        self.l3 = nn.Sequential(CBR(64, 128, stride=(2, 2, 2)),
                                # Bottleneck(64),
                                Block4(128),
                                CBR(128, 128),
                                # CBR(64, 64)
                                )
        self.l4 = nn.Sequential(CBR(128, 128, stride=(1, 2, 2)),
                                Block3(128),
                                Block4(128),
                                CBR(128, 256),
                                # CBR(128, 128)
                                )
        self.fc = nn.Sequential(nn.Linear(256 * 3 * 7 * 7, 4096),
                                nn.Dropout(0.5),
                                nn.LeakyReLU(0.1),
                                nn.Linear(4096, 311)
                                )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)
            # if isinstance(m, nn.BatchNorm3d):
            #     init.kaiming_normal_(m.weight)
            #     init.constant_(m.bias, 0)
    def validator_function(self):
        return _validate

    def loss(self):
        return nn.CrossEntropyLoss()

    def forward(self, input):
        # print(input.size())
        x = self.l1(input)
        # print(x.size())
        x = self.l2(x)
        # print(x.size())
        x = self.l3(x)
        # print(x.size())
        x = self.l4(x)
        # print(x.size())
        x = x.view(x.size()[0], -1)
        # print(x.size())
        output = self.fc(x)
        # print(output.size())
        return output

# input = torch.rand([4, 3, 24, 112, 112])
# model = LipNet()
# output = model(input)