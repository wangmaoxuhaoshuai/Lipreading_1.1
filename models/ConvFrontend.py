from import_sets import *
class ConvFrontend(nn.Module):
    def __init__(self):
        super(ConvFrontend, self).__init__()
        self.conv = nn.Conv3d(1, 64, (5,7,7),stride=(1,2,2),padding=(2,3,3))
        self.norm = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((1,3,3),stride=(1,2,2),padding=(0,1,1))

    def forward(self, input):
        output = self.pool(F.relu(self.conv(input)))
        return output
