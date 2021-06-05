import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, feat_dim=128, img_size=224, output_channle=3):
        super(Decoder, self).__init__()

        self.init_size = img_size // 4
        self.fc1 = nn.Sequential(nn.Linear(feat_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_channle, 3, stride=1, padding=1),
            nn.Tanh(),
        )


    def forward(self, feat):
        x = self.fc1(feat)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(x)
        return img

if __name__ == '__main__':
    x = torch.randn((10, 128))
    x = x.cuda()
    model = Decoder()
    model.cuda()
    res = model(x)
    print('hhh')
