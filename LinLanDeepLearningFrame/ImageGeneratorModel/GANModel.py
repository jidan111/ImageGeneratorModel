from .BaseStruct import *


class BaseGAN(nn.Module):
    def __init__(self):
        super(BaseGAN, self).__init__()


class DCGANResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DCGANResnetBlock, self).__init__()
        self.equal = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        if not self.equal:
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return F.leaky_relu(out + x)


class DCGANUp2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, head_num=8):
        super(DCGANUp2dBlock, self).__init__()
        self.attention = attention
        if self.attention:
            self.atten = Attention2dBlock(in_channels=in_channels, head_num=head_num)
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            # DCGANResnetBlock(in_channels=out_channels, out_channels=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        if self.attention:
            x = self.atten(x)
        return self.layer(x)


class DCGANDown2dSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DCGANDown2dSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(out_channels, 0.8)
        )

    def forward(self, x):
        return self.layer(x)


class DCGAN_Generator(BaseGAN):
    def __init__(self, in_dim, image_shape, depth, hidden_channels, attention=()):
        super(DCGAN_Generator, self).__init__()
        self.in_dim = in_dim
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.image_shape = image_shape
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Sequential(
            nn.Linear(in_dim, hidden_channels * (2 ** depth) * self.dim * self.dim),
            nn.Unflatten(1, (hidden_channels * (2 ** depth), self.dim, self.dim)),
        )
        self.set_layer(attention=attention)

    def set_layer(self, attention=()):
        arr = []
        if not attention:
            attention = [False] * self.depth
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(DCGANUp2dBlock(in_channels=self.hidden_channels * (2 ** i),
                                      out_channels=self.hidden_channels * (2 ** (i - 1)),
                                      attention=attention[index],
                                      head_num=self.hidden_channels))
            index += 1
        self.layer = nn.Sequential(
            *arr,
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.image_shape[0], kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init(x)
        out = self.layer(x) / 2 + 0.5
        return out


class DCGAN_Discriminator(BaseGAN):
    def __init__(self, image_shape, depth, hidden_channels):
        super(DCGAN_Discriminator, self).__init__()
        self.image_shape = image_shape
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.dim = image_shape[1] // (2 ** depth)
        self.set_layer()

    def set_layer(self):
        arr = [nn.Conv2d(in_channels=self.image_shape[0], out_channels=self.hidden_channels, kernel_size=1)]
        for i in range(self.depth):
            arr.append(DCGANDown2dSample(
                in_channels=self.hidden_channels * (2 ** i),
                out_channels=self.hidden_channels * (2 ** (i + 1))
            ))
        self.layer = nn.Sequential(
            *arr,
            nn.Flatten(1),
            nn.Linear(self.hidden_channels * (2 ** self.depth) * self.dim * self.dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class WGAN_Generator(BaseGAN):
    def __init__(self, in_dim, image_shape, depth, hidden_channels, attention=()):
        super(WGAN_Generator, self).__init__()
        self.in_dim = in_dim
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.image_shape = image_shape
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Sequential(
            nn.Linear(in_dim, hidden_channels * (2 ** depth) * self.dim * self.dim),
            nn.Unflatten(1, (hidden_channels * (2 ** depth), self.dim, self.dim)),
        )
        self.set_layer(attention=attention)

    def set_layer(self, attention=()):
        arr = []
        if not attention:
            attention = [False] * self.depth
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(DCGANUp2dBlock(in_channels=self.hidden_channels * (2 ** i),
                                      out_channels=self.hidden_channels * (2 ** (i - 1)),
                                      attention=attention[index],
                                      head_num=self.hidden_channels))
            index += 1
        self.layer = nn.Sequential(
            *arr,
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.image_shape[0], kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.init(x)
        out = self.layer(x) / 2 + 0.5
        return out



class WGAN_Discriminator(BaseGAN):
    def __init__(self, image_shape, depth, hidden_channels):
        super(WGAN_Discriminator, self).__init__()
        self.image_shape = image_shape
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.dim = image_shape[1] // (2 ** depth)
        self.set_layer()

    def set_layer(self):
        arr = [nn.Conv2d(in_channels=self.image_shape[0], out_channels=self.hidden_channels, kernel_size=1)]
        for i in range(self.depth):
            arr.append(DCGANDown2dSample(
                in_channels=self.hidden_channels * (2 ** i),
                out_channels=self.hidden_channels * (2 ** (i + 1))
            ))
        self.layer = nn.Sequential(
            *arr,
            nn.Flatten(1),
            nn.Linear(self.hidden_channels * (2 ** self.depth) * self.dim * self.dim, 1),
        )

    def forward(self, x):
        out = self.layer(x)
        return out
