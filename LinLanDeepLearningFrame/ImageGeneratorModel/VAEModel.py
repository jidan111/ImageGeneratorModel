from .BaseStruct import *
from .functions import *
from .utils import *

class VAE2d(nn.Module):
    def __init__(self, image_shape, depth, hidden_channels, latent_dim, attention=None, tanh=True):
        super(VAE2d, self).__init__()
        assert image_shape[1] % 2 ** depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{2 ** depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.image_shape = image_shape
        self.depth = depth
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.set_encoder()
        self.set_decoder(attention)
        self.tanh = tanh
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.q_enc = nn.Conv2d(in_channels=self.hidden_channels * (2 ** self.depth),
                               out_channels=latent_dim * 2, kernel_size=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=self.hidden_channels * (2 ** self.depth),
                               kernel_size=1)

    def set_encoder(self):
        arr = []
        for h in range(self.depth):
            arr.append(Down2dBlock(in_channels=self.hidden_channels * (2 ** h),
                                   out_channels=self.hidden_channels * (2 ** (h + 1)),
                                   head_num=self.hidden_channels
                                   ))
        self.encoder = nn.Sequential(*arr)

    def set_decoder(self, attention):
        if not attention:
            attention = [False] * self.depth
        arr = []
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(Up2dBlock(in_channels=self.hidden_channels * (2 ** i),
                                 out_channels=self.hidden_channels * (2 ** (i - 1)),
                                 head_num=self.hidden_channels,
                                 attention=attention[index]))
            index += 1
        self.decoder = nn.Sequential(
            *arr,
            ESRCNNBlock(in_channels=self.hidden_channels, out_channels=self.image_shape[0],
                        hidden_channels=self.hidden_channels)
        )

    def encode(self, x):
        h = self.encoder(x)
        q_enc = self.q_enc(h)
        mean, logvar = q_enc.chunk(2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(mean.device)
        return mean + std * eps

    def decode(self, x):
        x = self.p_dec(x)
        out = self.decoder(x)
        if self.tanh:
            return F.tanh(out) / 2 + 0.5
        return out

    def forward(self, x):
        x = self.init(x)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar, z

    @torch.no_grad()
    def get_latent_space(self, x):
        x = self.init(x)
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return z

    @torch.no_grad()
    def decode_latent_space(self, x):
        out = self.decode(x)
        return out

    @torch.no_grad()
    def sample(self, batch_size):
        norise = torch.randn(size=(batch_size, self.latent_dim, self.dim, self.dim)).to(self.init.weight.device)
        out = self.decode_latent_space(norise)
        return out


class LatentVAE2d(nn.Module):
    def __init__(self, image_shape, depth, hidden_channels, latent_dim, attention=None):
        super(LatentVAE2d, self).__init__()
        assert image_shape[1] % 2 ** depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{2 ** depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.image_shape = image_shape
        self.depth = depth
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.set_encoder()
        self.set_decoder(attention)
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.q_enc = nn.Conv2d(in_channels=self.hidden_channels * (2 ** self.depth),
                               out_channels=latent_dim * 2, kernel_size=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=self.hidden_channels * (2 ** self.depth),
                               kernel_size=1)

    def set_encoder(self):
        arr = []
        for h in range(self.depth):
            arr.append(Down2dBlock(in_channels=self.hidden_channels * (2 ** h),
                                   out_channels=self.hidden_channels * (2 ** (h + 1)),
                                   head_num=self.hidden_channels
                                   ))
        self.encoder = nn.Sequential(*arr)

    def set_decoder(self, attention):
        if not attention:
            attention = [False] * self.depth
        arr = []
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(Up2dBlock(in_channels=self.hidden_channels * (2 ** i),
                                 out_channels=self.hidden_channels * (2 ** (i - 1)),
                                 head_num=self.hidden_channels,
                                 attention=attention[index]))
            index += 1
        self.decoder = nn.Sequential(
            *arr,
            Resnet2dBlock(in_channels=self.hidden_channels, out_channels=self.image_shape[0])
        )

    def encode(self, x):
        h = self.encoder(x)
        q_enc = self.q_enc(h)
        q_prior = DiagonalGaussianDistribution(q_enc)
        return q_prior
    def decode(self, x):
        x = self.p_dec(x)
        out = self.decoder(x)
        return out

    def forward(self, x):
        x = self.init(x)
        q_prior = self.encode(x)
        z = q_prior.sample()
        out = self.decode(z)
        return out, q_prior