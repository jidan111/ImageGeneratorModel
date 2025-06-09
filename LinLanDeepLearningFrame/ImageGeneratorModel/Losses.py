from . import *
from .functions import *


class DiffusionLoss(nn.Module):
    def __init__(self, mode="mse"):
        super(DiffusionLoss, self).__init__()
        if mode == "mse":
            self.loss = nn.MSELoss()

    def forward(self, pre, y):
        return self.loss(pre, y)


class VAELoss(nn.Module):
    def __init__(self, alpha=0.8, beta=1, reduction="sum", bce=True):
        super(VAELoss, self).__init__()
        self.beta = beta
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        if bce:
            self.bce_loss = nn.BCELoss(reduction=reduction)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.alpha = alpha

    def KL(self, mean, logvar):
        if self.reduction == "mean":
            return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    def forward(self, pre, y, mean, logvar):
        mse_loss = self.mse(pre, y)
        bce_loss = self.bce_loss(pre, y)
        kl = self.KL(mean, logvar)
        return self.alpha * mse_loss + (1 - self.alpha) * bce_loss + self.beta * kl


class WVAE_MMDLoss(nn.Module):
    def __init__(self, beta=1, kernel_mul=2, kernel_num=5):
        super(WVAE_MMDLoss, self).__init__()
        self.beta = beta
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.mse = nn.MSELoss()

    def forward(self, pre, y, z):
        mse = self.mse(pre, y)
        prior_sample = torch.randn_like(z).to(z.device)
        mmd = compute_mmd(z=z, prior_samples=prior_sample, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)
        return mse + self.beta * mmd


class WGAN_DiscriminatorLoss(nn.Module):
    def __init__(self, lambda_gp=10):
        super(WGAN_DiscriminatorLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def forward(self, model, real_samples, fake_samples, create_graph=True, retain_graph=True):
        alpha = torch.rand(size=(real_samples.shape[0], 1, 1, 1)).to(real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(
            real_samples.device)
        d_interpolates = model(interpolates)
        fake = torch.ones(size=(real_samples.shape[0], 1), requires_grad=False).to(real_samples.device)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        true_score = model(real_samples)
        fake_score = model(fake_samples)
        return -true_score.mean() + fake_score.mean() + self.lambda_gp * gradient_penalty


class PerceptualLoss(nn.Module):
    def __init__(self, model):
        super(PerceptualLoss, self).__init__()
        self.model = model
        self.model.eval()

    def forward(self, pre, y):
        with torch.no_grad():
            a = self.model(pre)
            b = self.model(y)
        a = F.normalize(a, p=2, dim=1, eps=1e-12)
        b = F.normalize(b, p=2, dim=1, eps=1e-12)
        distance = (a - b) ** 2
        loss = distance.mean()
        return loss


class WeightPerceptualLoss(nn.Module):
    def __init__(self, model=None, weight=None, num_channels=None):
        super(WeightPerceptualLoss, self).__init__()
        assert weight or num_channels, "权重和通道数至少有一个已知"
        self.model = model
        self.model.eval()
        self.learnint_weight = not weight
        if weight is None:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        else:
            self.weight = weight.view(1, -1, 1, 1)

    def forward(self, pre, y):
        with torch.no_grad():
            pre = self.model(pre)
            y = self.model(y)
        assert pre.shape[1] == self.weight.shape[
            1], f"加权数量和预训练模型输出通道数不匹配, 模型输出通道数:{pre.shape[1]}, 加权数量:{self.weight.shape[1]}"
        if self.learnint_weight:
            weight = F.softmax(self.weight, dim=0)
        else:
            weight = self.weight
        distance = (pre - y) ** 2
        loss = distance * weight
        return loss.mean()
