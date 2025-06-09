from . import *


def NLL_Q2Unknown(q_mean, q_logvar, unknown_sample, reduction="sum"):
    logwopi = np.log(2. * np.pi)
    nll = 0.5 * (logwopi + q_logvar + (unknown_sample - q_mean).pow(2) * torch.exp(-q_logvar))
    if reduction == "mean":
        return nll.mean()
    return nll.sum()


def KL_Q2P(q_mean, q_logvar, p_mean, p_logvar, reduction="sum"):
    kl = 0.5 * (-1 + p_logvar - q_logvar + torch.exp(q_logvar - p_logvar) + (q_mean - p_mean).pow(2) * torch.exp(
        -p_logvar))
    if reduction == "mean":
        return kl.mean()
    return kl.sum()


def KL_Q2Normal(mean, logvar, reduction="sum"):
    if reduction == "mean":
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


def standard_normal_cdf(x):
    return 0.5 * (1.0 + F.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x.pow(3))))


def compute_mmd(z, prior_samples, kernel_mul=2., kernel_num=5):
    """
    计算最大均值差异
    :param z: 编码器输出:[batch_size, latent_dim]
    :param prior_samples: 从先验分布中采样结果:[batch_size, latent_dim]
    :param kernel_mul: 核带宽乘数因子
    :param kernel_num: 高斯核数量
    :return: mmd损失值
    """
    assert z.shape == prior_samples.shape, f"潜在空间和采样先验采样值形状不同,{z.shape}!={prior_samples.shape}"
    batch_size = z.shape[0]
    z = z.view(batch_size, -1)
    prior_samples = prior_samples.view(batch_size, -1)
    total = torch.cat([z, prior_samples], dim=0)  # (2*batch_size, latent_dim)
    # 欧式距离
    # distance = total.pow(2).sum(dim=1, keepdim=True) + \
    #            total.pow(2).sum(dim=1).unsqueeze(0) - \
    #            2 * torch.mm(total, total.t())
    distance = torch.cdist(total, total, p=2).pow(2)
    distance = distance.clamp(min=1e-6)
    kernels = []
    median_dist = torch.median(distance).item()
    if median_dist == 0:
        median_dist = 1
    sigma_list = [kernel_mul ** i * median_dist for i in range(kernel_num)]
    for sigma in sigma_list:
        gamma = 1. / (2 * sigma ** 2)
        kernels.append(torch.exp(-gamma * distance))
    kernel_val = sum(kernels) / len(kernels)
    k_z = kernel_val[:batch_size, :batch_size]
    k_prior = kernel_val[batch_size:, batch_size:]
    k_zp = kernel_val[:batch_size, batch_size:]
    mmd = k_z.mean() + k_prior.mean() - 2 * k_zp.mean()
    return mmd


def compute_gradient_penalty(D, real_samples, fake_samples,create_graph=True,retain_graph=True):
    alpha = torch.rand(size=(real_samples.shape[0], 1, 1, 1)).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(real_samples.device)
    d_interpolates = D(interpolates)
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
    return gradient_penalty
