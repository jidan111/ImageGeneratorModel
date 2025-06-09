from . import *
from .Losses import *


class DDPM(nn.Module):
    """
    实现了DDPM和DDIM，只是反采样方式不同，需要做后续模型验证研究，可以使用ddim_sample
    """

    def __init__(self, model, image_shape, beta=(1e-4, 0.02), T=100, t_dim=64, device="cpu", schedule_name="linear",
                 s=0.008, loss_mode="mse"):
        """
        :param model: 噪声预测器模型，接受(x, t)作为输入，返回norise
        :param image_shape: tuple: (c, h, w)，作为后续sample采样图片时候的默认大小
        :param beta: linear噪声调度时，需传入的beta值，建议使用默认值，会根据T来进行缩放，目前几个测试表现效果都不错
        :param T: 噪声步数T
        :param t_dim: 时间嵌入维度
        :param device: 硬件设备,cpu 或者 cuda
        :param schedule_name: 噪声调度方案，cosine 或者 linear
        :param s: cosine噪声调度的平滑系数
        :param loss_mode: 损失函数类型
        """
        super(DDPM, self).__init__()
        self.device = device
        self.T = T
        self.image_shape = image_shape
        self.t_dim = t_dim
        self.loss = DiffusionLoss(loss_mode)
        self.model = model
        self.time_embd = nn.Embedding(T, t_dim)
        for k, v in self.set_schedule_params(schedule_name=schedule_name, s=s, beta=beta).items():
            self.register_buffer(k, v)

    def get_named_beta_schedule(self, schedule_name, beta=(1e-4, 0.02), s=0.008):
        """
        :param schedule_name: 噪声调度方式名称，cosine或者linear
        :param beta: linear调度的参数
        :param s: cosine调度的平滑系数
        :return:
        """
        if schedule_name == "linear":
            scale = 1000 / self.T
            beta_start = scale * beta[0]
            beta_end = scale * beta[1]
            return torch.linspace(
                beta_start, beta_end, self.T
            ).view(-1, 1, 1, 1)
        elif schedule_name == "cosine":
            return self.betas_for_alpha_bar(
                lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2,
                max_beta=0.999
            ).view(-1, 1, 1, 1)
        else:
            raise NotImplementedError(f"没有预设该噪声调度算法: {schedule_name}")

    def betas_for_alpha_bar(self, alpha_bar, max_beta=0.999):
        """
        :param alpha_bar: 函数
        :param max_beta: beta最大值
        :return:
        """
        betas = []
        for i in range(self.T):
            t1 = i / self.T
            t2 = (i + 1) / self.T
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.Tensor(betas)

    def set_schedule_params(self, schedule_name, beta=(1e-4, 0.02), s=0.008):
        """
        用于计算alpha,beta,alpha_bar,beta_bar等参数
        :param schedule_name: 噪声调度方法名称：linear，cosine
        :param beta:
        :param s:
        :return:
        """
        beta = self.get_named_beta_schedule(schedule_name=schedule_name, beta=beta, s=s)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        beta_bar = 1 - alpha_bar
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_beta_bar = torch.sqrt(beta_bar)
        sqrt_beta = torch.sqrt(beta)
        sqrt_alpha = torch.sqrt(alpha)
        return {"alpha": alpha, "beta": beta, "sqrt_alpha": sqrt_alpha, "sqrt_beta": sqrt_beta,
                "sqrt_alpha_bar": sqrt_alpha_bar,
                "sqrt_beta_bar": sqrt_beta_bar}

    def add_norise(self, x, index, norise):
        """
        实现前向加噪过程 x_t = sqrt_alpha_bar*x_0 + sqrt(1-alpha_bar)*norise
        :param x: Tensor:(batch_size, channels, h, w)
        :param index: Tensor:(batch_size, 1)
        :param norise: Tensor:(batch_size, channels, h, w)
        :return:
        """
        out = self.sqrt_alpha_bar[index] * x + self.sqrt_beta_bar[index] * norise
        return out

    def forward(self, x):
        """
        返回预测噪声与真实噪声之间的差距
        :param x:
        :return:
        """
        norise = torch.randn_like(x)
        index = torch.randint(0, self.T, size=(x.shape[0],)).to(torch.long).to(x.device)
        t = self.time_embd(index)
        target = self.add_norise(x=x, index=index, norise=norise)
        pre = self.model(target, t)
        loss = self.loss(pre, norise)
        return loss

    @torch.no_grad()
    def sample_p(self, x, t, batch_size):
        """
        DDPM模型采样方法实现单个采样
        :param x:
        :param t:
        :param batch_size:
        :return:
        """
        index = torch.Tensor([t]).repeat(batch_size).to(torch.long).to(self.device)
        t = self.time_embd(index)
        pre = self.model(x, t)
        norise = torch.randn_like(x).to(self.device)
        mu = (x - (self.beta[index] / self.sqrt_beta_bar[index]) * pre) / self.sqrt_alpha[index]
        x = mu + self.sqrt_beta[index] * norise
        return x

    @torch.no_grad()
    def sample(self, batch_size):
        self.model.eval()
        self.time_embd.eval()
        x = torch.randn(size=(batch_size, *self.image_shape)).to(self.device)
        for i in tqdm(range(self.T - 1, -1, -1), desc=f"generate"):
            x = self.sample_p(x=x, t=i, batch_size=batch_size)
        return x

    @torch.no_grad()
    def ddim_sample_p(self, x, old_t, target_t, batch_size, sigma=0.0):
        """
        DDIM模型采样方法实现单个采样
        :param x:
        :param old_t:
        :param target_t:
        :param batch_size:
        :param sigma:
        :return:
        """
        old_index = torch.Tensor([old_t]).repeat(batch_size).to(torch.long).to(self.device)
        target_index = torch.Tensor([target_t]).repeat(batch_size).to(torch.long).to(self.device)
        t_emb = self.time_embd(old_index)
        epsilon_theta = self.model(x, t_emb)
        x0_pred = (x - self.sqrt_beta_bar[old_index] * epsilon_theta) / self.sqrt_alpha_bar[old_index]
        sigma = sigma
        noise = torch.randn_like(x)
        x_prev_mean = self.sqrt_alpha_bar[target_index] * x0_pred
        x_prev_var = torch.sqrt(1 - self.sqrt_alpha_bar[target_index] ** 2 - sigma ** 2)
        x_prev_noise = sigma * noise
        x_prev = x_prev_mean + x_prev_var * epsilon_theta + x_prev_noise
        return x_prev

    @torch.no_grad()
    def ddim_sample(self, batch_size, step=2, x=None, sigma=0.0):
        self.model.eval()
        self.time_embd.eval()
        if x is None:
            x = torch.randn(size=(batch_size, *self.image_shape)).to(self.device)
        for i in tqdm(range(self.T - 1, step, -step), desc="generate"):
            x = self.ddim_sample_p(x=x, old_t=i, target_t=i - step, batch_size=batch_size, sigma=sigma)
        return x

    @torch.no_grad()
    def save_gif(self, batch_size):
        self.model.eval()
        self.time_embd.eval()
        index = self.T
        row = int(math.sqrt(batch_size))
        x = torch.randn(size=(batch_size, *self.image_shape)).to(self.device)
        im = make_grid(x, nrow=row, padding=1, normalize=True)
        save_image(im, f"./tmp/{index}.png")
        index -= 1
        for i in tqdm(range(self.T - 1, -1, -1), desc=f"generate"):
            x = self.sample_p(x=x, t=i, batch_size=batch_size)
            im = make_grid(x, nrow=row, padding=1, normalize=True)
            save_image(im, f"./tmp/{index}.png")
            index -= 1