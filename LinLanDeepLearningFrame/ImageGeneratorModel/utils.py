from . import *
from .functions import *
from .Losses import *


# 计算模型参数量
def count_model_params(model, requires_grad=True):
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad is True)
    return sum(p.numel() for p in model.parameters())


class DiagonalGaussianDistribution(object):
    def __init__(self, tensor, deterministic=False):
        super(DiagonalGaussianDistribution, self).__init__()
        assert tensor.shape[1] % 2 == 0, f"输入的潜在向量无法划分为均值和方差, {tensor.shape[1]}%2 != 0"
        self.params = tensor
        self.mean, self.logvar = tensor.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.deterministic = deterministic
        if deterministic:
            self.var, self.std = torch.zeros_like(self.mean)
        else:
            self.std = torch.exp(0.5 * self.logvar)
            self.var = self.logvar.exp()

    def sample(self):
        out = self.mean + self.std * torch.randn_like(self.mean).to(self.params.device)
        return out

    def mode(self):
        return self.mean

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1. - self.logvar)
            else:
                return 0.5 * torch.sum((self.mean - other.mean).pow(
                    2) / other.var + self.var / other.var - 1. - self.logvar + other.logvar)

    def nll(self, sample):
        if self.deterministic:
            return torch.Tensor([0.])
        logwopi = np.log(2. * np.pi)
        return 0.5 * torch.sum(logwopi + self.logvar + (sample - self.mean).pow(2) / self.var)


def train_diffusion(dataloader, model, opt, Epoch, valid=5, device="cuda", valid_size=36, file_name=None,
                    valid_mode="ddpm", ddim_step=2):
    v = int(math.sqrt(valid_size))
    for epoch in range(Epoch):
        for tx, _ in tqdm(dataloader, desc=f"{epoch}/{Epoch}"):
            tx = tx.to(device)
            model.train()
            opt.zero_grad()
            loss = model(tx)
            loss.backward()
            opt.step()
        if epoch % valid == 0:
            if file_name:
                torch.save(model.state_dict(), file_name)
            model.eval()
            if valid_mode == "ddim":
                out = model.ddim_sample(batch_size=valid_size, step=ddim_step).clamp(0, 1)
            else:
                out = model.sample(batch_size=valid_size).clamp(0, 1)
            im = make_grid(out.detach().cpu(), nrow=v, padding=2).numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis("off")
            plt.show()


def train_diffusion_autocast(dataloader, model, opt, Epoch, valid=5, device="cuda", valid_size=36, file_name=None,
                             valid_mode="ddpm", ddim_step=2):
    v = int(math.sqrt(valid_size))
    scaler = GradScaler()
    for epoch in range(Epoch):
        for tx, _ in tqdm(dataloader, desc=f"{epoch}/{Epoch}"):
            tx = tx.to(device)
            model.train()
            opt.zero_grad()
            with autocast():  # 自动选择FP16/FP32计算
                loss = model(tx)
                # 反向传播（梯度缩放）
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(opt)  # 更新参数
            scaler.update()
        if epoch % valid == 0:
            if file_name:
                torch.save(model.state_dict(), file_name)
            model.eval()
            if valid_mode == "ddim":
                out = model.ddim_sample(batch_size=valid_size, step=ddim_step).clamp(0, 1)
            else:
                out = model.sample(batch_size=valid_size).clamp(0, 1)
            im = make_grid(out.detach().cpu(), nrow=v, padding=2).numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis("off")
            plt.show()


def train_vae(dataloader, model, opt, loss_func, Epoch, valid=5, device="cuda", valid_size=36, file_name=None):
    v = int(math.sqrt(valid_size))
    dim = model.image_shape[1] // (2 ** model.depth)
    norise_shape = (model.latent_dim, dim, dim)
    for epoch in range(Epoch):
        for tx, _ in tqdm(dataloader, desc=f"{epoch}/{Epoch}"):
            tx = tx.to(device)
            model.train()
            opt.zero_grad()
            _, mean, logvar, z = model(tx)
            loss = loss_func(_, tx, mean, logvar)
            loss.backward()
            opt.step()
        if epoch % valid == 0:
            if file_name:
                torch.save(model.state_dict(), file_name)
            model.eval()
            norise = torch.randn(size=(valid_size, *norise_shape)).to(device)
            out = model.decode(norise)
            im = make_grid(out.detach().cpu(), nrow=v, padding=2).numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis("off")
            plt.show()


def train_vae_autocast(dataloader, model, opt, loss_func, Epoch, valid=5, device="cuda", valid_size=36, file_name=None):
    v = int(math.sqrt(valid_size))
    dim = model.image_shape[1] // (2 ** model.depth)
    norise_shape = (model.latent_dim, dim, dim)
    scaler = GradScaler()
    for epoch in range(Epoch):
        for tx, _ in tqdm(dataloader, desc=f"{epoch}/{Epoch}"):
            tx = tx.to(device)
            model.train()
            opt.zero_grad()
            with autocast():  # 自动选择FP16/FP32计算
                _, mean, logvar, z = model(tx)
                loss = loss_func(_, tx, mean, logvar)
                # 反向传播（梯度缩放）
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(opt)  # 更新参数
            scaler.update()
        if epoch % valid == 0:
            if file_name:
                torch.save(model.state_dict(), file_name)
            model.eval()
            norise = torch.randn(size=(valid_size, *norise_shape)).to(device)
            out = model.decode(norise)
            im = make_grid(out.detach().cpu(), nrow=v, padding=2).numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis("off")
            plt.show()


def train_dcgan(dataloader, g_model, g_opt, d_model, d_opt, loss_func, Epoch=100, valid_size=36, valid=5,
                g_file_name=None,
                d_file_name=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    row = int(math.sqrt(valid_size))
    in_dim = g_model.in_dim
    for epoch in range(Epoch):
        g_model.train()
        d_model.train()
        for tx, _ in tqdm(dataloader, desc=f"{epoch}/{Epoch}"):
            tx = tx.to(device)
            true_labels = torch.ones(size=(tx.shape[0], 1)).to(device)
            false_labels = torch.zeros(size=(tx.shape[0], 1)).to(device)
            norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
            g_opt.zero_grad()
            d_opt.zero_grad()
            d_true_pre = d_model(tx)
            d_true_loss = loss_func(d_true_pre, true_labels)
            fake = g_model(norise)
            d_false_pre = d_model(fake.detach())
            d_false_loss = loss_func(d_false_pre, false_labels)
            d_loss = d_true_loss + d_false_loss
            d_loss.backward()
            d_opt.step()
            gen_image = g_model(norise)
            g_true_pre = d_model(gen_image)
            g_loss = loss_func(g_true_pre, true_labels)
            g_loss.backward()
            g_opt.step()
        if epoch % valid == 0:
            if g_file_name:
                torch.save(g_model.state_dict(), g_file_name)
            if d_file_name:
                torch.save(d_model.state_dict, d_file_name)
            valid_norise = torch.randn(size=(valid_size, in_dim)).to(device)
            g_model.eval()
            out = g_model(valid_norise)
            im = make_grid(out, nrow=row, padding=2).detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis("off")
            plt.show()


def train_dcgan_autocast(dataloader, g_model, g_opt, d_model, d_opt, loss_func, Epoch=100, valid_size=36, valid=5,
                         g_file_name=None,
                         d_file_name=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    row = int(math.sqrt(valid_size))
    in_dim = g_model.in_dim
    scaler = GradScaler()
    for epoch in range(Epoch):
        g_model.train()
        d_model.train()
        for tx, _ in tqdm(dataloader, desc=f"{epoch}/{Epoch}"):
            tx = tx.to(device)
            true_labels = torch.ones(size=(tx.shape[0], 1)).to(device)
            false_labels = torch.zeros(size=(tx.shape[0], 1)).to(device)
            norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
            g_opt.zero_grad()
            d_opt.zero_grad()
            with autocast():  # 自动选择FP16/FP32计算
                d_true_pre = d_model(tx)
                d_true_loss = loss_func(d_true_pre, true_labels)
                fake = g_model(norise)
                d_false_pre = d_model(fake.detach())
                d_false_loss = loss_func(d_false_pre, false_labels)
                d_loss = d_true_loss + d_false_loss
            scaler.scale(d_loss).backward()  # 缩放损失并反向传播
            scaler.step(d_opt)  # 更新参数
            scaler.update()
            with autocast():
                gen_image = g_model(norise)
                g_true_pre = d_model(gen_image)
                g_loss = loss_func(g_true_pre, true_labels)
            scaler.scale(g_loss).backward()  # 缩放损失并反向传播
            scaler.step(g_opt)  # 更新参数
            scaler.update()
        if epoch % valid == 0:
            if g_file_name:
                torch.save(g_model.state_dict(), g_file_name)
            if d_file_name:
                torch.save(d_model.state_dict, d_file_name)
            valid_norise = torch.randn(size=(valid_size, in_dim)).to(device)
            g_model.eval()
            out = g_model(valid_norise)
            im = make_grid(out, nrow=row, padding=2).detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis("off")
            plt.show()


def train_wgan(dataloader, g_model, g_opt, d_model, d_opt, Epoch=100, valid_size=36, valid=5, device="cuda",
               lambda_gp=10, train_g_model_step=2, g_file_name=None, d_file_name=None,create_graph=True, retain_graph=True):
    in_dim = g_model.in_dim
    row = int(math.sqrt(valid_size))
    dis_loss_func = WGAN_DiscriminatorLoss(lambda_gp=lambda_gp)
    for epoch in range(Epoch):
        g_model.train()
        d_model.train()
        for cnt, (tx, _) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
            d_opt.zero_grad()
            with torch.no_grad():
                fake_img = g_model(norise)
            d_loss = dis_loss_func(d_model, tx, fake_img,create_graph=create_graph, retain_graph=retain_graph)
            d_loss.backward()
            d_opt.step()
            if cnt % train_g_model_step == 0:
                g_opt.zero_grad()
                norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
                fake_img = g_model(norise)
                dis_g_score = d_model(fake_img)
                g_loss = -dis_g_score.mean()
                g_loss.backward()
                g_opt.step()
        if epoch % valid == 0:
            if g_file_name:
                torch.save(g_model.state_dict(), g_file_name)
            if d_file_name == 0:
                torch.save(d_model.state_dict(), d_file_name)
            g_model.eval()
            norise = torch.randn(size=(valid_size, in_dim)).to(device)
            out = g_model(norise)
            im = make_grid(out, nrow=row, padding=2).detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis('off')
            plt.show()


def train_wgan_autocast(dataloader, g_model, g_opt, d_model, d_opt, Epoch=100, valid_size=36, valid=5, device="cuda",
                        lambda_gp=10, train_g_model_step=2, g_file_name=None, d_file_name=None, create_graph=True, retain_graph=True):
    in_dim = g_model.in_dim
    row = int(math.sqrt(valid_size))
    dis_loss_func = WGAN_DiscriminatorLoss(lambda_gp=lambda_gp)
    scaler = GradScaler()
    for epoch in range(Epoch):
        g_model.train()
        d_model.train()
        for cnt, (tx, _) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
            d_opt.zero_grad()
            with autocast():
                with torch.no_grad():
                    fake_img = g_model(norise)
                d_loss = dis_loss_func(d_model, tx, fake_img, create_graph=create_graph, retain_graph=retain_graph)
            scaler.scale(d_loss).backward()  # 缩放损失并反向传播
            scaler.step(d_opt)  # 更新参数
            scaler.update()
            if cnt % train_g_model_step == 0:
                g_opt.zero_grad()
                with autocast():
                    norise = torch.randn(size=(tx.shape[0], in_dim)).to(device)
                    fake_img = g_model(norise)
                    dis_g_score = d_model(fake_img)
                    g_loss = -dis_g_score.mean()
                scaler.scale(g_loss).backward()  # 缩放损失并反向传播
                scaler.step(g_opt)  # 更新参数
                scaler.update()
        if epoch % valid == 0:
            if g_file_name:
                torch.save(g_model.state_dict(), g_file_name)
            if d_file_name == 0:
                torch.save(d_model.state_dict(), d_file_name)
            g_model.eval()
            norise = torch.randn(size=(valid_size, in_dim)).to(device)
            out = g_model(norise)
            im = make_grid(out, nrow=row, padding=2).detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis('off')
            plt.show()


def train_wgan_vae2d(dataloader, vae_model, vae_opt, dis_model, dis_opt, Epoch=100, valid_size=36, valid=5,
                     vae_file_name=None, dis_file_name=None, device="cuda", lambda_gp=10, create_graph=True, retain_graph=True):
    row = int(math.sqrt(valid_size))
    dis_loss_func = WGAN_DiscriminatorLoss(lambda_gp=lambda_gp)
    vae_struct_loss_func = nn.MSELoss()
    for epoch in range(Epoch):
        vae_model.train()
        dis_model.train()
        for cnt, (tx, _) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            dis_opt.zero_grad()
            with torch.no_grad():
                vae_out, mean, logvar, fake_latent = vae_model(tx)
            real_latent = torch.randn_like(fake_latent)
            d_loss = dis_loss_func(model=dis_model, real_samples=real_latent, fake_samples=fake_latent,
                                   create_graph=create_graph, retain_graph=retain_graph)
            d_loss.backward()
            dis_opt.step()
            vae_opt.zero_grad()
            vae_out, mean, logvar, fake_latent = vae_model(tx)
            vae_latent_score = -dis_model(fake_latent).mean()
            vae_struct_loss = vae_struct_loss_func(vae_out, tx)
            kl_loss = KL_Q2Normal(mean=mean, logvar=logvar)
            vae_loss = vae_struct_loss + kl_loss + vae_latent_score
            vae_loss.backward()
            vae_opt.step()
        if epoch % valid == 0:
            if vae_file_name:
                torch.save(vae_model.state_dict(), vae_file_name)
            if dis_file_name:
                torch.save(dis_model.state_dict(), dis_file_name)
            vae_model.eval()
            out = vae_model.sample(valid_size)
            im = make_grid(out, nrow=row, padding=2).detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis('off')
            plt.show()


def train_wgan_vae2d_autocast(dataloader, vae_model, vae_opt, dis_model, dis_opt, Epoch=100, valid_size=36, valid=5,
                              vae_file_name=None, dis_file_name=None, device="cuda", lambda_gp=10, create_graph=True, retain_graph=True):
    row = int(math.sqrt(valid_size))
    dis_loss_func = WGAN_DiscriminatorLoss(lambda_gp=lambda_gp)
    vae_struct_loss_func = nn.MSELoss()
    scaler = GradScaler()
    for epoch in range(Epoch):
        vae_model.train()
        dis_model.train()
        for cnt, (tx, _) in enumerate(tqdm(dataloader, desc=f"{epoch}/{Epoch}")):
            tx = tx.to(device)
            dis_opt.zero_grad()
            with autocast():
                with torch.no_grad():
                    vae_out, mean, logvar, fake_latent = vae_model(tx)
                real_latent = torch.randn_like(fake_latent)
                d_loss = dis_loss_func(model=dis_model, real_samples=real_latent, fake_samples=fake_latent,
                                       create_graph=create_graph, retain_graph=retain_graph)
            scaler.scale(d_loss).backward()  # 缩放损失并反向传播
            scaler.step(dis_opt)  # 更新参数
            scaler.update()
            vae_opt.zero_grad()
            with autocast():
                vae_out, mean, logvar, fake_latent = vae_model(tx)
                vae_latent_score = -dis_model(fake_latent).mean()
                vae_struct_loss = vae_struct_loss_func(vae_out, tx)
                kl_loss = KL_Q2Normal(mean=mean, logvar=logvar)
                vae_loss = vae_struct_loss + kl_loss + vae_latent_score
            scaler.scale(vae_loss).backward()  # 缩放损失并反向传播
            scaler.step(vae_opt)  # 更新参数
            scaler.update()
        if epoch % valid == 0:
            if vae_file_name:
                torch.save(vae_model.state_dict(), vae_file_name)
            if dis_file_name:
                torch.save(dis_model.state_dict(), dis_file_name)
            vae_model.eval()
            out = vae_model.sample(valid_size)
            im = make_grid(out, nrow=row, padding=2).detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.axis('off')
            plt.show()


def get_load_state_dict_from_compile(file, device="cuda"):
    new_dict = OrderedDict()
    for k, v in torch.load(file, map_location=device).items():
        key = k.replace("_orig_mod.", "")
        new_dict[key] = v
    return new_dict
