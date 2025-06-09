from . import *


class ConditionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConditionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
            nn.Linear(in_features=out_dim, out_features=out_dim)
        )
        self.out_dim = out_dim

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, self.out_dim, 1, 1)


class AdaGroupNormConditional(nn.Module):
    def __init__(self, in_channels, num_groups, condition_dim):
        super(AdaGroupNormConditional, self).__init__()
        assert in_channels % num_groups == 0, f"输入通道数未能被平均分组, {in_channels}%{num_groups}!=0"
        self.condition_mlp = ConditionMLP(condition_dim, in_channels * 2)
        self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups)

    def forward(self, x, condition):
        condition = self.condition_mlp(condition)
        scale, shift = condition.chunk(2, dim=1)
        out = self.norm(x)
        out = (1 + scale) * out + shift
        return out


class QKVAttention(nn.Module):
    def __init__(self, head_num):
        super(QKVAttention, self).__init__()
        self.head_num = head_num

    def forward(self, x):
        b, c, hw = x.shape
        assert c % (3 * self.head_num) == 0, f"通道数无法被头数均分, {c}%{self.head_num}!=0"
        ch = c // (3 * self.head_num)
        q, k, v = x.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(b * self.head_num, ch, hw),
            (k * scale).view(b * self.head_num, ch, hw),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(b * self.head_num, ch, hw))
        return a.reshape(b, -1, hw)


class Attention2dBlock(nn.Module):
    def __init__(self, in_channels, head_num):
        super(Attention2dBlock, self).__init__()
        self.in_channels = in_channels
        self.head_num = head_num
        self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=head_num)
        self.qkv = nn.Conv1d(in_channels=in_channels, out_channels=3 * in_channels, kernel_size=1)
        if self.head_num is None:
            self.atten = QKVAttention(head_num=1)
        else:
            assert in_channels * 3 % (3 * self.head_num) == 0, f"通道数无法被头数均分, {in_channels}%{self.head_num}!=0"
            self.atten = QKVAttention(head_num=head_num)
        self.proj_out = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        b, c, *_ = x.shape
        x = x.view(b, self.in_channels, -1)
        qkv = self.qkv(x)
        atten = self.atten(qkv)
        out = self.proj_out(atten)
        out = (x + out).reshape(b, c, *_)
        return self.norm(out)


class Attention2dBlockConditional(nn.Module):
    def __init__(self, in_channels, head_num, condition_dim):
        super(Attention2dBlockConditional, self).__init__()
        self.in_channels = in_channels
        self.head_num = head_num
        self.condition = ConditionMLP(in_dim=condition_dim, out_dim=in_channels)
        self.norm = AdaGroupNormConditional(in_channels=in_channels, num_groups=head_num, condition_dim=condition_dim)
        self.qkv = nn.Conv1d(in_channels=in_channels, out_channels=3 * in_channels, kernel_size=1)
        if self.head_num is None:
            self.atten = QKVAttention(head_num=1)
        else:
            assert in_channels * 3 % (3 * self.head_num) == 0, f"通道数无法被头数均分, {in_channels}%{self.head_num}!=0"
            self.atten = QKVAttention(head_num=head_num)
        self.proj_out = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x, condition):
        b, c, *_ = x.shape
        condition_ = self.condition(condition)
        x = x + condition_
        x = x.view(b, self.in_channels, -1)
        qkv = self.qkv(x)
        atten = self.atten(qkv)
        out = self.proj_out(atten)
        out = (x + out).reshape(b, c, *_)
        return self.norm(out, condition)


class Resnet2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resnet2dBlock, self).__init__()
        self.equal = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        if not self.equal:
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return self.activation(self.norm(x + out))


class Resnet2dBlockConditional(Resnet2dBlock):
    def __init__(self, in_channels, out_channels, condition_dim):
        super(Resnet2dBlockConditional, self).__init__(in_channels=in_channels, out_channels=out_channels)
        self.condition_mlp = ConditionMLP(in_dim=condition_dim, out_dim=in_channels)

    def forward(self, x, condition):
        condition = self.condition_mlp(condition)
        x = x + condition
        out = self.conv1(x)
        if not self.equal:
            x = self.conv2(x)
        return self.activation(self.norm(x + out))


class Down2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, head_num=8):
        super(Down2dBlock, self).__init__()
        self.resnet = Resnet2dBlock(in_channels=in_channels, out_channels=in_channels)
        self.attention = attention
        if self.attention:
            self.atten = Attention2dBlock(in_channels=in_channels, head_num=head_num)
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        if self.attention:
            x = self.atten(x)
        x = self.resnet(x)
        out = self.down(x)
        return out


class Down2dBlockConditional(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim, attention=False, head_num=8, group_num=8):
        super(Down2dBlockConditional, self).__init__()
        self.resnet = Resnet2dBlockConditional(in_channels=in_channels, out_channels=in_channels,
                                               condition_dim=condition_dim)
        self.attention = attention
        if self.attention:
            self.atten = Attention2dBlockConditional(in_channels=in_channels, head_num=head_num,
                                                     condition_dim=condition_dim)
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = AdaGroupNormConditional(in_channels=out_channels, num_groups=group_num, condition_dim=condition_dim)

    def forward(self, x, condition):
        if self.attention:
            x = self.atten(x, condition)
        x = self.resnet(x, condition)
        out = self.down(x)
        return self.norm(out, condition)


class Up2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, head_num=8):
        super(Up2dBlock, self).__init__()
        self.resnet = Resnet2dBlock(in_channels=in_channels, out_channels=in_channels)
        self.attention = attention
        if self.attention:
            self.atten = Attention2dBlock(in_channels=in_channels, head_num=head_num)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.out = Resnet2dBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        if self.attention:
            x = self.atten(x)
        x = self.resnet(x)
        out = self.up(x)
        out = self.out(out)
        return out


class Up2dBlockConditional(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim, attention=False, head_num=8, group_num=8):
        super(Up2dBlockConditional, self).__init__()
        self.resnet = Resnet2dBlockConditional(in_channels=in_channels, out_channels=in_channels,
                                               condition_dim=condition_dim)
        self.attention = attention
        if self.attention:
            self.atten = Attention2dBlockConditional(in_channels=in_channels, head_num=head_num,
                                                     condition_dim=condition_dim)
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                     padding=1)
        self.norm = AdaGroupNormConditional(in_channels=out_channels, condition_dim=condition_dim, num_groups=group_num)
        self.out = Resnet2dBlockConditional(in_channels=out_channels,out_channels=out_channels, condition_dim=condition_dim)

    def forward(self, x, condition):
        if self.attention:
            x = self.atten(x, condition)
        x = self.resnet(x, condition)
        out = self.up(x)
        out = self.out(out, condition)
        return self.norm(out, condition)


class ESRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(ESRCNNBlock, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=5, padding=2),
            nn.PReLU(),
        )
        self.shrinking_layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels // 2, kernel_size=1),
            nn.PReLU(),
        )
        self.mapping_layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels // 2, out_channels=hidden_channels // 2, kernel_size=3, stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_channels // 2, out_channels=hidden_channels // 2, kernel_size=3, stride=1,
                      padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=hidden_channels // 2, out_channels=hidden_channels // 2, kernel_size=3, stride=1,
                      padding=1),
            nn.PReLU(),
        )
        self.expanding_layer = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels // 2, out_channels=hidden_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.out = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.feature_layer(x)
        out = self.shrinking_layer(out)
        out = self.mapping_layer(out)
        out = self.expanding_layer(out)
        out = self.out(out)
        return out
