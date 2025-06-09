from .BaseStruct import *


class Unet2d(nn.Module):
    def __init__(self, image_shape, depth, hidden_channels, attention=None):
        super(Unet2d, self).__init__()
        assert image_shape[1] % 2**depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{2**depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.depth = depth
        self.hidden_channels = hidden_channels
        self.set_encoder()
        self.set_decoder(attention)
        self.set_midlayer()
        self.out = nn.Sequential(
            Resnet2dBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels),
            nn.Conv2d(in_channels=self.hidden_channels, out_channels=image_shape[0], kernel_size=1)
        )

    def set_encoder(self):
        arr = []
        for h in range(self.depth):
            arr.append(Down2dBlock(in_channels=self.hidden_channels * (2 ** h),
                                   out_channels=self.hidden_channels * (2 ** (h + 1)),
                                   head_num=self.hidden_channels))
        self.encoder = nn.ModuleList(arr)

    def set_decoder(self, attention: list):
        arr = []
        if not attention:
            attention = [False] * self.depth
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(Up2dBlock(in_channels=2 * self.hidden_channels * (2 ** i),
                                 out_channels=self.hidden_channels * (2 ** (i - 1)),
                                 head_num=self.hidden_channels,
                                 attention=attention[index]))
            index += 1
        self.decoder = nn.ModuleList(arr)

    def set_midlayer(self):
        self.mid_layer = nn.Sequential(
            Attention2dBlock(in_channels=self.hidden_channels * (2 ** self.depth), head_num=self.hidden_channels),
            Resnet2dBlock(in_channels=self.hidden_channels * (2 ** self.depth),
                          out_channels=2 * self.hidden_channels * (2 ** self.depth)),
        )

    def forward(self, x):
        x = self.init(x)
        tmp = []
        for model in self.encoder:
            if not tmp:
                tmp.append(model(x))
            else:
                tmp.append(model(tmp[-1]))
        mid = self.mid_layer(tmp.pop())
        out = None
        for model in self.decoder:
            if out is None:
                out = model(mid)
            else:
                t = torch.cat((tmp.pop(), out), dim=1)
                out = model(t)
        out = self.out(out)
        return out


class Unet2dConditional(nn.Module):
    def __init__(self, image_shape, depth, hidden_channels, condition_dim, attention=None, num_groups=8, mean_logvar=False):
        super(Unet2dConditional, self).__init__()
        assert image_shape[1] % 2**depth == 0, f"输入图像与模型深度不匹配，{image_shape[1]}%{depth} is not int"
        assert image_shape[1] == image_shape[2], f"暂时只支持方形图片， {image_shape[1]} != {image_shape[2]}"
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.init_condition = ConditionMLP(in_dim=condition_dim, out_dim=hidden_channels)
        self.depth = depth
        self.num_groups = num_groups
        self.condition_dim = condition_dim
        self.hidden_channels = hidden_channels
        self.set_encoder()
        self.set_decoder(attention)
        self.set_midlayer()
        if mean_logvar:
            self.out = nn.Sequential(
                Resnet2dBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels),
                nn.Conv2d(in_channels=self.hidden_channels, out_channels=image_shape[0]*2, kernel_size=1)
            )
        else:
            self.out = nn.Sequential(
                Resnet2dBlock(in_channels=self.hidden_channels, out_channels=self.hidden_channels),
                nn.Conv2d(in_channels=self.hidden_channels, out_channels=image_shape[0], kernel_size=1)
            )

    def set_encoder(self):
        arr = []
        for h in range(self.depth):
            arr.append(Down2dBlockConditional(in_channels=self.hidden_channels * (2 ** h),
                                              out_channels=self.hidden_channels * (2 ** (h + 1)),
                                              head_num=self.hidden_channels,
                                              condition_dim=self.condition_dim,
                                              group_num=self.num_groups))
        self.encoder = nn.ModuleList(arr)

    def set_decoder(self, attention: list):
        arr = []
        if not attention:
            attention = [False] * self.depth
        index = 0
        for i in range(self.depth, 0, -1):
            arr.append(Up2dBlockConditional(in_channels=2 * self.hidden_channels * (2 ** i),
                                            out_channels=self.hidden_channels * (2 ** (i - 1)),
                                            head_num=self.hidden_channels,
                                            attention=attention[index],
                                            condition_dim=self.condition_dim,
                                            group_num=self.num_groups))
            index += 1
        self.decoder = nn.ModuleList(arr)

    def set_midlayer(self):
        self.mid_layer = nn.ModuleList(
            [Attention2dBlockConditional(in_channels=self.hidden_channels * (2 ** self.depth),
                                         head_num=self.hidden_channels, condition_dim=self.condition_dim),
             Resnet2dBlockConditional(in_channels=self.hidden_channels * (2 ** self.depth),
                                      out_channels=2 * self.hidden_channels * (2 ** self.depth),
                                      condition_dim=self.condition_dim)]
        )

    def forward(self, x, condition):
        x = self.init(x)
        condition_ = self.init_condition(condition)
        x = x + condition_
        tmp = []
        for model in self.encoder:
            if not tmp:
                tmp.append(model(x, condition))
            else:
                tmp.append(model(tmp[-1], condition))
        mid = self.mid_layer[0](tmp.pop(), condition)
        mid = self.mid_layer[1](mid, condition)
        out = None
        for model in self.decoder:
            if out is None:
                out = model(mid, condition)
            else:
                t = torch.cat((tmp.pop(), out), dim=1)
                out = model(t, condition)
        out = self.out(out)
        return out
