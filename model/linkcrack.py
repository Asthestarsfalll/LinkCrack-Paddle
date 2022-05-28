import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ResidualBlock(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            dilation=[1, 1],
            shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation[0],
                dilation=dilation[0],
                bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation[1],
                dilation=dilation[1],
                bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        self.relu = nn.ReLU()
        self.right = shortcut if shortcut is not None else nn.Identity()

    def forward(self, x):
        out = self.left(x)
        res = self.right(x)
        return self.relu(out + res)


class DecoderBlock(nn.Layer):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        scale,
        upsample_mode="bilinear",
        BN_enable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
        self.scale = scale

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            bias_attr=False
        )

        self.conv2 = nn.Conv2D(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias_attr=False
        )

        if BN_enable:
            self.norm1 = nn.BatchNorm2D(mid_channels)
            self.norm2 = nn.BatchNorm2D(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, down_inp, up_inp):
        x = paddle.concat([down_inp, up_inp], axis=1)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=self.scale,
                          mode='bilinear', align_corners=True)
        return self.relu2(self.norm2(self.conv2(x)))


class LinkCrack(nn.Layer):
    def __init__(self):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2D(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias_attr=True),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias_attr=True),
            nn.BatchNorm2D(64),
            nn.ReLU(),
        )

        self._res1_shorcut = nn.Sequential(
            nn.Conv2D(64, 64, 1, 2, bias_attr=False),
            nn.BatchNorm2D(64),
        )

        self.res1 = nn.Sequential(
            ResidualBlock(64, 64, stride=2, shortcut=self._res1_shorcut),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self._res2_shorcut = nn.Sequential(
            nn.Conv2D(64, 64, 1, 2, bias_attr=False),
            nn.BatchNorm2D(64),
        )

        self.res2 = nn.Sequential(
            ResidualBlock(64, 64, stride=2, shortcut=self._res2_shorcut),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self._res3_shorcut = nn.Sequential(
            nn.Conv2D(64, 128, 1, 2, bias_attr=False),
            nn.BatchNorm2D(128),
        )

        self.res3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, shortcut=self._res3_shorcut),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128, dilation=(2, 2)),
            ResidualBlock(128, 128, dilation=(2, 2)),
            ResidualBlock(128, 128, dilation=(2, 2)),
        )

        self._res4_shorcut = nn.Sequential(
            nn.Conv2D(128, 128, 1, 1, bias_attr=False),
            nn.BatchNorm2D(128),
        )

        self.res4 = nn.Sequential(
            ResidualBlock(128, 128, dilation=(2, 2),
                          shortcut=self._res4_shorcut),
            ResidualBlock(128, 128, dilation=(4, 4)),
            ResidualBlock(128, 128, dilation=(4, 4)),
        )

        self.dec4 = DecoderBlock(128+128, 128, 64, scale=2)
        self.dec3 = DecoderBlock(64+64, 64, 64, scale=2)
        self.dec2 = DecoderBlock(64+64, 64, 64, scale=2)

        self.mask = nn.Sequential(
            nn.Conv2D(64, 32, 3, padding=1, bias_attr=True),
            nn.ReLU(),
            nn.Conv2D(32, 1, 1, bias_attr=False),
        )

        self.link = nn.Sequential(
            nn.Conv2D(64, 32, 3, padding=1, bias_attr=True),
            nn.ReLU(),
            nn.Conv2D(32, 8, 1, bias_attr=False),
        )

    def forward(self, x):
        x = self.pre(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.dec4(x4, x3)
        x6 = self.dec3(x5, x2)
        x7 = self.dec2(x6, x1)
        mask = self.mask(x7)
        link = self.link(x7)
        return mask, link
        # return x, x1, x2, x3, x4, x5, x6, x7, mask, link


if __name__ == "__main__":
    # x = paddle.randn([2, 3, 256, 256])
    import numpy as np
    from reprod_log import ReprodLogger
    sta = paddle.load("./pre.pdparams")
    # print(sta['_res1_shorcut.0.bias'])
    # print(sta.keys())
    # x = np.load('./input.npy')
    model = LinkCrack()

    model.set_state_dict(sta)
    # x, x1, x2, x3, x4, x5, x6, x7, mask, link = model(paddle.to_tensor(x))
    # a = ReprodLogger()
    # a.add("x", x.detach().numpy())
    # a.add("x1", x1.detach().numpy())
    # a.add("x2", x2.detach().numpy())
    # a.add("x3", x3.detach().numpy())
    # a.add("x4", x4.detach().numpy())
    # a.add("x5", x5.detach().numpy())
    # a.add("x6", x6.detach().numpy())
    # a.add("x7", x7.detach().numpy())
    # a.add("mask", mask.detach().numpy())
    # a.add("link", link.detach().numpy())
    # a.save('./paddle.npy')
