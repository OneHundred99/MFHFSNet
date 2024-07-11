#from torchsummary import summary
import torch
from torch import nn
from torch.nn import functional

class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):  # 6 12 18 | 3 5 7 | 4 8 12
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    
class Conv3X3(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, dilation=1):
        super(Conv3X3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=(self.padding, self.padding),
                               dilation=(self.dilation, self.dilation))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        return x3


class Conv1X1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1X1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=(1, 1),
                               stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        return x3


class MultiScaleConcat(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(MultiScaleConcat, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = Conv1X1(in_channels, mid_channels)
        self.conv2 = Conv1X1(mid_channels, out_channels)
        self.conv3 = Conv3X3(self.out_channels, self.out_channels // 2)
        self.conv4 = Conv3X3(self.out_channels // 2, self.out_channels // 4)
        self.conv5 = Conv3X3(self.out_channels // 4, self.out_channels // 8)
        self.conv6 = Conv3X3(self.out_channels // 8, self.out_channels // 8)

        self.conv7 = Conv3X3(self.out_channels, self.out_channels // 2, dilation=1)
        self.conv8 = Conv3X3(self.out_channels, self.out_channels // 4, dilation=2, padding=2)
        self.conv9 = Conv3X3(self.out_channels, self.out_channels // 8, dilation=3, padding=3)
        self.conv10 = Conv3X3(self.out_channels, self.out_channels // 8, dilation=4, padding=4)

        self.conv11 = Conv1X1(self.out_channels, self.out_channels // 2)
        self.conv12 = Conv1X1(self.out_channels // 2, self.out_channels // 4)
        self.conv13 = Conv1X1(self.out_channels // 4, self.out_channels // 8)
        self.conv14 = Conv1X1(self.out_channels // 4, self.out_channels // 8)

        self.conv15 = Conv1X1(self.out_channels, self.out_channels)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)

        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        x7 = self.conv7(x2)
        x8 = self.conv8(x2)
        x9 = self.conv9(x2)
        x10 = self.conv10(x2)

        x11 = self.conv11(torch.cat([x3, x7], dim=1))
        x12 = self.conv12(torch.cat([x4, x8], dim=1))
        x13 = self.conv13(torch.cat([x5, x9], dim=1))
        x14 = self.conv14(torch.cat([x6, x10], dim=1))
        res = self.conv15(torch.cat([x11, x12, x13, x14], dim=1))
        return res


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, padding=None, kernel_size=2, stride=2):
        super(Down, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool2d = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        self.conv = MultiScaleConcat(in_channels=self.in_channels, mid_channels=self.out_channels,
                                     out_channels=self.out_channels)

    def forward(self, inputs):
        y1 = self.max_pool2d(inputs)
        y2 = self.conv(y1)
        return y2


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = Conv3X3(in_chan, out_chan)
        self.conv_atten1 = nn.Conv1d(1, 1, (5,), bias=False, padding=2)
        self.conv_atten2 = nn.Conv1d(1, 1, (5,), bias=False, padding=2)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()
        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x):
        feat = self.conv(x)
        atten1 = functional.avg_pool2d(feat, feat.size()[2:])
        atten2 = functional.max_pool2d(feat, feat.size()[2:])
        atten1 = self.conv_atten2(self.conv_atten1(atten1.squeeze(-1).transpose(-1, -2)))
        atten2 = self.conv_atten2(self.conv_atten1(atten2.squeeze(-1).transpose(-1, -2)))
        atten1 = atten1.transpose(-1, -2).unsqueeze(-1)
        atten2 = atten2.transpose(-1, -2).unsqueeze(-1)

        atten = self.bn_atten(self.alpha * atten1 + self.beta * atten2)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        y = self.bn(x)
        z = self.relu(y)
        return z


class UpConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.double_conv = MultiScaleConcat(self.in_channels, self.mid_channels, self.out_channels)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()
        self.trans_conv = nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels//2,
                                             kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, inputs):
     #   x = self.double_conv(inputs)
        y = self.bn(inputs)
        z = self.relu(y)
        res = self.trans_conv(z)
        return res

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class EncoderHead(nn.Module):
    def __init__(self, in_channels, out_channels, head=False):
        super(EncoderHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if head:
            self.conv = MultiScaleConcat(in_channels=in_channels, mid_channels=self.out_channels,
                                         out_channels=self.out_channels)
        else:
            self.conv = Down(self.in_channels, self.out_channels, padding='same')

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )

        self.SE = Squeeze_Excite_Block(self.out_channels)        

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.SE(x + s)
        return y
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, head=False):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if head:
            self.conv = MultiScaleConcat(in_channels=in_channels, mid_channels=self.out_channels,
                                         out_channels=self.out_channels)
        else:
            self.conv = Down(self.in_channels, self.out_channels, padding='same')

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )

        self.SE = Squeeze_Excite_Block(self.out_channels)        

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.SE(x + s)
        return y

class PFC(nn.Module):
    def __init__(self,channels, kernel_size=1):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, padding=  kernel_size // 2),
                    #nn.Conv2d(3, channels, kernel_size=3, padding= 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x
    
class MixUpHead(nn.Module):
    def __init__(self, in_channels):
        super(MixUpHead, self).__init__()
        self.in_channels = in_channels
        self.arm = AttentionRefinementModule(in_chan=self.in_channels, out_chan=self.in_channels)
        
        self.pfc = PFC(self.in_channels)
    def forward(self, inputs):
      #  out = self.arm(inputs)
        out = self.pfc(inputs)
        return out
class AttentionGate(nn.Module):
    # 因为我们是将encoder的当前层 以及decoder的下一层上采样之后的结果送入Attention_Block的
    # 所以他们的尺寸以及通道数是相同的
    def __init__(self, in_channels, out_channels):
        super(AttentionGate, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU()
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1, 1, 0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class MixUp(nn.Module):
    def __init__(self, in_channels):
        super(MixUp, self).__init__()
        self.in_channels = in_channels
        self.arm = AttentionRefinementModule(in_chan=self.in_channels, out_chan=self.in_channels)
        self.down_conv = DownConv(in_channels=self.in_channels // 2, out_channels=self.in_channels)
        
        self.atten = AttentionGate(in_channels=self.in_channels // 2, out_channels=self.in_channels // 2)
        
        ################AFF##############
        inter_channels = int(self.in_channels // 4)
        self.local_att = nn.Sequential(
                         nn.Conv2d(self.in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
                         nn.BatchNorm2d(inter_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
                         nn.BatchNorm2d(self.in_channels),
                         )

        self.global_att = nn.Sequential(
                          nn.AdaptiveAvgPool2d(1),
                          nn.Conv2d(self.in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(inter_channels),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(self.in_channels),
                          )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs1, inputs2, inputs3):
        att = self.atten(inputs2, inputs3)
        down = self.down_conv(att)        

        xl = self.local_att(inputs1)
        xg = self.global_att(inputs1)
        xlg = xl+xg
        wei = self.sigmoid(xlg)
        xo = 2*inputs1*wei+2*down*wei     
        return xo
        #out = self.arm(inputs1) + self.down_conv(torch.cat([inputs2, inputs3], dim=1))
        #return out


class DecoderTail(nn.Module):
    def __init__(self, in_channels):
        super(DecoderTail, self).__init__()
        self.in_channels = in_channels
        self.arm = AttentionRefinementModule(in_chan=self.in_channels, out_chan=self.in_channels)
        
        self.pfc = PFC(self.in_channels)
        
    def forward(self, inputs):
#        out = self.arm(inputs)
        out = self.pfc(inputs)
        return out

class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]  # 获取batch_size

        feats = [conv(x) for conv in self.convs]  # 让x分成3*3和5*5进行卷积
        feats = torch.cat(feats, dim=1)  # 合并卷积结果
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        # reshape一下大小
        # 接下来计算图中的U
        feats_U = torch.sum(feats, dim=1)  # 两个分支得到的卷积结果相加
        feats_S = self.gap(feats_U)  # 自适应池化，也就是对各个chanel求均值得到图中的S
        feats_Z = self.fc(feats_S)  # fc层压缩特征得到图中的Z

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # 不同的分支各自恢复特征Z到channel的宽度
        attention_vectors = torch.cat(attention_vectors, dim=1)  # 连接起来方便后续操作
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        # reshape起来方便后续操作
        attention_vectors = self.softmax(attention_vectors)  # softmax得到图中的a和b

        feats_V = torch.sum(feats * attention_vectors, dim=1)
        # 把softmax后的各自自注意力跟卷积后的结果相乘，得到图中select的结果，然后相加得到最终输出

        return feats_V
def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.arm = AttentionRefinementModule(in_chan=self.out_channels, out_chan=self.out_channels)
        self.up_conv = UpConv(in_channels=self.in_channels//2, mid_channels=self.mid_channels,
                              out_channels=self.out_channels*2)


        self.atten = AttentionGate(in_channels=self.in_channels // 2,out_channels=self.in_channels // 2)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=1, stride=1, padding=0)
        self.conv1x1 = conv1x1(self.in_channels//2, self.in_channels//2)
        self.SE = Squeeze_Excite_Block(self.out_channels*2)
        
        ################AFF##############
        inter_channels = int(self.in_channels // 4)
        self.local_att = nn.Sequential(
                         nn.Conv2d(self.out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
                         nn.BatchNorm2d(inter_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(inter_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
                         nn.BatchNorm2d(self.out_channels),
                         )

        self.global_att = nn.Sequential(
                          nn.AdaptiveAvgPool2d(1),
                          nn.Conv2d(self.out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(inter_channels),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(inter_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
                          nn.BatchNorm2d(self.out_channels),
                          )

        self.sigmoid = nn.Sigmoid()

        self.SK = SKConv(features=self.in_channels//4)

    def forward(self, inputs1, inputs2, inputs3):
        att = self.atten(inputs2, inputs3)
        conv = self.conv1x1(att)
        SE = self.SE(conv)
        upconv = self.up_conv(SE)
        
        xl = self.local_att(inputs1)
        xg = self.global_att(inputs1)
        xlg = xl+xg
        wei = self.sigmoid(xlg)
        xo = 2*inputs1*wei+2*upconv*wei
        
        out = self.SK(xo)
       # out = self.arm(inputs1) + self.up_conv(torch.cat([inputs2, inputs3], dim=1))
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.filters = [self.base_channels * 2 ** i for i in range(5)]
        # encoder
        self.encoderhead = EncoderHead(self.in_channels, self.base_channels)
        self.encoder1 = Encoder(self.in_channels, self.base_channels)
        self.encoder2 = Encoder(self.base_channels, self.base_channels * 2)
        self.encoder3 = Encoder(self.base_channels * 2, self.base_channels * 4)
        self.encoder4 = Encoder(self.base_channels * 4, self.base_channels * 8)
        self.encoder5 = Encoder(self.base_channels * 8, self.base_channels * 16)
        # mixup
        self.mixuphead = MixUpHead(self.filters[0])
        self.mixup1 = MixUp(self.filters[1])
        self.mixup2 = MixUp(self.filters[2])
        self.mixup3 = MixUp(self.filters[3])
        self.mixup4 = MixUp(self.filters[4])
        # decoder
        self.decoder_tail = DecoderTail(self.filters[4])
        self.decoder4 = Decoder(self.filters[4] * 2, self.filters[4], self.filters[3])
        self.decoder3 = Decoder(self.filters[3] * 2, self.filters[3], self.filters[2])
        self.decoder2 = Decoder(self.filters[2] * 2, self.filters[2], self.filters[1])
        self.decoder1 = Decoder(self.filters[1] * 2, self.filters[1], self.filters[0])

        self.aspp = ASPP(32, 32)        

        self.out_conv1 = nn.Conv2d(in_channels=self.filters[0], out_channels=self.filters[0], kernel_size=(3, 3),
                                   padding=(1, 1), stride=(1, 1))
        self.out_bn1 = nn.BatchNorm2d(num_features=self.filters[0])
        self.out_relu1 = nn.ReLU()
        self.out_conv2 = nn.Conv2d(in_channels=self.filters[0], out_channels=self.num_classes, kernel_size=(1, 1),
                                   stride=(1, 1))

    def forward(self, inputs):
        # encoder
        x1 = self.encoderhead(inputs)
#        x2 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        # mixup
        y1 = self.mixuphead(x1)
        y2 = self.mixup1(x2, x1, y1)
        y3 = self.mixup2(x3, x2, y2)
        y4 = self.mixup3(x4, x3, y3)
        y5 = self.mixup4(x5, x4, y4)
        # decoder
        z5 = self.decoder_tail(y5)
        z4 = self.decoder4(y4, y5, z5)
        z3 = self.decoder3(y3, y4, z4)
        z2 = self.decoder2(y2, y3, z3)
        z1 = self.decoder1(y1, y2, z2)

#        z4 = self.decoder_tail(y4)
#        z3 = self.decoder3(y3, y4, z4)
#        z2 = self.decoder2(y2, y3, z3)
#        z1 = self.decoder1(y1, y2, z2)
        out = self.aspp(z1)
        out = self.out_conv1(z1)
        out = self.out_bn1(out)
        out = self.out_relu1(out)
        out = self.out_conv2(out)
        out = torch.sigmoid(out)
        return out


