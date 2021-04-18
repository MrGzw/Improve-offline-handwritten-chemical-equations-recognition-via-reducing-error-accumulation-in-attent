import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision
import resnet as resnet
import random
'''
Feature_Extractor
'''


class Feature_Extractor(nn.Module):
    def __init__(self, strides, compress_layer, input_shape):
        super(Feature_Extractor, self).__init__()
        self.model = resnet.resnet45(strides, compress_layer)
        self.input_shape = input_shape

    def forward(self, input):
        features = self.model(input)
        return features

    def Iwantshapes(self):  # 输出resnet45 layer1-6各个层输出向量的尺寸  即[3, 4, 6, 6, 3]共5层网络的输出尺寸[[1,2],[2,3],[3,4]..]
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]  # [[1,2,3],[2,3,4]]...剔除了x的第一维度即batch，剩下[chanel,height,weight]


'''
Convolutional Alignment Module
'''


# Current version only supports input whose size is a power of 2, such as 32, 64, 128 etc.
# You can adapt it to any input size by changing the padding or stride.
class CAM(nn.Module):
    def __init__(self, scales, maxT, depth, num_channels):  # scales:即[3, 4, 6, 6, 3]共5层网络的输出尺寸[[1,2],[2,3],[3,4]..]
        super(CAM, self).__init__()
        # cascade multiscale features
        fpn = []  # 这里就是论文中CAM的最上面，卷积相加操作，通过记录特征提取层每层数据尺寸的变化，再生成卷积操作，确保两次卷积后大小一样，最后相加，这里也就解释了特征提取层为什么要记录尺寸变化
        for i in range(1, len(scales)):  # scales[a,b,c]:即输入的[chanel,height,weight]
            assert not (scales[i - 1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i - 1,
                                                                                                         i)  # 判断resnet5层输出尺寸必须是整数倍
            assert not (scales[i - 1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]  # if downsampling ratio >= 3, the kernel size is 5, else 3
            r_h, r_w = int(scales[i - 1][1] / scales[i][1]), int(
                scales[i - 1][2] / scales[i][2])  # 尺寸缩小比例，即下采样比例或者卷积核stride
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]  # if r_h == 3 the kernel size is 5
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][0],  # inputChanel，outputchannel
                                               (ksize_h, ksize_w),  # kenersize
                                               (r_h, r_w),  # stride
                                               (int((ksize_h - 1) / 2), int((ksize_w - 1) / 2))),
                                     # padding ksize_h - 1)/2 确保尺寸不因kenersize变化
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        self.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # convs
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        in_shape = scales[-1]  # 特征提取层最后输出尺寸
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth / 2 - i) <= h else [1]  # 确定经过depth/2次卷积后，最小尺寸不会低于2，但尽量减小（接近2） 因为底数是2
            stride = stride + [2] if 2 ** (depth / 2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])  # stride每个元素平方
        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,  # 这是第一次卷积操作
                                         tuple(conv_ksizes[0]),  # conv_ksizes[0]:[3,3]
                                         tuple(strides[0]),
                                         (int((conv_ksizes[0][0] - 1) / 2), int((conv_ksizes[0][1] - 1) / 2))),
                               # padding
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]
        for i in range(1, int(depth / 2)):  # 剩下的 int(depth / 2) - 1 次卷积，只有第一次卷积改变输入通道数，后面的卷积操作就不需要改变通道数
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                 tuple(conv_ksizes[i]),
                                                 tuple(strides[i]),
                                                 (int((conv_ksizes[i][0] - 1) / 2), int((conv_ksizes[i][1] - 1) / 2))),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        # deconvs
        deconvs = []
        for i in range(1, int(depth / 2)):  # 前int(depth / 2) - 1 次逆卷积
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,  # 逆卷积不改变通道数
                                                            tuple(deconv_ksizes[int(depth / 2) - i]),
                                                            # 逆卷积核的大小是卷积操作stride的平方，别问我为什么，经验
                                                            tuple(strides[int(depth / 2) - i]),  # stride与正卷积相反，见论文图
                                                            (int(deconv_ksizes[int(depth / 2) - i][0] / 4.),
                                                             int(deconv_ksizes[int(depth / 2) - i][1] / 4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,  # 最后一次逆卷积从 num_channels到maxT
                                                        tuple(deconv_ksizes[0]),
                                                        tuple(strides[0]),
                                                        (int(deconv_ksizes[0][0] / 4.), int(deconv_ksizes[0][1] / 4.))),
                                     nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        x = input[0]
        for i in range(0, len(self.fpn)):
            x = self.fpn[i](x) + input[i + 1]  # 这是CAM最上面的残差相加
        conv_feats = []
        for i in range(0, len(self.convs)):
            x = self.convs[i](x)
            conv_feats.append(x)
        for i in range(0, len(self.deconvs) - 1):
            x = self.deconvs[i](x)
            x = x + conv_feats[
                len(conv_feats) - 2 - i]  # len(conv_feats) - 2 - i？      len(conv_feats)应该等于int(depth / 2)
            # 这里，并不是每个正卷积和反卷积都要对应相加，最里面的对应层不相加
        x = self.deconvs[-1](x)
        return x


class CAM_transposed(nn.Module):
    # In this version, the input channel is reduced to 1-D with sigmoid activation.
    # We found that this leads to faster convergence for 1-D recognition.
    def __init__(self, scales, maxT, depth, num_channels):
        super(CAM_transposed, self).__init__()
        # cascade multiscale features
        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i - 1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            assert not (scales[i - 1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
            ksize = [3, 3, 5]
            r_h, r_w = int(scales[i - 1][1] / scales[i][1]), int(scales[i - 1][2] / scales[i][2])
            ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
            ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][0],
                                               (ksize_h, ksize_w),
                                               (r_h, r_w),
                                               (int((ksize_h - 1) / 2), int((ksize_w - 1) / 2))),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        fpn.append(nn.Sequential(nn.Conv2d(scales[i][0], 1,
                                           (1, ksize_w),
                                           (1, r_w),
                                           (0, int((ksize_w - 1) / 2))),
                                 nn.Sigmoid()))
        self.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # deconvs
        in_shape = scales[-1]
        deconvs = []
        ksize_h = 1 if in_shape[1] == 1 else 4
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                            (ksize_h, 4),
                                                            (r_h, 2),
                                                            (int(ksize_h / 4.), 1)),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
                                                        (ksize_h, 4),
                                                        (r_h, 2),
                                                        (int(ksize_h / 4.), 1)),
                                     nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, input):
        x = input[0]
        for i in range(0, len(self.fpn) - 1):
            x = self.fpn[i](x) + input[i + 1]
            # Reducing the input to 1-D form
        x = self.fpn[-1](x)
        # Transpose B-C-H-W to B-W-C-H
        x = x.permute(0, 3, 1, 2).contiguous()

        for i in range(0, len(self.deconvs)):
            x = self.deconvs[i](x)
        return x


'''
Decoupled Text Decoder
'''


class DTD(nn.Module):
    # LSTM DTD
    def __init__(self, nclass, nchannel, dropout=0.3):
        super(DTD, self).__init__()
        self.nclass = nclass
        self.nchannel = nchannel
        self.pre_lstm = nn.LSTM(nchannel, int(nchannel / 2), bidirectional=True)  # inputsize，hiddensize（输出维度）
        self.rnn = nn.GRUCell(nchannel * 2, nchannel)
        self.generator = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(nchannel, nclass)
        )
        self.char_embeddings = Parameter(torch.randn(nclass, nchannel))

    # def forward(self, feature, A, text, text_length,
    #             test=False):  # feature: feature map   A: attention map  text:ture labels
    def forward(self, feature, A, text, text_length, current_epoch, test=False):  # feature: feature map   A: attention map  text:ture labels #尝试修改处4 加了current_epoch
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]  # nT应该就是输入文本的长度
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1,
                                               1)  # attention map 归一化 -1 - 1   把A的最后两个向量相乘得到一个行向量，变成3维，再把刚刚的最后一维向量相加
        # 等于说，把一通道的 H*W打平成一维再求和，最后再sum值reshape成二维[1,1]
        # weighted sum
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH,
                                                     nW)  # scaled dot product and sum with feature: feature map   A: attention map
        # 张量内积要求： 1. 张量的维度数必须相同，即5维*5维   2. 如果某一个维度 数值不同，只能一个1个张量中是1，另一个随意，比如上例中nC和nT不同，为了相乘，就在各自维度上插入一个1维，值为1，输出的维度值为张量中大的那个
        C = C.view(nB, nT, nC, -1).sum(3).transpose(1, 0)  # B: [B,T,C,H,W] -> [B,T,C,H*W] -> [B,T,C,1] -> [T,B,C,1]
        C, _ = self.pre_lstm(
            C)  # inputsize: C   [T,B,C,1]   nT:timestep  nB:batch  nC:vec size   输出C就是out向量：[2T,B,int(nchannel / 2),1]
        C = F.dropout(C, p=0.3, training=self.training)
        if not test:
            lenText = int(text_length.sum())
            nsteps = int(text_length.max())

            gru_res = torch.zeros(C.size()).type_as(C.data)
            out_res = torch.zeros(lenText, self.nclass).type_as(
                feature.data)  # [行：所有的单词长度，列：所有单词的种类] 所以输出就是每行代表该单词的输出概率
            out_attns = torch.zeros(lenText, nH, nW).type_as(A.data)  # 每个单词一个注意力矩阵

            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))
            # for i in range(0, nsteps):
            #     hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim = 1), hidden)
            #     gru_res[i, :, :] = hidden
            #     prev_emb = self.char_embeddings.index_select(0, text[:, i])
            # gru_res = self.generator(gru_res)

            # for i in range(0, nsteps):  # schedule simpling
            #     # if
            #     hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim=1), hidden)  # pre_emb:上一个真实的标签
            #     gru_res[i, :, :] = hidden
            #     prev_emb = self.char_embeddings.index_select(0, text[:, i])  # 选择真实label作为下一节点输入
            # gru_res = self.generator(gru_res)


            for i in range(0, nsteps):     #schedule simpling
                if random.randint(1,500) > current_epoch :      #linear probability
                    hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim = 1), hidden)    #pre_emb:上一个真实的标签
                    gru_res[i, :, :] = hidden
                    prev_emb = self.char_embeddings.index_select(0, text[:, i])     #选择真实label作为下一节点输入
                else:
                    hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim=1),
                                      hidden)
                    gru_res[i, :, :] = hidden                   #gru_res用于保存DTD的输出（未解码，是矩阵），原本用真实值作为输入时，每个step的结果全部放进gru_res，
                                                                #统一用Attention_AR_counter解码，计算loss,但要用计划采样，必须立即解码，作为下一阶段的输入，但是仍要存进gru_res,用于计算loss
                    tmp_result = self.generator(hidden)         #输出模型预测字符对应概率
                    tmp_result = tmp_result.topk(1)[1].squeeze()    #选出概率最大的字符
                    prev_emb = self.char_embeddings.index_select(0, tmp_result)  #对上面的字符编码
            gru_res = self.generator(gru_res)

            start = 0
            for i in range(0, nB):
                cur_length = int(text_length[i])
                out_res[start: start + cur_length] = gru_res[0: cur_length, i, :]
                out_attns[start: start + cur_length] = A[i, 0:cur_length, :, :]
                start += cur_length

            return out_res, out_attns

        else:
            lenText = nT
            nsteps = nT
            out_res = torch.zeros(lenText, nB, self.nclass).type_as(feature.data)
            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))
            out_length = torch.zeros(nB)
            # print("out_length: ",out_length,type(out_length))
            now_step = 0
            while 0 in out_length and now_step < nsteps:
                hidden = self.rnn(torch.cat((C[now_step, :, :], prev_emb), dim=1),
                                  hidden)
                tmp_result = self.generator(hidden)
                out_res[now_step] = tmp_result
                # print("tmp_result: before ", tmp_result, type(tmp_result))
                tmp_result = tmp_result.topk(1)[1].squeeze()
                # print("tmp_result: ",tmp_result, type(tmp_result))
                for j in range(nB):
                    if out_length[j] == 0 and tmp_result[j] == 0:  # out_length:每个batch中句子的长度，初始都为0，有输出后更新
                        out_length[j] = now_step + 1
                prev_emb = self.char_embeddings.index_select(0, tmp_result)
                now_step += 1
            for j in range(0, nB):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps

            start = 0
            output = torch.zeros(int(out_length.sum()), self.nclass).type_as(feature.data)
            for i in range(0, nB):
                cur_length = int(out_length[i])
                output[start: start + cur_length] = out_res[0: cur_length, i, :]
                start += cur_length

            return output, out_length 
