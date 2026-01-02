""" CLIP Model
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from collections import OrderedDict# 导入有序字典，它可以记住元素插入的顺序
import logging
import math
import os
from typing import List, Tuple, Union#导入类型检查库，可以定义变量或函数的类型
import hashlib#导入哈希库，用于生成哈希值
import urllib#导入URL处理模块，可以用来获取网页数据
from tqdm import tqdm#导入进度条模块，可以在终端显示进度条
import warnings#导入警告模块，可以在终端显示警告信息
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger("IRRA.model")#创建一个日志记录器，名为"IRRA.model"，用于记录程序运行过程中的信息

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}#定义一个字典，包含CLIP模型的名称和下载地址

def available_models() -> List[str]:#定义一个函数，函数名为available_models，返回值类型为字符串列表
    """Returns the names of available CLIP models
    该函数返回可用的CLIP模型的名称
    """
    return list(_MODELS.keys())#返回_MODELS字典的所有键，即所有可用的CLIP模型的名称

def _download(url: str, root: str):#定义一个函数，函数名为_download，参数为url和root，返回值类型为字符串
    '''
    这个函数的主要用途是下载预训练的深度学习模型。这些模型通常很大，直接下载可能会花费很长时间，因此这个函数使用了tqdm库来显示下载进度。
    此外，为了确保下载的文件没有被篡改，这个函数还会检查下载的文件的SHA256哈希值是否与预期的哈希值匹配。如果不匹配，函数会抛出一个异常。
    '''
    os.makedirs(root, exist_ok=True)#创建一个目录，如果目录已存在则不会报错
    filename = os.path.basename(url)#从url中获取文件名

    expected_sha256 = url.split("/")[-2]#从url中获取sha256值
    download_target = os.path.join(root, filename)#将目录和文件名拼接成下载目标路径

    if os.path.exists(download_target) and not os.path.isfile(download_target):#如果下载目标路径存在且不是一个文件
        raise RuntimeError(f"{download_target} exists and is not a regular file")#抛出一个运行时错误

    if os.path.isfile(download_target):#如果下载目标路径存在且是一个文件
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:#如果文件的SHA256哈希值与预期的哈希值匹配，则返回文件路径
            return download_target#返回文件路径
        else:#如果文件的SHA256哈希值与预期的哈希值不匹配，则发出警告并重新下载文件
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:#打开url和目标文件
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:#创建一个进度条
            while True:#循环读取源文件并写入目标文件，直到源文件读取完毕
                buffer = source.read(8192)#每次读取8192字节
                if not buffer:#如果读取到文件末尾，则退出循环
                    break

                output.write(buffer)#将读取到的数据写入目标文件
                loop.update(len(buffer))#更新进度条

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:#如果下载后的文件的SHA256哈希值与预期的哈希值不匹配，则抛出异常
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target#返回下载的文件的路径


class Bottleneck(nn.Module):
    '''
    定义了瓶颈结构,具体来说，输入数据首先通过三个卷积层，每个卷积层后都跟着一个批标准化层和ReLU激活函数。
    然后，如果步长大于1或输入通道数不等于输出通道数，输入数据会经过一个降采样层。最后，降采样后的数据（或原始的输入数据）会被加到卷积后的结果上，实现残差连接
    '''
    expansion = 4#类变量，用于指定扩展系数，即输出通道数相对于输入通道数的倍数

    def __init__(self, inplanes, planes, stride=1):#构造函数，接受三个参数：输入通道数、输出通道数和步长
        super().__init__()#调用父类的构造函数

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # 所有卷积层的步长都为1。当步长大于1时，在第二个卷积后执行平均池化
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)## 第一个卷积层，卷积核大小为1，步长为1，不使用偏置
        self.bn1 = nn.BatchNorm2d(planes)#第一个批标准化层，用于对卷积的输出进行标准化处理

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)#第二个卷积层，卷积核大小为3，步长为1，不使用偏置，使用填充以保持空间尺寸
        self.bn2 = nn.BatchNorm2d(planes)## 第二个批标准化层，用于对卷积的输出进行标准化处理

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()# 平均池化层，当步长大于1时使用，否则为恒等映射

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)#第三个卷积层，卷积核大小为1，步长为1，不使用偏置，输出通道数为输入通道数的expansion倍
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)#第三个批标准化层，用于对卷积的输出进行标准化处理

        self.relu = nn.ReLU(inplace=True)#ReLU激活函数，inplace=True表示直接修改输入，而不是创建新的输出.使用原地计算，即直接在原来的内存上进行计算，不再开辟新的内存
        self.downsample = None#降采样层，初始为None
        self.stride = stride#步长

        if stride > 1 or inplanes != planes * Bottleneck.expansion:#如果步长大于1或者输入通道数不等于输出通道数的expansion倍,那么需要降采样
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            # 降采样层在前面加上一个平均池化层，后面的卷积层步长为1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))#降采样层，包含一个平均池化层、一个卷积层和一个批标准化层

    def forward(self, x: torch.Tensor):#定义前向传播函数，接受一个参数x，类型为torch.Tensor
        identity = x#将输入x赋值给identity，用于后续的残差连接

        out = self.relu(self.bn1(self.conv1(x)))#第一层：卷积 -> 批标准化 -> ReLU激活
        out = self.relu(self.bn2(self.conv2(out)))#第二层：卷积 -> 批标准化 -> ReLU激活
        out = self.avgpool(out)#对第二层的输出进行平均池化
        out = self.bn3(self.conv3(out))#第三层：卷积 -> 批标准化

        if self.downsample is not None:#如果存在降采样层
            identity = self.downsample(x)#对输入x进行降采样

        out += identity#降采样后的x（或原始的x）加到out上，实现残差连接
        out = self.relu(out)#对结果进行ReLU激活
        return out


class AttentionPool2d(nn.Module):
    '''
    这个类定义了一个注意力池化层，用于对输入进行注意力池化。具体来说，输入首先通过一个线性层，然后通过一个多头注意力层，最后通过一个线性层，得到注意力池化的结果
    '''
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):#构造函数，接受四个参数：空间维度、嵌入维度、头数和输出维度
        super().__init__()#调用父类的构造函数
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.positional_embedding = nn.Parameter(torch.randn((spacial_dim[0] * spacial_dim[1]) + 1, embed_dim)/ embed_dim ** 0.5)#位置嵌入，用于对位置信息进行编码
        self.k_proj = nn.Linear(embed_dim, embed_dim)#线性层，用于计算注意力机制中的键
        self.q_proj = nn.Linear(embed_dim, embed_dim)#线性层，用于计算注意力机制中的查询
        self.v_proj = nn.Linear(embed_dim, embed_dim)#线性层，用于计算注意力机制中的值
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)#线性层，用于计算注意力机制的输出
        self.num_heads = num_heads#头数

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC#将输入x进行形状变换，将空间维度合并到通道维度上.将输入x从NCHW格式转换为(HW)NC格式
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC#在x的第0维上添加x的均值
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC#将位置嵌入添加到x上
        x, _ = F.multi_head_attention_forward(#调用多头注意力函数
            query=x, key=x, value=x,#查询、键和值都为x
            embed_dim_to_check=x.shape[-1],#嵌入维度为x的最后一维
            num_heads=self.num_heads,#头数
            q_proj_weight=self.q_proj.weight,#查询的权重
            k_proj_weight=self.k_proj.weight,#键的权重
            v_proj_weight=self.v_proj.weight,#值的权重
            in_proj_weight=None,#输入的权重
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),# 输入的投影偏置，为查询、键和值的偏置的拼接
            bias_k=None,#键的偏置，这里不使用
            bias_v=None,#值的偏置，这里不使用
            add_zero_attn=False,#是否添加零注意力，这里不添加
            dropout_p=0,#Dropout概率，这里为0，即不使用Dropout
            out_proj_weight=self.c_proj.weight,#输出的投影权重
            out_proj_bias=self.c_proj.bias,#输出的投影偏置
            use_separate_proj_weight=True,#是否使用单独的投影权重，这里使用
            training=self.training,#是否处于训练模式
            need_weights=False#是否需要返回注意力权重，这里不需要
        )

        return x[0]#返回结果的第0个元素，即注意力池化的结果


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    这个类的主要用途是实现修改后的ResNet模型。与原始的ResNet模型相比，它有以下几个改变：
    现在有3个"stem"卷积，而不是1个，使用平均池化而不是最大池化。
    执行抗混叠的跨步卷积，当步长大于1时，在卷积前面添加一个avgpool。
    最后的池化层是一个QKV注意力，而不是一个平均池化
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):#构造函数，接受五个参数：层数、输出维度、注意力头数、输入分辨率和宽度
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem#3层stem卷积
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)#第一层卷积，卷积核大小为3，步长为2，使用填充以保持空间尺寸，不使用偏置
        self.bn1 = nn.BatchNorm2d(width // 2)#第一层批标准化，用于对卷积的输出进行标准化处理
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)#第二层卷积，卷积核大小为3，步长为1，使用填充以保持空间尺寸，不使用偏置
        self.bn2 = nn.BatchNorm2d(width // 2)#第二层批标准化
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)#第三层卷积，卷积核大小为3，步长为1，使用填充以保持空间尺寸，不使用偏置
        self.bn3 = nn.BatchNorm2d(width)#第三层批标准化
        self.avgpool = nn.AvgPool2d(2)#平均池化层，池化核大小为2
        self.relu = nn.ReLU(inplace=True)#ReLU激活函数，inplace=True表示直接修改输入，而不是创建新的输出

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction#这是一个在构造过程中使用的可变变量
        self.layer1 = self._make_layer(width, layers[0])#第一组残差层
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)#第二组残差层，步长为2
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)#第三组残差层，步长为2
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)#第四组残差层，步长为2

        embed_dim = width * 32  # the ResNet feature dimension#ResNet特征维度
        spacial_dim = (
            input_resolution[0] // 32,
            input_resolution[1] // 32,
        )#空间维度，即输入分辨率除以32
        self.attnpool = AttentionPool2d(spacial_dim, embed_dim, heads, output_dim)#注意力池化层

    def _make_layer(self, planes, blocks, stride=1):#定义_make_layer方法，接受三个参数：planes（输出通道数）、blocks（残差块的数量）和stride（步长）
        '''
        这个方法用于构建残差层，具体来说，它会创建一个列表，包含blocks个Bottleneck实例，然后返回一个顺序容器，包含所有的Bottleneck实例
        '''
        layers = [Bottleneck(self._inplanes, planes, stride)]#创建一个列表，包含一个Bottleneck实例

        self._inplanes = planes * Bottleneck.expansion#更新_inplanes的值，为输出通道数乘以Bottleneck的扩展系数
        for _ in range(1, blocks):#对于剩余的每一个残差块
            layers.append(Bottleneck(self._inplanes, planes))#向列表中添加一个Bottleneck实例

        return nn.Sequential(*layers)#返回一个顺序容器，包含所有的Bottleneck实例

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:#对于每一对卷积层和批标准化层
                x = self.relu(bn(conv(x)))#对x进行卷积、批标准化和ReLU激活
            x = self.avgpool(x)#对x进行平均池化
            return x#返回x

        x = x.type(self.conv1.weight.dtype)#将x的数据类型转换为第一层卷积的权重的数据类型
        x = stem(x)#对x进行stem操作
        x = self.layer1(x)#对x进行第一组残差层操作
        x = self.layer2(x)#对x进行第二组残差层操作
        x = self.layer3(x)#对x进行第三组残差层操作
        x = self.layer4(x)#对x进行第四组残差层操作
        x = self.attnpool(x)#对x进行注意力池化操作

        return x#返回x


class LayerNorm(nn.LayerNorm):#定义一个类，继承自nn.LayerNorm
    """Subclass torch's LayerNorm to handle fp16.
    扩展 PyTorch 的 LayerNorm 类以处理半精度浮点数（fp16）。
    """

    def forward(self, x: torch.Tensor):#定义了 LayerNorm 类的前向传播函数，该函数接受一个类型为 torch.Tensor 的参数 x
        orig_type = x.dtype#保存x的数据类型
        ret = super().forward(x.type(torch.float32))#首先将输入张量 x 的数据类型转换为 torch.float32，然后调用父类 nn.LayerNorm 的前向传播函数，并将结果保存在 ret 中。
        return ret.type(orig_type)#将 ret 的数据类型转换回 x 的原始数据类型，并返回结果。


class QuickGELU(nn.Module):
    '''
    这个类用于快速近似计算 GELU（高斯误差线性单元）激活函数。
    '''
    def forward(self, x: torch.Tensor):#定义了 QuickGELU 类的前向传播函数，该函数接受一个类型为 torch.Tensor 的参数 x。
        return x * torch.sigmoid(1.702 * x)#计算了 x 和 x 的 sigmoid 函数的乘积，并返回结果。这是一种快速近似计算 GELU（高斯误差线性单元）激活函数的方法。


class ResidualAttentionBlock(nn.Module):
    '''
    这个类定义了残差注意力块，用于对输入进行残差注意力处理。具体来说，它包含一个多头注意力模块和一个多层感知机（MLP）。
    '''
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):#定义了 ResidualAttentionBlock 类的初始化函数，该函数接受三个参数：d_model（模型的维度），n_head（注意力头的数量），和 attn_mask（注意力掩码）。
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)#创建了一个多头注意力模块，并将其保存在 self.attn 中。
        self.ln_1 = LayerNorm(d_model)#创建了一个层归一化模块，并将其保存在 self.ln_1 中。
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))#创建了一个多层感知机（MLP），并将其保存在 self.mlp 中。这个 MLP 包含一个线性层、一个 GELU 激活函数，和另一个线性层。
        self.ln_2 = LayerNorm(d_model)#建了另一个层归一化模块，并将其保存在 self.ln_2 中。
        self.attn_mask = attn_mask#将输入的注意力掩码保存在 self.attn_mask 中

    def attention(self, x: torch.Tensor):#定义了 ResidualAttentionBlock 类的 attention 方法，该方法接受一个类型为 torch.Tensor 的参数 x。
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None#将 self.attn_mask 的数据类型和设备设置为与输入张量 x 相同。
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]#调用 self.attn 的前向传播函数，并返回结果的第一个元素。

    def forward(self, x: torch.Tensor):#义了 ResidualAttentionBlock 类的前向传播函数，该函数接受一个类型为 torch.Tensor 的参数 x
        x = x + self.attention(self.ln_1(x))#首先对 x 进行层归一化，然后计算注意力，最后将结果与原始的 x 相加
        x = x + self.mlp(self.ln_2(x))#首先对 x 进行层归一化，然后通过 MLP，最后将结果与原始的 x 相加。
        return x


class Transformer(nn.Module):#定义了一个名为 Transformer 的类，该类继承自 PyTorch 的 nn.Module 类。
    '''
    这个类定义了 Transformer 模型，用于对输入进行 Transformer 处理。具体来说，它包含 layers 个 ResidualAttentionBlock。
    '''
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):#初始化函数，该函数接受四个参数：width（模型的宽度），layers（层数），heads（注意力头的数量），和 attn_mask（注意力掩码）
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])#创建了一个序列，该序列包含 layers 个 ResidualAttentionBlock，并将其保存在 self.resblocks 中

    def forward(self, x: torch.Tensor):#定义了 Transformer 类的前向传播函数，该函数接受一个类型为 torch.Tensor 的参数 x
        return self.resblocks(x)#返回 self.resblocks 的前向传播函数的结果


class VisionTransformer(nn.Module):
    '''
    这个类定义了 Vision Transformer 模型，用于对输入进行 Vision Transformer 处理。该函数首先将输入张量通过卷积层、重塑操作、置换操作、拼接操作、相加操作、层归一化模块、Transformer 模型、另一个层归一化模块，和一个条件矩阵乘法操作，然后返回最终的结果。
    '''
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        #初始化函数，该函数接受七个参数：input_resolution（输入分辨率），patch_size（块大小），stride_size（步长大小），width（模型的宽度），layers（层数），heads（注意力头的数量），和 output_dim（输出维度）。
        super().__init__()
        self.input_resolution = input_resolution # (384, 128)#实例化输入分辨率
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1#计算了在 x 方向上的块数量，并将结果保存在 self.num_x 中。
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1#计算了在 y 方向上的块数量，并将结果保存在 self.num_y 中。
        num_patches = self.num_x * self.num_y#计算了总的块数量，并将结果保存在 num_patches 中

        self.output_dim = output_dim#实例化输出维度
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)#创建了一个卷积层，并将其保存在 self.conv1 中

        scale = width ** -0.5 # 1/sqrt(768)#计算了缩放因子，并将结果保存在 scale 中
        self.class_embedding = nn.Parameter(scale * torch.randn(width))#创建了一个类别嵌入，并将其保存在 self.class_embedding 中。
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))#创建了一个位置嵌入，并将其保存在 self.positional_embedding 中。
        self.ln_pre = LayerNorm(width)#创建了一个层归一化模块，并将其保存在 self.ln_pre 中。

        self.transformer = Transformer(width, layers, heads)#创建了一个 Transformer 模型，并将其保存在 self.transformer 中。

        self.ln_post = LayerNorm(width)#建了另一个层归一化模块，并将其保存在 self.ln_post 中。
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))#创建了一个投影矩阵，并将其保存在 self.proj 中


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]#将输入张量 x 通过卷积层 self.conv1，并将结果保存在 x 中。
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]#将 x 的形状从 [*, width, grid, grid] 转换为 [*, width, grid ** 2]，并将结果保存在 x 中。
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]#将 x 的形状从 [*, width, grid ** 2] 转换为 [*, grid ** 2, width]，并将结果保存在 x 中。
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]#嵌入 self.class_embedding 和 x 进行拼接，并将结果保存在 x 中。
        x = x + self.positional_embedding.to(x.dtype)#将 x 和位置嵌入 self.positional_embedding 进行相加，并将结果保存在 x 中
        x = self.ln_pre(x)#将 x 通过层归一化模块 self.ln_pre，并将结果保存在 x 中。

        x = x.permute(1, 0, 2)  # NLD -> LND将 x 的维度进行置换，并将结果保存在 x 中
        x = self.transformer(x)#将 x 通过 Transformer 模型 self.transformer，并将结果保存在 x 中。
        x = x.permute(1, 0, 2)  # LND -> NLD#将 x 的维度进行置换，并将结果保存在 x 中

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)#将 x 通过层归一化模块 self.ln_post，并将结果保存在 x 中。

        if self.proj is not None:#检查 self.proj 是否为 None，如果不是，则将 x 和 self.proj 进行矩阵乘法，并将结果保存在 x 中
            x = x @ self.proj
    
        return x



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: Union[int, Tuple[int, int]],
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        '''
        定义了一个 CLIP 模型，该模型包含一个视觉模块（可以是 ModifiedResNet 或 VisionTransformer）、一个 Transformer 模块、一个嵌入层、一个位置嵌入、一个层归一化模块和一个文本投影。
        '''
        #定义了 CLIP 类的初始化函数，该函数接受多个参数，包括嵌入维度、图像分辨率、视觉层参数、视觉宽度、视觉块大小、步长大小、上下文长度、词汇表大小、Transformer 宽度、Transformer 头的数量和 Transformer 层的数量。
        super().__init__()

        self.context_length = context_length#实例化输入的上下文长度

        if isinstance(vision_layers, (tuple, list)):#检查 vision_layers 是否为元组或列表。如果是，那么它将创建一个 ModifiedResNet 对象，并将其保存在 self.visual 中。
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:#如果 vision_layers 不是元组或列表，那么它将创建一个 VisionTransformer 对象，并将其保存在 self.visual 中。
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )#创建了一个 Transformer 对象，并将其保存在 self.transformer 中。

        self.vocab_size = vocab_size#实例化输入的词汇表大小
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)#创建了一个嵌入层，并将其保存在 self.token_embedding 中。
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))#创建了一个位置嵌入，并将其保存在 self.positional_embedding 中
        self.ln_final = LayerNorm(transformer_width)#创建了一个层归一化模块，并将其保存在 self.ln_final 中。

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))#创建了一个文本投影，并将其保存在 self.text_projection 中。
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()#调用了 initialize_parameters 方法，用于初始化模型的参数。

    def initialize_parameters(self):
        '''
        这个方法使用各种正态分布初始化模型的参数。
        '''
        nn.init.normal_(self.token_embedding.weight, std=0.02)#使用标准差为 0.02 的正态分布初始化 self.token_embedding.weight
        nn.init.normal_(self.positional_embedding, std=0.01)#使用标准差为 0.01 的正态分布初始化 self.positional_embedding。

        if isinstance(self.visual, ModifiedResNet):#检查 self.visual 是否是 ModifiedResNet 的实例
            if self.visual.attnpool is not None:#如果 self.visual.attnpool 不为 None，这段代码会计算标准差 std，并使用这个标准差的正态分布初始化 self.visual.attnpool 的四个权重
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:#遍历 self.visual 的四个层，并将名字以 “bn3.weight” 结尾的参数初始化为零
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)#算了三个标准差 proj_std、attn_std 和 fc_std
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:#遍历 self.transformer.resblocks 的每个块，并使用对应的标准差的正态分布初始化每个块的权重。
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:#如果 self.text_projection 不为 None，这段代码会使用标准差为 self.transformer.width ** -0.5 的正态分布初始化 self.text_projection。
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        '''
        这个方法用于构建注意力掩码，用于对输入进行注意力处理。
        '''
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)#创建一个空的张量 mask，形状为 [self.context_length, self.context_length]
        mask.fill_(float("-inf"))#将 mask 中的所有元素填充为负无穷
        mask.triu_(1)  # zero out the lower diagonal#将 mask 的下三角部分填充为 0
        return mask

    @property#使用 @property 装饰器，将 encode_image 和 encode_text 方法转换为属性
    def dtype(self):#定义了 CLIP 类的 dtype 属性
        return self.visual.conv1.weight.dtype#返回 self.visual.conv1.weight 的数据类型

    def encode_image(self, image):#定义了 CLIP 类的 encode_image 方法，该方法接受一个参数 image。
        return self.visual(image.type(self.dtype))#将 image 的数据类型转换为 self.dtype，然后通过 self.visual，并返回结果

    def encode_text(self, text):#定义了 CLIP 类的 encode_text 方法，该方法接受一个参数 text。
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]#将 text 通过 self.token_embedding，将结果的数据类型转换为 self.dtype，并将结果保存在 x 中

        x = x + self.positional_embedding.type(self.dtype)#将 x 和 self.positional_embedding 的数据类型转换为 self.dtype 的结果进行相加，并将结果保存在 x 中
        x = x.permute(1, 0, 2)  # NLD -> LND#将 x 的维度进行置换，并将结果保存在 x 中
        x = self.transformer(x)#将 x 通过 self.transformer，并将结果保存在 x 中。
        x = x.permute(1, 0, 2)  # LND -> NLD#将 x 的维度进行置换，并将结果保存在 x 中
        x = self.ln_final(x).type(self.dtype)#将 x 通过 self.ln_final，将结果的数据类型转换为 self.dtype，并将结果保存在 x 中。

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection#将 x 和 self.text_projection 进行矩阵乘法，并将结果保存在 x 中

        return x

    def forward(self, image, text):#定义了 CLIP 类的 forward 方法，该方法接受两个参数：image 和 text。
        image_features = self.encode_image(image)#将 image 通过 encode_image 方法进行编码，并将结果保存在 image_features 中。
        text_features = self.encode_text(text)#将 text 通过 encode_text 方法进行编码，并将结果保存在 text_features 中。

        # # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text

        return image_features, text_features#返回 image_features 和 text_features
    
    
    def load_param(self, state_dict):#定义了 CLIP 类的 load_param 方法，该方法接受一个参数 state_dict
        '''
        这个方法用于加载预训练模型的参数。具体来说，它会将 state_dict 中的参数复制到 self.state_dict() 中。
        '''
        # 将pretrained_dict里不属于model_dict的键剔除掉
        param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}#创建了一个新的字典 param_dict，该字典只包含 state_dict 中键也在 self.state_dict() 中的键值对

        if 'model' in param_dict:#检查 param_dict 中是否有键为 ‘model’ 的键值对，如果有，则将 param_dict 更新为 param_dict['model']
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:#检查 param_dict 中是否有键为 ‘state_dict’ 的键值对，如果有，则将 param_dict 更新为 param_dict['state_dict']
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():#遍历 param_dict 中的每个键值对
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                #(对于图像)查键 k 是否等于 ‘visual.positional_embedding’，并且 v 的形状是否不等于 self.visual.positional_embedding 的形状。如果条件满足，则将 v 通过 resize_pos_embed 方法进行调整，并将结果保存在 v 中。
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:#(对于文本)查键 k 是否等于 ‘positional_embedding’，并且 v 的形状是否不等于 self.positional_embedding 的形状。如果条件满足，则将 v 通过 resize_text_pos_embed 方法进行调整，并将结果保存在 v 中
                v = resize_text_pos_embed(v, self.context_length)
            try:
                self.state_dict()[k].copy_(v)#将 v 复制到 self.state_dict()[k] 中。如果出现异常，则打印错误信息。
            except:
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
    


def resize_pos_embed(posemb, posemb_new, hight, width):#接受四个参数：posemb（旧的位置嵌入），posemb_new（新的位置嵌入），hight（新的高度），width（新的宽度）。
    '''
    这个函数用于调整位置嵌入的大小。
    '''
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0)#在第0维度增加一个维度。这是为了让posemb和posemb_new能够用于后续的操作。
    posemb_new = posemb_new.unsqueeze(0)

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]#将posemb分解为两部分：posemb_token（第一个位置嵌入）和posemb_grid（剩余的位置嵌入）

    gs_old = int(math.sqrt(len(posemb_grid)))#计算posemb_grid的长度的平方根，并将结果转换为整数。这是因为posemb_grid是一个二维网格，所以它的长度应该是一个完全平方数。
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))#打印一条消息，显示旧的和新的位置嵌入的大小，以及新的高度和宽度。
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)#将posemb_grid重塑为一个四维张量，并改变其维度的顺序。
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')#使用双线性插值将posemb_grid的大小调整为新的高度和宽度
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)#改变posemb_grid的维度的顺序，并将其重塑为一个三维张量
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)#将posemb_token和posemb_grid沿着第1维度连接起来
    return posemb.squeeze(0)#删除posemb的第0维度，并返回结果。这是因为我们在前面增加了一个维度，现在需要将其删除。最后，函数返回调整大小后的位置嵌入。


def convert_weights(model: nn.Module):#定义一个函数convert_weights，它接受一个参数：model（一个PyTorch模型）。
    """Convert applicable model parameters to fp16"""
    '''
    这个函数用于将模型的参数转换为fp16。
    '''

    def _convert_weights_to_fp16(l):#定义一个内部函数_convert_weights_to_fp16，它接受一个参数：l（一个层）。
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):#如果l是nn.Conv1d、nn.Conv2d或nn.Linear的实例，那么将其权重和偏置（如果存在）转换为fp16。
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):#如果l是nn.MultiheadAttention的实例，那么将其所有相关的权重和偏置（如果存在）转换为fp16。
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", "mcq_proj"]:#如果l有text_projection、proj或mcq_proj属性，那么将这些属性的值（如果存在）转换为fp16。
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)#对模型的每一层应用_convert_weights_to_fp16函数。这将遍历模型的所有层，并将适用的参数转换为fp16。最后，函数不返回任何值，但会修改传入的模型。这是因为PyTorch的张量是可变的，所以对张量的修改会影响原始模型。


def build_CLIP_from_openai_pretrained(name: str, image_size: Union[int, Tuple[int, int]], stride_size: int, jit: bool = False, download_root: str = None):
    #函数的定义，它接受五个参数：模型名称、图像大小、步长大小、是否使用JIT模型以及模型文件的下载路径。
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    
    image_size: Union[int, Tuple[int, int]]
        Input image size, in Re-ID task, image size commonly set to 384x128, instead of 224x224

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    '''
    这个函数用于加载和构建CLIP模型。它接受五个参数：模型名称、图像大小、步长大小、是否使用JIT模型以及模型文件的下载路径。它返回一个PyTorch模型。
    '''
    download_root = './'


    if name in _MODELS:#检查输入的模型名称是否在预定义的模型列表中。
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))#如果模型名称在列表中，这一行代码会下载模型文件，并返回模型文件的路径。
    elif os.path.isfile(name):#检查输入的模型名称是否是一个文件路径
        model_path = name#如果输入的模型名称是一个文件路径，这一行代码会直接使用这个路径。
    else:#如果输入的模型名称既不在预定义的模型列表中，也不是一个文件路径，那么就会抛出一个异常。
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:#尝试加载模型文件
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")#尝试使用Torch的JIT模型加载功能来加载模型
        state_dict = None#设置状态字典为None。状态字典通常用于保存和加载模型的参数。
    except RuntimeError:#这一行开始一个异常处理块，用于处理模型加载过程中可能出现的运行时错误
        # loading saved state dict
        if jit:#检查是否尝试加载JIT模型
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")# 如果尝试加载JIT模型失败，这一行代码会发出警告，并尝试以状态字典的方式加载模型。
            jit = False#将JIT标志设置为False
        state_dict = torch.load(model_path, map_location="cpu")#使用PyTorch的torch.load函数加载模型的状态字典。

    state_dict = state_dict or model.state_dict()#这一行代码确保状态字典不为空。如果前面的加载失败，它会尝试获取模型的当前状态字典。

    vit = "visual.proj" in state_dict#检查状态字典中是否包含"visual.proj"这个键，以判断模型是否是ViT模型

    if vit:#只有当模型是ViT模型时，才会执行这个块中的代码。
        vision_width = state_dict["visual.conv1.weight"].shape[0]#获取模型的视觉宽度。
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])#计算模型的视觉层的数量。
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]#获取模型的视觉块大小。
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)#计算模型的网格大小
        image_resolution = vision_patch_size * grid_size#计算模型的图像分辨率
    else:#如果模型不是ViT模型，那么就会执行这个块中的代码。
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]#计算每一层的唯一元素数量，并将结果保存在列表counts中。
        vision_layers = tuple(counts)#将计数转换为元组，作为视觉层的表示。
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]#获取模型的视觉宽度。
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)#计算模型的输出宽度。
        vision_patch_size = None#将视觉块大小设置为None。
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]#查输出宽度是否满足特定条件
        image_resolution = output_width * 32#计算模型的图像分辨率。

    embed_dim = state_dict["text_projection"].shape[1]#获取模型的嵌入维度。
    context_length = state_dict["positional_embedding"].shape[0]#获取上下文长度。
    vocab_size = state_dict["token_embedding.weight"].shape[0]#获取词汇表大小。
    transformer_width = state_dict["ln_final.weight"].shape[0]#获取Transformer的宽度。
    transformer_heads = transformer_width // 64#计算Transformer的头数。
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))#计算Transformer的层数。

    model_cfg = {
        'embed_dim': embed_dim,
        'image_resolution': image_resolution,
        'vision_layers': vision_layers, 
        'vision_width': vision_width, 
        'vision_patch_size': vision_patch_size,
        'context_length': context_length, 
        'vocab_size': vocab_size, 
        'transformer_width': transformer_width, 
        'transformer_heads': transformer_heads, 
        'transformer_layers': transformer_layers
    }#创建一个字典，包含了模型的所有配置参数。


    # modify image resolution to adapt Re-ID task
    model_cfg['image_resolution'] = image_size#修改图像分辨率以适应Re-ID任务。
    model_cfg['stride_size'] = stride_size#设置步长大小。
    logger.info(f"Load pretrained {name} CLIP model with model config: {model_cfg}")#记录加载预训练模型的信息。
    model = CLIP(**model_cfg)#使用提取的参数创建一个新的CLIP模型。

    # covert model to fp16
    # convert_weights(model)

    # resize modified pos embedding
    model.load_param(state_dict)#加载模型的参数
    return model, model_cfg#返回创建的模型和模型的配置。


