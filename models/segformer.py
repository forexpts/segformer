"""
Segformerのモデル

デコーダ部分改造中

論文: https://arxiv.org/abs/2105.15203
参考: https://github.com/NVlabs/SegFormer
      https://ai-scholar.tech/articles/segmentation/voyager
      https://qiita.com/gensal/items/ab2f36e77c5ce1b55ecb
"""

from torch import nn
from einops import rearrange #テンソルの形状を変換しやすいライブラリ
import torch

class LayerNorm2d(nn.Module):
    #LayerNormで正規化する層
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
    
    def forward(self, x): #rearrangeでテンソルの形状を変換してから正規化
        x = rearrange(x, "a b c d -> a c d b")
        x = self.ln(x)
        x = rearrange(x, "a c d b -> a b c d")
        return x
    

class OverlappatchMerging(nn.Module):
    #畳み込みと正規化を行う層
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2)
        self.ln = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        return x


class MultiHeadAttention(nn.Module):
    #MultiHeadAttention
    #入力テンソルをヘッド数に分割し、それぞれでAttentionを計算し、結果を結合する
    #Attentionの計算は、QとKの内積を取り、その結果をソフトマックス関数に通す
    def __init__(self, channels, dim, head_num, reduction_ratio, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.r = reduction_ratio
        self.ln1 = LayerNorm2d(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.linear_reduceK = nn.Linear(channels * reduction_ratio, channels, bias = False)
        self.linear_reduceV = nn.Linear(channels * reduction_ratio, channels, bias = False)
        self.linear_Q = nn.Linear(dim, dim, bias = False)
        self.linear_K = nn.Linear(dim // reduction_ratio, dim // reduction_ratio, bias = False)
        self.linear_V = nn.Linear(dim // reduction_ratio, dim // reduction_ratio, bias = False)
        self.linear = nn.Linear(dim, dim, bias = False)
        self.soft = nn.Softmax(dim = 3)
        self.dropout = nn.Dropout(dropout)
    
    def split_head(self, x):
        x = torch.tensor_split(x, self.head_num, dim = 2)
        x = torch.stack(x, dim = 1)
        return x
    
    def concat_head(self, x):
        x = torch.tensor_split(x, x.size()[1], dim = 1)
        x = torch.concat(x, dim = 3).squeeze(dim = 1)
        return x
    
    def forward(self, x):
        _x = x
        x = self.ln1(x)
        x = rearrange(x, "a b c d -> a (c d) b")
        Q = K = V = x
        K = rearrange(K, "a (cd r) b -> a cd (b r)", r = self.r)
        V = rearrange(V, "a (cd r) b -> a cd (b r)", r = self.r)
        
        K = self.linear_reduceK(K)
        K = self.ln2(K)
        V = self.linear_reduceV(V)
        V = self.ln2(V)
        Q = rearrange(Q, "a cd br -> a br cd")
        K = rearrange(K, "a cd br -> a br cd")
        V = rearrange(V, "a cd br -> a br cd")

        Q = self.linear_Q(Q)
        K = self.linear_K(K)
        V = self.linear_V(V)
        
        #分割
        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)
        
        Q = rearrange(Q, "a h br cd -> a h cd br")
        K = rearrange(K, "a h br cd -> a h cd br")
        V = rearrange(V, "a h br cd -> a h cd br")

        #Attensionの計算
        QK = torch.matmul(Q, torch.transpose(K, 3, 2))
        QK = QK/((self.dim//self.head_num)**0.5)
        
        softmax_QK = self.soft(QK)
        softmax_QK = self.dropout(softmax_QK)

        QKV = torch.matmul(softmax_QK, V)

        #結合
        QKV = rearrange(QKV, "a h br cd -> a h cd br")
        QKV = self.concat_head(QKV)
        QKV = self.linear(QKV)

        QKV = rearrange(QKV, "a b (c d) -> a b c d", c = int(self.dim**0.5))
        QKV = QKV + _x
        return QKV


class MixFFN(nn.Module):
    #線形変換と畳み込み ※全結合層をConvで代用してるところに注意
    def __init__(self, in_channels, expantion_ratio):
        super().__init__()
        self.ln = LayerNorm2d(in_channels)
        self.linear1 = nn.Conv2d(in_channels, in_channels, kernel_size = 1)
        self.linear2 = nn.Conv2d(in_channels * expantion_ratio, in_channels, kernel_size = 1)
        self.conv = nn.Conv2d(in_channels, in_channels * expantion_ratio, kernel_size = 3, padding = "same")
        self.bn = nn.BatchNorm2d(in_channels * expantion_ratio)
        self.gelu = nn.GELU()

    def forward(self, x):
        _x = x
        x = self.ln(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = x + _x
        return x


class EncoderBlock1(nn.Module):
    #EncoderBlockの一つ目のブロック
    #OLMのとこミスってる…？
    #↑論文の画像がミスってるらしい: https://towardsdatascience.com/implementing-segformer-in-pytorch-8f4705e2ed0e
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_dim, head_num, reduction_ratio, expantion_ratio, enclayer_num):
        super().__init__()
        self.layer_num = enclayer_num
        self.OLM = OverlappatchMerging(in_channels, out_channels, kernel_size, stride)
        self.Enclayer = nn.ModuleList([nn.Sequential(
            MultiHeadAttention(out_channels, input_dim, head_num, reduction_ratio = 8),
            MixFFN(out_channels, expantion_ratio)
            ) 
        for _ in range(enclayer_num)])

    def forward(self, x):
        x = self.OLM(x) 
        for i in range(self.layer_num):
            x = self.Enclayer[i](x)
        return x


class EncoderBlock(nn.Module):
    #EncoderBlockの2目以降
    #(MultiHeadAttention -> MixFFN)^+ -> OverlappatchMerging
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_dim, head_num, reduction_ratio, expantion_ratio, enclayer_num):
        super().__init__()
        self.layer_num = enclayer_num
        self.Enclayer = nn.ModuleList([nn.Sequential(
            MultiHeadAttention(in_channels,input_dim,head_num, reduction_ratio = 8),
            MixFFN(in_channels = in_channels, expantion_ratio = expantion_ratio)
            ) 
        for _ in range(enclayer_num)])
        self.OLM = OverlappatchMerging(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.Enclayer[i](x)
        x = self.OLM(x)
        return x


class AllMLPDecoder(nn.Module):
    #DecoderのMLP
    #各EncoderBlockの出力を結合してからMLPに入力
    #逆畳み込みメイン？<-改善点な希ガス
    def __init__(self, l1_channels, l2_channels, l3_channels, l4_channels, class_num):
        super().__init__()
        scale = 4
        self.declayer1 = nn.Sequential(
            nn.Conv2d(l1_channels, 256, kernel_size = 1),
            nn.Upsample(scale_factor=1 * scale, mode="bilinear", align_corners=True)
        )
        self.declayer2 = nn.Sequential(
            nn.Conv2d(l2_channels, 256, kernel_size = 1),
            nn.Upsample(scale_factor=2 * scale, mode="bilinear", align_corners=True)
        )
        self.declayer3 = nn.Sequential(
            nn.Conv2d(l3_channels, 256, kernel_size = 1),
            nn.Upsample(scale_factor=4 * scale, mode="bilinear", align_corners=True)
        )
        self.declayer4 = nn.Sequential(
            nn.Conv2d(l4_channels, 256, kernel_size = 1),
            nn.Upsample(scale_factor=8 * scale, mode="bilinear", align_corners=True)
        )
        self.linear1 = nn.Conv2d(256 * 4, 256, kernel_size = 1)
        self.linear2 = nn.Conv2d(256, class_num, kernel_size = 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(256)

    def forward(self, x1, x2, x3, x4):#(32, 64, 160, 256, class_num = class_num)
        x1 = self.declayer1(x1)
        x2 = self.declayer2(x2)
        x3 = self.declayer3(x3)
        x4 = self.declayer4(x4)
        #x1, x2, x3, x4 = torch.Size([5, 256, 64, 64]) torch.Size([5, 256, 64, 64]) torch.Size([5, 256, 64, 64]) torch.Size([5, 256, 64, 64])
        x = torch.concat([x1, x2, x3, x4], dim = 1) #([5, 1024, 64, 64])
        x = self.linear1(x) # ([5, 256, 64, 64])
        x = self.relu(x) # ([5, 256, 64, 64])
        x = self.bn(x) # ([5, 256, 64, 64])
        x = self.linear2(x) # ([5, class_num, 64, 64])
        return x


class SegFormer(nn.Module):
    #SegFormerのモデル
    #入力画像からエンコード1回のx1, 2回のx2, 3回のx3, 4回のx4を出力
    #それらを結合してデコーダーMLPに入力
    def __init__(self, input_height, class_num):
        super().__init__()
        self.n_channels = 3
        self.n_classes = class_num
        self.bilinear = False
        self.EncBlock1 = EncoderBlock1(in_channels = 3,
                            out_channels = 32,
                            kernel_size = 7,
                            stride = 4, 
                            input_dim = (input_height//4)**2,
                            head_num = 1,
                            reduction_ratio = 4,   
                            expantion_ratio = 4,
                            enclayer_num = 2)
        self.EncBlock2 = EncoderBlock(in_channels = 32,
                            out_channels = 64,
                            kernel_size = 3,
                            stride = 2, 
                            input_dim = (input_height//4)**2,
                            head_num = 2,
                            reduction_ratio = 2,  
                            expantion_ratio = 8,
                            enclayer_num = 2)
        self.EncBlock3 = EncoderBlock(in_channels = 64,
                            out_channels = 160,
                            kernel_size = 3,
                            stride = 2, 
                            input_dim = (input_height//8)**2,
                            head_num = 4,
                            reduction_ratio = 1,   
                            expantion_ratio = 4,
                            enclayer_num = 2)
        self.EncBlock4 = EncoderBlock(in_channels = 160,
                            out_channels = 256,
                            kernel_size = 3,
                            stride = 2, 
                            input_dim = (input_height//16)**2,
                            head_num = 8,
                            reduction_ratio = 1,
                            expantion_ratio = 4,
                            enclayer_num = 2)
        self.Dec = AllMLPDecoder(32, 64, 160, 256, class_num = class_num)

    def forward(self, x):
        #エンコード 
        x1 = self.EncBlock1(x)
        x2 = self.EncBlock2(x1)
        x3 = self.EncBlock3(x2)
        x4 = self.EncBlock4(x3)
        
        #デコード
        x = self.Dec(x1, x2, x3, x4) # x1 = [32, 64,64] x2 = [64, 32,32] x3 = [160, 16,16] x4 = [256, 8,8]
        
        return x # [class_num, 64, 64] mask画像
