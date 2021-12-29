from main import *
from config import *
import torch.nn as nn


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 网络主要结构
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim,
                            num_layers=Config.num_layers)
        # 进行分类
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        # print(input.shape)
        # 初始化
        if hidden is None:
            h_0 = input.data.new(Config.num_layers, batch_size,
                                 self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.num_layers, batch_size,
                                 self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 输入 序列长度 * batch(每个汉字是一个数字下标)，
        # 输出 序列长度 * batch * 向量维度
        # input是一个序列，通过embedding，对每一个词的int变成一个词向量，所以对整体input进行embedding，会得到一个矩阵
        embeds = self.embeddings(input)
        # 输出hidden的大小： 序列长度 * batch * hidden_dim
        # 将得到的矩阵交给lstm，得到一个向量序列。出来之后得到的是所有隐含层的输出，依次排列
        output, hidden = self.lstm(embeds, (h_0, c_0))
        # 然后进行分类计算
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden
