import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_dim):
        super(SkipGramModel, self).__init__()
        initrange = 0.5 / embedding_dim
        self.u_embedding_matrix = nn.Embedding(vocab_dim, embedding_dim)
        self.u_embedding_matrix.weight.data.uniform_(-initrange, initrange)
        self.v_embedding_matrix = nn.Embedding(vocab_dim, embedding_dim)
        self.v_embedding_matrix.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_neg_v):

        embed_u = self.u_embedding_matrix(pos_u)
        embed_v = self.v_embedding_matrix(pos_neg_v)

        pred = embed_u.bmm(embed_v.transpose(1, 2)).squeeze()

        loss_pos = torch.log(torch.nn.Sigmoid()(pred[:, 0])).sum()
        loss_neg = -torch.log(torch.nn.Sigmoid()(pred[:, 1:])).sum()
        loss = loss_pos + loss_neg
        return loss

    def inference(self, pos):
        embed_u = self.u_embedding_matrix(pos)
        return embed_u
