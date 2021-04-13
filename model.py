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

    def forward(self, pos_u, pos_v, neg_v):

        pos_u = self.u_embedding_matrix(pos_u).transpose(1, 2)
        pos_v = self.v_embedding_matrix(pos_v)
        neg_v = self.v_embedding_matrix(neg_v)

        pred_pos = torch.bmm(pos_v, pos_u).squeeze()
        pred_neg = torch.bmm(neg_v, pos_u).squeeze()

        loss_pos = F.logsigmoid(pred_pos).sum()
        loss_neg = F.logsigmoid(-pred_neg).sum()
        loss = loss_pos + loss_neg
        return -loss

    def inference(self, pos):
        embed_u = self.u_embedding_matrix(pos)
        return embed_u
