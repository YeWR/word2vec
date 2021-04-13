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
        batch_size, pred_size = pos_neg_v.shape

        embed_u = self.u_embedding_matrix(pos_u)
        embed_v = self.v_embedding_matrix(pos_neg_v)

        label = torch.zeros(batch_size).to(pos_u.device).long()
        pred = embed_u.bmm(embed_v.transpose(1, 2)).squeeze()

        loss = F.cross_entropy(pred, label)
        return loss

    def inference(self, pos):
        embed_u = self.u_embedding_matrix(pos)
        return embed_u
