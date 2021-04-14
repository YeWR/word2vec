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

    def forward(self, cent_word, pos_word, neg_word):

        cent_word = self.u_embedding_matrix(cent_word).transpose(1, 2)
        pos_word = self.v_embedding_matrix(pos_word)
        neg_word = self.v_embedding_matrix(neg_word)

        pred_pos = torch.bmm(pos_word, cent_word).squeeze()
        pred_neg = torch.bmm(neg_word, cent_word).squeeze()

        loss_pos = F.logsigmoid(pred_pos).sum()
        loss_neg = F.logsigmoid(-pred_neg).sum()
        loss = loss_pos + loss_neg
        return -loss

    def inference(self, word):
        embed = self.u_embedding_matrix(word)
        return embed
