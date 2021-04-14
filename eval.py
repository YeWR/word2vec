import torch
from torch.utils.data import DataLoader
import argparse
import os
import json
import numpy as np
from model import SkipGramModel
from dataset import Word2VecDataset
from tqdm import tqdm
from torch import optim
from tensorboardX import SummaryWriter
from scipy.stats import pearsonr


def load_test_file(file_name, word2id):
    word1_lst = []
    word2_lst = []
    score_lst = []

    f = open(file_name)
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        data = line.strip().split(',')
        if len(data) != 3:
            continue

        word1_lst.append(word2id[data[0].lower()])
        word2_lst.append(word2id[data[1].lower()])
        score_lst.append(float(data[2]))

    return word1_lst, word2_lst, score_lst


def test(model, word1_id, word2_id, score_lst):
    model.eval()

    with torch.no_grad():
        word1_embed = model.inference(word1_id)
        word2_embed = model.inference(word2_id)

        similarity = torch.cosine_similarity(word1_embed, word2_embed, dim=1)
        similarity = similarity.detach().cpu().numpy()
        p_coeff = pearsonr(similarity, score_lst)
    return p_coeff


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Word2Vec Implementation')
    parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dim')
    parser.add_argument('--vector_size', type=int, default=100000, help='vector size')
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--min_count', type=int, default=10, help='minimum count')
    parser.add_argument('--workers', type=int, default=4, help='workers')

    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='save model interval (len // interval)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0025, help='learning rate')

    parser.add_argument('--load_path', type=str, default='res/lr=0.025-embed=300-vocab=100000/model_best.pth',
                        help='load_path')
    parser.add_argument('--test_file', type=str, default='combined.csv', help='test file')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    args = parser.parse_args()

    word2id_file = 'wiki-vocab-10000000-100000-w2i'
    with open(word2id_file) as f:
        word2id = json.load(f)['word2id']

    word1_lst, word2_lst, score_lst = load_test_file(args.test_file, word2id)

    assert os.path.exists(args.load_path)
    model = SkipGramModel(embedding_dim=args.embedding_dim, vocab_dim=args.vector_size)
    print('Load model from {}'.format(args.load_path))
    model.load_state_dict(torch.load(args.load_path)['model'])

    model = model.to(args.device)
    model.eval()

    with torch.no_grad():

        word1_id = torch.LongTensor(word1_lst).to(args.device)
        word2_id = torch.LongTensor(word2_lst).to(args.device)
        score_lst = np.asarray(score_lst)

        p_coeff = test(model, word1_id, word2_id, score_lst)
        print(p_coeff)
