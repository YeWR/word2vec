import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from model import SkipGramModel
from wiki_dataset import Word2vecDataset
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

    parser.add_argument('--load_path', type=str, default='res/lr=0.0025-embed=300-vocab=100000/model_0.pth',
                        help='load_path')
    parser.add_argument('--test_file', type=str, default='combined.csv', help='test file')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    args = parser.parse_args()

    dataset = Word2vecDataset("wiki.txt", "wiki.vocab", min_count=args.min_count, window=args.window,
                              fix_vocab_len=args.vector_size)
    word1_lst, word2_lst, score_lst = load_test_file(args.test_file, dataset.word2id)

    assert os.path.exists(args.load_path)
    model = SkipGramModel(embedding_dim=args.embedding_dim, vocab_dim=args.vector_size)
    model.load_state_dict(torch.load(args.load_path)['model'])

    model = model.to(args.device)
    model.eval()

    with torch.no_grad():

        word1_id = torch.LongTensor(word1_lst).to(args.device)
        word2_id = torch.LongTensor(word2_lst).to(args.device)

        word1_embed = model.inference(word1_id)
        word2_embed = model.inference(word2_id)

        similarity = torch.cosine_similarity(word1_embed, word2_embed, dim=1)
        similarity = similarity.detach().cpu().numpy()
        score_lst = np.asarray(score_lst)
        p_coeff = pearsonr(similarity, score_lst)
        print(p_coeff)
