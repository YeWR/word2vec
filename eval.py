import torch
from torch.utils.data import DataLoader
import argparse
import os
from model import SkipGramModel
from wiki_dataset import Word2vecDataset
from tqdm import tqdm
from torch import optim
from tensorboardX import SummaryWriter


def load_test_file(file_name):
    word_pair_lst = []
    score_lst = []

    f = open(file_name)
    line = f.readline()
    while line:
        line = f.readline()

        data = line.split(',')
        word_pair_lst.append((data[0], data[1]))
        score_lst.append(float(data[2]))

    return word_pair_lst, score_lst


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Word2Vec Implementation')
    parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dim')
    parser.add_argument('--vector_size', type=int, default=50000, help='vector size')
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--min_count', type=int, default=10, help='minimum count')
    parser.add_argument('--workers', type=int, default=4, help='workers')

    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='save model interval (len // interval)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0025, help='learning rate')

    parser.add_argument('--load_path', type=str, default='res', help='load_path')
    parser.add_argument('--test_file', type=str, default='combined.csv', help='test file')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    args = parser.parse_args()

    summary_writer = SummaryWriter(log_dir=args.output_dir)

    assert os.path.exists(args.load_path)
    model = SkipGramModel(embedding_dim=args.embedding_dim, vocab_dim=args.vector_size)
    model.load_state_dict(torch.load(args.load_path))

    model = model.to(args.device)


