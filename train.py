import torch
from torch.utils.data import DataLoader
import argparse
import os
from model import SkipGramModel
from dataset import Word2VecDataset
from tqdm import tqdm
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter
from eval import test, load_test_file


def train(args):
    summary_writer = SummaryWriter(log_dir=args.output_dir)

    model = SkipGramModel(embedding_dim=args.embedding_dim, vocab_dim=args.vocab_len)
    model = model.to(args.device)
    model.train()

    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)

    dataset = Word2VecDataset("wiki.txt", "wiki.vocab", count=args.count, window=args.window, vocab_len=args.vocab_len)
    if args.wordnet_freq > 0:
        dataset.set_wordnet_aug_freq(args.wordnet_freq)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                             collate_fn=dataset.collate_fn)

    iteration = 0
    save_interval = len(data_loader) // args.save_interval
    test_interval = args.test_interval

    # ==========================================
    # load test
    word1_lst, word2_lst, score_lst = load_test_file(args.test_file, dataset.word2id)
    word1_id = torch.LongTensor(word1_lst).to(args.device)
    word2_id = torch.LongTensor(word2_lst).to(args.device)
    score_lst = np.asarray(score_lst)
    # ==========================================

    best_coeff = -1.
    for ep in range(args.epochs):
        pbar = tqdm(data_loader)
        for batch in pbar:
            cent_word, pos_word, neg_word = batch
            cent_word, pos_word, neg_word = cent_word.to(args.device), pos_word.to(args.device), neg_word.to(args.device)

            optimizer.zero_grad()
            loss = model(cent_word, pos_word, neg_word)
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: %.8s, best coeff: %.6s" % (loss.item(), best_coeff))
            summary_writer.add_scalar('loss', loss.item(), iteration)

            if iteration % save_interval == 0:
                torch.save({
                    'model': model.state_dict()
                }, os.path.join(args.output_dir, 'model_{}.pth'.format(iteration // save_interval)))
            if iteration % test_interval == 0:
                p_coeff = test(model, word1_id, word2_id, score_lst)
                model.train()

                summary_writer.add_scalar('test', p_coeff[0], iteration)

                if p_coeff[0] > best_coeff:
                    best_coeff = p_coeff[0]
                    torch.save({
                        'model': model.state_dict()
                    }, os.path.join(args.output_dir, 'model_best.pth'.format(iteration)))

            iteration += 1

    torch.save({
        'model': model.state_dict()
    }, os.path.join(args.output_dir, 'model.pth'.format(iteration)))
    return model


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Word2Vec Implementation')
    parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dim')
    parser.add_argument('--vocab_len', type=int, default=100000, help='vocabulary vector size')
    parser.add_argument('--data_tag', type=str, default='wiki', help='test file')
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--count', type=int, default=10, help='minimum count')
    parser.add_argument('--wordnet_freq', type=int, default=0, help='minimum count')
    parser.add_argument('--workers', type=int, default=4, help='workers')

    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--save_interval', type=int, default=10, help='save model interval (len // interval)')
    parser.add_argument('--test_interval', type=int, default=10000, help='test interval')
    parser.add_argument('--test_file', type=str, default='combined.csv', help='test file')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.025, help='learning rate')

    parser.add_argument('--output_dir', type=str, default='res', help='output dir')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    args = parser.parse_args()

    temp_path = 'lr={}-embed={}-vocab={}'.format(args.lr, args.embedding_dim, args.vocab_len)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, temp_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args)

    model = train(args)
