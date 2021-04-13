import torch
from torch.utils.data import DataLoader
import argparse
import os
from model import SkipGramModel
from wiki_dataset import Word2vecDataset
from tqdm import tqdm
from torch import optim
from tensorboardX import SummaryWriter


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

    parser.add_argument('--output_dir', type=str, default='res', help='output dir')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    args = parser.parse_args()

    temp_path = 'lr={}-embed={}-vocab={}'.format(args.lr, args.embedding_dim, args.vector_size)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, temp_path)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    summary_writer = SummaryWriter(log_dir=args.output_dir)

    model = SkipGramModel(embedding_dim=args.embedding_dim, vocab_dim=args.vector_size)
    model = model.to(args.device)

    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)

    dataset = Word2vecDataset("wiki.txt", "wiki.vocab", min_count=args.min_count, window=args.window,
                              fix_vocab_len=args.vector_size)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                             collate_fn=dataset.collater)

    iteration = 0
    save_interval = len(data_loader) // args.save_interval
    interval = 10000
    print(args)
    for ep in range(args.epochs):
        pbar = tqdm(data_loader)
        for batch in pbar:
            pos_u, pos_neg_v = batch
            pos_u, pos_neg_v = pos_u.to(args.device), pos_neg_v.to(args.device)

            optimizer.zero_grad()
            loss = model(pos_u, pos_neg_v)
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: %.8s" % loss.item())
            summary_writer.add_scalar('loss', loss.item(), iteration)

            if iteration % save_interval == 0:
                torch.save({
                    'model': model.state_dict()
                }, os.path.join(args.output_dir, 'model_{}.pth'.format(iteration // save_interval)))
            if iteration % interval == 0:
                torch.save({
                    'model': model.state_dict()
                }, os.path.join(args.output_dir, 'model.pth'.format(iteration)))

            iteration += 1

    torch.save({
        'model': model.state_dict()
    }, os.path.join(args.output_dir, 'model.pth'.format(iteration)))