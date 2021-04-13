import torch.utils.data
import os
import pickle
import json
import numpy as np
from tqdm import tqdm
import bisect
import copy

from torch.utils.data import DataLoader

class Word2vecDataset(torch.utils.data.Dataset):
    """A dataset that provides helpers for batching."""
    def __init__(self, filename, vocab_file, min_count=5, window=5, fix_vocab_len=1e4, num_sent=1e7):
        vocab_file += ".{}.{}".format(int(num_sent) if num_sent else None, int(fix_vocab_len))
        # self.positive_file = "/mnt/word2vec/wiki_pair.{}.{}.{}.bin".format(int(num_sent) if num_sent else None, window, int(fix_vocab_len))
        self.center_file = "wiki_center.{}.{}".format(int(num_sent) if num_sent else None, int(fix_vocab_len))
        print(self.center_file)
        self.filename = filename

        if os.path.exists(vocab_file):
            with open(vocab_file) as f:
                for k, v in json.load(f).items():
                    setattr(self, k, v)
        else:
            self.vocab = self.get_vocab(min_count, vocab_file, fix_vocab_len=fix_vocab_len, num_sent=num_sent)
        self.negative_sample_table = np.asarray(self.negative_sample_table)
        self.cumsum_sizes = np.cumsum(self.sizes)

        self.init_sample_ratio()
        self.window = window
        # self.get_positive(num_sent)
        self.get_center(num_sent)

        self.negative_num = 5
        print("load_done")

    def get_center(self, num_sent=None):
        if os.path.exists(self.center_file):
            with open(self.center_file, 'rb') as f:
                self.split_lines = pickle.load(f)
            return

        self.split_lines = []
        with open(self.filename) as f:
            for n, sentence in enumerate(tqdm(f, desc="get_center", total=num_sent)):
                if num_sent and n >= num_sent:
                    break
                # word_ids = []
                self.split_lines.append([])
                for i, word in enumerate(sentence.strip().split(' ')):
                    try:
                        self.split_lines[-1].append(self.word2id[word])
                    except:
                        self.split_lines[-1].append(-1)
        with open(self.center_file, 'wb') as f:
            pickle.dump(self.split_lines, f)

    def __len__(self):
        return self.cumsum_sizes[-1]
    
    def get_vocab(self, min_count, vocab_file, fix_vocab_len=None, num_sent=None):

        word_frequency = dict()
        with open(self.filename) as f:
            self.sizes = []
            for n, line in enumerate(tqdm(f, desc="get_vocab", total=num_sent)):
                if num_sent and n >= num_sent:
                    break
                line = line.strip().split(' ')
                self.sizes.append(len(line))
                for w in line:
                    word_frequency[w] = word_frequency.get(w, 0) + 1
            self.word2id = dict()
            self.id2word = dict()
            wid = 0
            self.word_frequency = []
            for n, (w, c) in enumerate(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)):
                if c < min_count:
                    continue
                if fix_vocab_len and len(self.word_frequency) >= fix_vocab_len:
                    break
                self.word2id[w] = wid
                self.id2word[wid] = w
                assert len(self.word_frequency) == wid
                self.word_frequency.append(c)
                wid += 1

            self.len_vocab = len(self.word_frequency)

        self.negative_sample_table = []
        sample_table_size = 1e8
        pow_frequency = np.array(list(word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.negative_sample_table += [wid] * int(c)

        with open(vocab_file, "w") as f:
            json.dump({
                "word2id": self.word2id,
                "id2word": self.id2word,
                "word_frequency": self.word_frequency,
                "sizes": self.sizes,
                "len_vocab": self.len_vocab,
                "negative_sample_table": self.negative_sample_table
            }, f)

    def init_sample_ratio(self):
        self.sample_table_size = 1e8
        self.sample_table = []
        pow_frequency = np.array(self.word_frequency)**0.75
        self.sample_ratio = pow_frequency / pow_frequency.sum()
        count = np.round(self.sample_ratio * self.sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = np.array(self.sample_table)
        np.random.shuffle(self.sample_table)
        self.sample_table_size = self.sample_table.size

    def get_neg_word(self, u):
        neg_v = []
        while len(neg_v) < self.negative_num:
            n_w = np.random.choice(self.negative_sample_table, size=self.negative_num).tolist()[0]
            if n_w != u:
                neg_v.append(n_w)
        return neg_v

    def __getitem__(self, index):
        # n = np.searchsorted(self.cumsum_sizes - 1, index)
        n = bisect.bisect(self.cumsum_sizes, index)
        i = index - self.cumsum_sizes[n - 1] if n > 0 else index
        cur_sent = self.split_lines[n]
        center_word = cur_sent[i]
        count = 0
        while center_word == -1 and count < 10:
            i = np.random.randint(len(cur_sent))
            count += 1
            if count == 10:
                return self.__getitem__((index + 1) % len(self))
        poses = cur_sent[max(i - self.window, 0): i] + cur_sent[ i + 1: i + self.window]
        poses = [pos for pos in poses if pos >= 0]
        np.random.shuffle(poses)
        pos = [center_word, poses[0] if poses else center_word]

        rand_idx = index % (self.sample_table_size - 5)
        sample = self.sample_table[rand_idx: rand_idx + 5]
        # neg_u = [pos[0] for _ in range(self.negative_num)]
        # neg_v = [v for v in self.get_neg_word(pos[0])]
        neg_v = sample.tolist()
        return pos + neg_v

    def collater(self, samples):
        pos_u = torch.LongTensor(samples)[:, 0].unsqueeze(1)
        pos_neg_v = torch.LongTensor(samples)[:, 1:]
        return pos_u, pos_neg_v

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)


if __name__ == "__main__":
    dataset = Word2vecDataset("wiki.txt", "wiki.vocab")
    for i in range(100):
        print(dataset[i])
