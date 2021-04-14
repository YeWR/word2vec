import os
import pickle
import json
from itertools import chain
import random
from nltk.corpus import wordnet as wn
import numpy as np
import torch.utils.data
from tqdm import tqdm
import bisect


class Word2VecDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, data_tag='wiki', count=5, window=5, vocab_len=1e5, total_data=1e7):
        self.vocab_file = data_tag + "-vocab-{}-{}".format(int(total_data), int(vocab_len))
        self.center_file = data_tag + "-center-{}-{}".format(int(total_data), int(vocab_len))
        print(self.center_file)
        self.data_file = data_file
        self.window = window

        # ================================================================================
        # make data or load data
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file) as f:
                for k, v in json.load(f).items():
                    setattr(self, k, v)
        else:
            self.make_vocab(count, vocab_len=vocab_len, total_data=total_data)
        # ================================================================================
        self.cumsum_sizes = np.cumsum(self.sizes)

        self.init_sample_ratio()
        # self.get_positive(total_data)
        self.get_center(total_data)
        self.wordnet_freq = 0
        print("load_done")

    def set_wordnet_aug_freq(self, freq):
        print('set word net augmentation frequency to {}'.format(freq))
        self.wordnet_freq = freq

    def get_center(self, total_data=None):
        if os.path.exists(self.center_file):
            with open(self.center_file, 'rb') as f:
                self.split_lines = pickle.load(f)
            return

        self.split_lines = []
        with open(self.data_file) as f:
            for n, sentence in enumerate(tqdm(f, desc="get_center", total=total_data)):
                if total_data and n >= total_data:
                    break
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

    def make_vocab(self, count, vocab_len=None, total_data=None):

        word_frequency = dict()
        with open(self.data_file) as f:
            self.sizes = []
            for n, line in enumerate(tqdm(f, desc="make_vocab", total=total_data)):
                if total_data and n >= total_data:
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
                if c < count:
                    continue
                if vocab_len and len(self.word_frequency) >= vocab_len:
                    break
                self.word2id[w] = wid
                self.id2word[wid] = w
                assert len(self.word_frequency) == wid
                self.word_frequency.append(c)
                wid += 1
            self.len_vocab = len(self.word_frequency)

        with open(self.vocab_file, "w") as f:
            json.dump({
                "word2id": self.word2id,
                "id2word": self.id2word,
                "word_frequency": self.word_frequency,
                "sizes": self.sizes,
                "len_vocab": self.len_vocab,
                "negative_sample_table": self.negative_sample_table
            }, f)

        word2id_file = self.vocab_file + '-w2i'
        with open(word2id_file, "w") as f:
            json.dump({
                "word2id": self.word2id,
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
            if n_w != u and n_w >= 0:
                neg_v.append(n_w)
        return neg_v

    def __getitem__(self, index):
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
        poses = cur_sent[max(i-self.window, 0):i] + cur_sent[i+1:i+self.window]
        poses = [pos for pos in poses if pos >= 0]
        np.random.shuffle(poses)

        # use word net augmentation
        if self.wordnet_freq > 0 and index % self.wordnet_freq == 0:
            synonyms = wn.synsets(self.id2word[center_word])
            synonyms_set = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            if len(synonyms_set) > 0:
                pos = [center_word, random.sample(synonyms_set, 1)[0]]
            else:
                pos = [center_word, center_word]
        else:
            pos = [center_word, poses[0] if poses else center_word]

        index = np.random.randint(0, self.sample_table_size - 5 - 1)
        rand_idx = index % (self.sample_table_size - 5)
        sample = self.sample_table[rand_idx: rand_idx + 5]
        return pos + sample.tolist()

    def collate_fn(self, batch):
        cent_word = torch.LongTensor(batch)[:, 0].unsqueeze(1)
        pos_word = torch.LongTensor(batch)[:, 1].unsqueeze(1)
        neg_word = torch.LongTensor(batch)[:, 2:]
        return cent_word, pos_word, neg_word
