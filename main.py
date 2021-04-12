import torch
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Word2Vec Implementation')
    parser.add_argument('--vector_size', type=int, default=100, help='vector size')
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--min_count', type=int, default=1, help='minimum count')
    parser.add_argument('--workers', type=int, default=4, help='workers')

