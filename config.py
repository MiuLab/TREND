
import os
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch

def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_epoch', type=int, default=30, help='batch_size')
    parser.add_argument('--train_path', type=str, default='data/train.pkl')
    parser.add_argument('--dev_path', type=str, default='data/dev.pkl')
    parser.add_argument('--test_path', type=str, default='data/test.pkl')
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--device', type=torch.device, default='cuda:1')

    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--ckpt_dir', type=Path, default='./models')
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--load_path', type=Path, default='models/best10000.pt')


    args = parser.parse_args()
    return args

args = parse_arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
