import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net

from data import get_denoising_training_set, get_denoising_testing_set
from train_sr import train, test, checkpoint

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads',
                    type=int,
                    default=4,
                    help='number of threads for data loader to use')
parser.add_argument('--batchSize',
                    type=int,
                    default=64,
                    help='training batch size')
parser.add_argument('--testBatchSize',
                    type=int,
                    default=10,
                    help='testing batch size')
parser.add_argument('--nEpochs',
                    type=int,
                    default=2,
                    help='number of epochs to train for')
parser.add_argument('--lr',
                    type=float,
                    default=0.01,
                    help='Learning Rate. Default=0.01')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_denoising_training_set(awgn_sigma=25)
test_set = get_denoising_testing_set(awgn_sigma=25)
training_data_loader = DataLoader(dataset=train_set,
                                  num_workers=opt.threads,
                                  batch_size=opt.batchSize,
                                  shuffle=True)
testing_data_loader = DataLoader(dataset=test_set,
                                 num_workers=opt.threads,
                                 batch_size=opt.testBatchSize,
                                 shuffle=False)

print('===> Building model')
model = Net(num_img_channel=3).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    checkpoint(epoch)
