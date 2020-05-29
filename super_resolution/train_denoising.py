import argparse
from math import log10
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net

from data import get_denoising_training_set, get_denoising_testing_set
from train_sr import parser, train, test, checkpoint, device

if __name__ == "__main__":
    # Training settings
    opt = parser.parse_args()

    print(opt)

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

    model_dirpath = "denoising_models"
    os.makedirs(model_dirpath, exist_ok=True)
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch, model, optimizer, criterion, training_data_loader)
        test(model, criterion, testing_data_loader)
        checkpoint(epoch, model, model_dirpath)
