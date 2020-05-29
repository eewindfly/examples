import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_sr_denoising_training_set, get_sr_denoising_testing_set

from train_sr import parser, train, test, checkpoint, device

if __name__ == "__main__":
    parser.add_argument('--upscale_factor',
                        type=int,
                        required=True,
                        help="super resolution upscale factor")
    opt = parser.parse_args()

    print(opt)

    print('===> Loading datasets')
    train_set = get_sr_denoising_training_set(
        upscale_factor=opt.upscale_factor, awgn_sigma=10)
    test_set = get_sr_denoising_testing_set(upscale_factor=opt.upscale_factor,
                                            awgn_sigma=10)
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=opt.threads,
                                      batch_size=opt.batchSize,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set,
                                     num_workers=opt.threads,
                                     batch_size=opt.testBatchSize,
                                     shuffle=False)

    print('===> Building model')
    model = Net(upscale_factor=opt.upscale_factor,
                num_img_channel=1).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    model_dirpath = "denoising_sr_models"
    os.makedirs(model_dirpath, exist_ok=True)
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch,
              model=model,
              optimizer=optimizer,
              criterion=criterion,
              data_loader=training_data_loader)
        test(model=model, criterion=criterion, data_loader=testing_data_loader)
        checkpoint(epoch, model, dirpath=model_dirpath)
