from __future__ import print_function
import argparse
import os
from math import log10
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.measure import compare_ssim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import Net
from data import get_training_set, get_test_set

try:
    device = torch.device("cuda")
except:
    device = torch.device("cpu")

import matplotlib.pyplot as plt

NUM_THREADs = 4
TEST_BATCH_SIZE = 100


def train(epoch, model, optimizer, criterion, data_loader):
    epoch_loss = 0
    for iteration, batch in enumerate(data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
        #     epoch, iteration, len(data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(data_loader)))


def test(model, criterion, data_loader):
    avg_psnr = 0
    avg_ssim = 0
    with torch.no_grad():
        for batch in data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr

            prediction = np.squeeze(
                np.transpose(prediction.cpu().numpy(), (1, 2, 3, 0)))
            target = np.squeeze(
                np.transpose(target.cpu().numpy(), (1, 2, 3, 0)))
            ssim = compare_ssim(
                prediction, target, data_range=1,
                multichannel=True)  # NOTE: multichannel is tricky
            avg_ssim += ssim

            # sample_input = torch.squeeze(input[0].cpu().permute(1, 2, 0))
            # sample_target = torch.squeeze(target[0].cpu().permute(1, 2, 0))
            # sample_result = torch.squeeze(prediction[0].cpu().permute(1, 2, 0))
            # plt.imshow(sample_input, cmap='gray')
            # plt.show()
            # plt.imshow(sample_target, cmap='gray')
            # plt.show()
            # plt.imshow(sample_result, cmap='gray')
            # plt.show()
    avg_psnr /= len(data_loader)
    avg_ssim /= len(data_loader)

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f}".format(avg_ssim))

    return avg_psnr, avg_ssim


def checkpoint(epoch, model, dirpath=""):
    print(dirpath)
    model_out_path = os.path.join(dirpath, "model_epoch_{}.pth".format(epoch))
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# Training settings
torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize',
                    type=int,
                    default=64,
                    help='training batch size')
parser.add_argument('--nEpochs',
                    type=int,
                    default=2,
                    help='number of epochs to train for')
parser.add_argument('--lr',
                    type=float,
                    default=0.01,
                    help='Learning Rate. Default=0.01')
parser.add_argument('--residual',
                    action='store_true',
                    help='network use residual architecture? Default=False')
if __name__ == "__main__":
    parser.add_argument('--upscale_factor',
                        type=int,
                        required=True,
                        help="super resolution upscale factor")
    opt = parser.parse_args()

    print(opt)

    print('===> Loading datasets')
    train_set = get_training_set(opt.upscale_factor)
    test_set = get_test_set(opt.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=NUM_THREADs,
                                      batch_size=opt.batchSize,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set,
                                     num_workers=NUM_THREADs,
                                     batch_size=TEST_BATCH_SIZE,
                                     shuffle=False)

    print('===> Building model')
    model = Net(upscale_factor=opt.upscale_factor,
                num_img_channel=1,
                global_residual=opt.residual).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # setup tensorboard
    hyper_params_str = json.dumps(vars(opt))
    log_dirpath = os.path.join('logs/sr', hyper_params_str)
    writer = SummaryWriter(log_dirpath)

    # write model to tensorboard
    sample_inputs, sample_targets = next(iter(training_data_loader))
    sample_inputs = sample_inputs.to(device)
    writer.add_graph(model, sample_inputs)

    model_dirpath = os.path.join("models/sr", hyper_params_str)
    os.makedirs(model_dirpath, exist_ok=True)
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch,
              model=model,
              optimizer=optimizer,
              criterion=criterion,
              data_loader=training_data_loader)
        psnr, ssim = test(model=model,
                          criterion=criterion,
                          data_loader=testing_data_loader)
        writer.add_scalar("PSNR/test", psnr, epoch)
        writer.add_scalar("SSIM/test", ssim, epoch)
        checkpoint(epoch, model, dirpath=model_dirpath)
