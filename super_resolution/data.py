from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from PIL import Image

from dataset import DatasetFromFolder, load_img_y_channel, load_img_rgb_channels

import torch


class AdditiveWhiteGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        result = tensor + torch.randn(tensor.size()) * self.std + self.mean
        result = torch.clamp(result, 0, 1)
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)


def download_bsd300(dest="datasets"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor=None, awgn_sigma=None):
    actions = [CenterCrop(crop_size)]
    if upscale_factor is not None:
        actions.append(
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC), )
    actions.append(ToTensor())
    if awgn_sigma is not None:
        actions.append(AdditiveWhiteGaussianNoise(std=awgn_sigma / 255))

    return Compose(actions)


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    # download BSD300
    root_dir = download_bsd300()
    train_dir1 = join(root_dir, "train")

    # download DIV2K
    root_dir = "./datasets/DIV2K"  # prepare this dataset manually
    train_dir2 = join(root_dir, "train")

    # download Flickr2K
    root_dir = "./datasets/Flickr2K"  # prepare this dataset manually
    train_dir3 = root_dir

    train_dirs = [train_dir1, train_dir2,
                  train_dir3]  # TODO: support multi datasets
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir2,
                             input_transform=input_transform(
                                 crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(
                                 crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


# denoising
def get_denoising_training_set(awgn_sigma):
    root_dir = "./datasets/DIV2K"  # prepare this dataset manually
    train_dir = join(root_dir, "train")

    crop_size = 256

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(
                                 crop_size, awgn_sigma=awgn_sigma),
                             target_transform=target_transform(crop_size),
                             load_img_func=load_img_rgb_channels)


def get_denoising_testing_set(awgn_sigma):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")

    crop_size = 256

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(
                                 crop_size, awgn_sigma=awgn_sigma),
                             target_transform=target_transform(crop_size),
                             load_img_func=load_img_rgb_channels)


# sr with denoising
def get_sr_denoising_training_set(upscale_factor, awgn_sigma):
    root_dir = "./datasets/DIV2K"  # prepare this dataset manually
    train_dir = join(root_dir, "train")

    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(
                                 crop_size,
                                 upscale_factor=upscale_factor,
                                 awgn_sigma=awgn_sigma),
                             target_transform=target_transform(crop_size),
                             load_img_func=load_img_y_channel)


def get_sr_denoising_testing_set(upscale_factor, awgn_sigma):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")

    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(
                                 crop_size,
                                 upscale_factor=upscale_factor,
                                 awgn_sigma=awgn_sigma),
                             target_transform=target_transform(crop_size),
                             load_img_func=load_img_y_channel)
