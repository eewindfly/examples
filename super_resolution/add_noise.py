import argparse

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description='Generate additive white gaussian noise image')
parser.add_argument('--input_image', type=str,
                    required=True, help='input image to use')
parser.add_argument('--sigma',
                    type=int,
                    default=25,
                    help="AWGN standard deviation")
parser.add_argument('--output_filename', type=str,
                    help='where to save the output image')
parser.add_argument('--y_only', action='store_true',
                    help='add noise on y channel only?')
opt = parser.parse_args()

print(opt)

input_img = Image.open(opt.input_image)

# add noise on RGB space
if not opt.y_only:
    input = np.asarray(input_img)/255
    noise = np.random.normal(size=input.shape)*opt.sigma/255
    output = input + noise
    output = np.clip(output, 0, 1)
    output *= 255
    output_img = Image.fromarray(np.uint8(output), mode='RGB')
else:
# add noise on Y channel only
    input_img = input_img.convert('YCbCr')
    y, cb, cr = input_img.split()
    input = np.asarray(y)/255
    noise = np.random.normal(size=input.shape)*opt.sigma/255
    output = input + noise
    output = np.clip(output, 0, 1)
    output *= 255
    output_img_y = Image.fromarray(np.uint8(output), mode='L')
    output_img = Image.merge('YCbCr', [output_img_y, cb, cr]).convert('RGB')

# save
output_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)

