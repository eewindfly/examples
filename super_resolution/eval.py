
from __future__ import print_function

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

from super_resolve import resolve
from torchvision.transforms import ToTensor
from pathlib import Path

def get_latest_model_path_in_dir(dirpath):
    paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)

    if len(paths) > 0:
        return str(paths[-1])

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--input_dir',
                        type=str,
                        required=True,
                        help='input image dirpath')
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='model file to use')
    parser.add_argument('--output_dir',
                        type=str,
                        help='where to save the output image')
    opt = parser.parse_args()
    print(opt)

    input_image_files = glob.glob(os.path.join(opt.input_dir, "*.jpg"))
    input_image_files += glob.glob(os.path.join(opt.input_dir, "*.png"))
    
    for root, dirs, files in os.walk(opt.model_dir):
        for model_dirname in dirs:
            model_path = get_latest_model_path_in_dir(os.path.join(root, model_dirname))
            if model_path is None:
                continue
            
            for input_image in input_image_files:
                output_filename = os.path.basename(input_image)
                output_filename = os.path.splitext(output_filename)[0] + ".png"
                output_image_path = os.path.join(opt.output_dir, model_dirname, output_filename)
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                
                resolve(model_path, input_image, output_image_path)
