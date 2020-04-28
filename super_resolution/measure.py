import warnings
import argparse
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--result_image', type=str,
                    help='super resolution result image')
parser.add_argument('--gt_image', type=str, default=None,
                    help='ground truth image')
opt = parser.parse_args()

result_img = Image.open(opt.result_image).convert('YCbCr')
result_img, _, _ = result_img.split()
gt_img = Image.open(opt.gt_image).convert('YCbCr')
gt_img, _, _ = gt_img.split()
X = np.array(result_img)
Y = np.array(gt_img)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pnsr_val = compare_psnr(Y, X, data_range=255)
    ssim_val = compare_ssim(X, Y, data_range=255)

print(f"psnr: {pnsr_val:.2f}, ssim: {ssim_val:.4f}")
