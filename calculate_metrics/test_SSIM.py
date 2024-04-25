import argparse
import numpy as np
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim



def calculate_ssim(img_path1, img_path2):
    pred_img = Image.open(img_path1)
    pred_np = np.asarray(pred_img.convert('L'))

    gt_img = Image.open(img_path2)
    gt_np = np.asarray(gt_img.convert('L'))

    returnSSIM = ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)
    return returnSSIM


def main(folder1, folder2):
    filenames1 = set(os.listdir(folder1))
    filenames2 = set(os.listdir(folder2))

    common_filenames = sorted(list(filenames1.intersection(filenames2)))
    ssim_values = []

    for filename in common_filenames:
        img_path1 = os.path.join(folder1, filename)
        img_path2 = os.path.join(folder2, filename)
        ssim = calculate_ssim(img_path1, img_path2)
        print("SSIM value for {} is {}".format(filename, ssim))
        ssim_values.append(ssim)

    mean_ssim = np.mean(ssim_values)

    print("Mean SSIM value:", mean_ssim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate SSIM for images in two folders')
    parser.add_argument('--folder1', type=str, required=True, help='Path to the first folder')
    parser.add_argument('--folder2', type=str, required=True, help='Path to the second folder')
    args = parser.parse_args()

    main(args.folder1, args.folder2)