# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth

import argparse
import os
from glob import glob

import numpy as np
from cv2 import cv2
from tqdm import tqdm


def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--dataset", type=str, default="Paprika", help="dataset_name")
    parser.add_argument("--image-size", type=int, default=256, help="The size of image")

    return parser.parse_args()


def make_edge_smooth(dataset_name, img_size):
    file_list = glob("dataset/{}/{}/*.*".format(dataset_name, "style"))
    save_dir = "dataset/{}/smooth".format(dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    for f in tqdm(file_list):
        file_name = os.path.basename(f)

        bgr_img = cv2.imread(f)
        gray_img = cv2.imread(f, 0)

        bgr_img = cv2.resize(bgr_img, (img_size, img_size))
        pad_img = np.pad(bgr_img, ((2, 2), (2, 2), (0, 0)), mode="reflect")
        gray_img = cv2.resize(gray_img, (img_size, img_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        0,
                    ],
                    gauss,
                )
            )
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        1,
                    ],
                    gauss,
                )
            )
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        2,
                    ],
                    gauss,
                )
            )

        assert cv2.imwrite(os.path.join(save_dir, file_name), gauss_img)


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_edge_smooth(args.dataset, args.image_size)


if __name__ == "__main__":
    main()
