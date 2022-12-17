import os

import numpy as np
import pandas as pd
import torch
from cv2 import cv2
from torch.utils.data import Dataset

from utils import compute_data_mean, normalize_input


class AnimeDataSet(Dataset):
    def __init__(self, args, transform=None):
        """
        folder structure:
            - {data_dir}
                - photo
                    1.jpg, ..., n.jpg
                - {dataset}  # E.g Hayao
                    smooth
                        1.jpg, ..., n.jpg
                    style
                        1.jpg, ..., n.jpg
        """
        data_dir = args.data_dir
        dataset = args.dataset

        self.photo_dir = args.photo_dir
        self.style_dir = args.style_dir

        self._image_size = (512, 512)

        # anime_dir = os.path.join(data_dir, dataset)
        # if not os.path.exists(data_dir):
        #     raise FileNotFoundError(f"Folder {data_dir} does not exist")

        # if not os.path.exists(anime_dir):
        #     raise FileNotFoundError(f"Folder {anime_dir} does not exist")

        self.mean = compute_data_mean(os.path.join(data_dir, self.style_dir))
        print(f"Mean(B, G, R) of {dataset} are {self.mean}")

        self.debug_samples = args.debug_samples or 0
        self.data_dir = data_dir
        self.image_files = {}
        self.photo = self.photo_dir
        self.style = self.style_dir
        # self.smooth = f"{anime_dir}/smooth"
        # self.dummy = torch.zeros(3, 256, 256)

        self._photo_limit = 3500

        for opt in [
            self.photo,
            self.style,
            # self.smooth
        ]:
            folder = os.path.join(data_dir, opt)
            files = os.listdir(folder)

            if opt == self.photo:
                files = files[: self._photo_limit]

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]

        self.transform = transform

        print(
            # f"Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}, smooth {self.len_smooth}"
            f"Dataset: real {len(self.image_files[self.photo])} style {self.len_anime}"
        )

    def __len__(self):
        return self.debug_samples or len(self.image_files[self.photo])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    # @property
    # def len_smooth(self):
    #     return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        image = self.load_photo(index)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray = self.load_anime(anm_idx)
        # smooth_gray = self.load_anime_smooth(anm_idx)

        # return image, anime, anime_gray, smooth_gray
        return image, anime, anime_gray

    def load_photo(self, index):
        fpath = self.image_files[self.photo][index]
        image = cv2.imread(fpath)
        image = cv2.resize(image, self._image_size)
        image = image[:, :, ::-1]
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def load_anime(self, index):
        fpath = self.image_files[self.style][index]
        image = cv2.imread(fpath)
        image = cv2.resize(image, self._image_size)
        image = image[:, :, ::-1]

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)

        image = self._transform(image, addmean=True)
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.tensor(image_gray)

    def load_anime_smooth(self, index):
        fpath = self.image_files[self.smooth][index]
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self._image_size)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = img.astype(np.float32)
        if addmean:
            img += self.mean

        return normalize_input(img)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    anime_loader = DataLoader(
        AnimeDataSet("dataset/Hayao/smooth"), batch_size=2, shuffle=True
    )

    img, img_gray = next(iter(anime_loader))
    plt.imshow(img[1].numpy().transpose(1, 2, 0))
    plt.show()
