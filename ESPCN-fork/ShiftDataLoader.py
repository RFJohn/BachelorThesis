from os import listdir

from PIL import Image
from torch.utils.data.dataset import Dataset
from data_utils import is_image_file
import torchvision.transforms as transforms
import numpy as np


class ShiftDataLoader(Dataset):
    def __init__(self, folder_name, data_folder, upscale_factor):
        super(ShiftDataLoader, self).__init__()
        assert upscale_factor == 4  # warns if default changes, adjustment necessary
        # data paths are relative but still somewhat hard coded, especially the original data folder name
        self.image_dir = f"../{folder_name}/{data_folder}/{data_folder}-"
        self.target_dir = f"../{folder_name}/original/original-"
        self.number_of_images = len([f for f in listdir(f"../{folder_name}/{data_folder}") if is_image_file(f)])

    def __getitem__(self, index):
        image, target = self.open_images(index)
        return image, target

    def __len__(self):
        return self.number_of_images

    def open_images(self, index):
        image, _, _ = Image.open(f"{self.image_dir}{index:05d}.png").convert('YCbCr').split()
        target, _, _ = Image.open(f"{self.target_dir}{index:05d}.png").convert('YCbCr').split()
        return transforms.ToTensor()(image), transforms.ToTensor()(target)


class ShiftXYDataLoader(ShiftDataLoader):
    def __init__(self, folder_name, data_folder, upscale_factor):
        super(ShiftXYDataLoader, self).__init__(folder_name, data_folder, upscale_factor)

        self.x_coords, self.y_coords = self.load_xy(folder_name, data_folder)
        self.original_x, self.original_y = self.load_xy(folder_name, "original")

    @staticmethod
    def load_xy(top_folder, data_folder):
        x = np.loadtxt(f"../{top_folder}/{data_folder}/x_coords.txt")
        y = np.loadtxt(f"../{top_folder}/{data_folder}/y_coords.txt")
        x = x / (64.0 / 2.0) - 1.0
        y = y / (64.0 / 2.0) - 1.0
        return x.astype(np.float32), y.astype(np.float32)

    def __getitem__(self, index):
        image, target = self.open_images(index)
        image = image.numpy()[0, :, :]
        target = target.numpy()[0, :, :]
        first = np.stack([image, self.x_coords, self.y_coords], axis=2)
        second = np.stack([target, self.original_x, self.original_y], axis=2)
        first = transforms.ToTensor()(first)
        second = transforms.ToTensor()(second)
        return first, second
