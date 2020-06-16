from os import listdir
from PIL import Image
from torch.utils.data.dataset import Dataset
from data_utils import is_image_file
import torchvision.transforms as transforms


class ShiftDataLoader(Dataset):
    def __init__(self, folder_name, data_folder, upscale_factor,
                 input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):

        super(ShiftDataLoader, self).__init__()
        assert upscale_factor == 4  # warns if default changes, adjustment necessary
        # data paths are relative but still somewhat hard coded, especially the original data folder name
        self.image_dir = f"../{folder_name}/{data_folder}/{data_folder}-"
        self.target_dir = f"../{folder_name}/original/original-"
        self.number_of_images = len([file for file in listdir(self.image_dir) if is_image_file(file)])
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, _, _ = Image.open(f"{self.image_dir}{index:05d}.png").convert('YCbCr').split()
        target, _, _ = Image.open(f"{self.target_dir}{index:05d}.png").convert('YCbCr').split()
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return self.number_of_images
