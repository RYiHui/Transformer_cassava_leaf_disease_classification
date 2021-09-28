import torch
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms as transforms
import Global_Variable as gl

transforms_train = transforms.Compose([
    transforms.Resize((gl.IMG_SIZE,gl.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomResizedCrop(gl.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

transforms_valid = transforms.Compose([
    transforms.Resize((gl.IMG_SIZE,gl.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])


class CassavaDataset(Dataset):
    def __init__(self, df, data_path=gl.DATA_PATH, mode='train', transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode
        self.data_dir = "train_images" if mode == "train" else "test_images"

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        img_name, label = self.df_data[item]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(img)
        return image, label


