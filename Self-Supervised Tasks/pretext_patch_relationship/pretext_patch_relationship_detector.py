import subprocess
import os
import zipfile

import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

#########################################
### 1. Download the animals dataset
#########################################

dataset_name = "alessiocorrado99/animals10"
zip_file_name = "animals10.zip"
folder_name = zip_file_name.split(".")[0]

# Execute the command using subprocess
if os.path.exists(folder_name) == True:
    print("Dataset already downloaded")
else:
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name])
    # unzip the dataset 
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(folder_name)

#########################################
### 2. Load the dataset
#########################################


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self._load_image_paths()

    def _load_image_paths(self):
        for animal_folder in os.listdir(self.root_dir):
            animal_path = os.path.join(self.root_dir, animal_folder)
            if os.path.isdir(animal_path):
                for image_file in os.listdir(animal_path):
                    self.image_paths.append(os.path.join(animal_path, image_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        # Randomly select a context patch
        context_patch = random.randint(0, 8)

        # Randomly select another patch
        other_patch = random.choice([i for i in range(9) if i != context_patch])

        if self.transform:
            image = self.transform(image)

        return image, context_patch, other_patch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CustomDataset(root_dir='animal10', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
