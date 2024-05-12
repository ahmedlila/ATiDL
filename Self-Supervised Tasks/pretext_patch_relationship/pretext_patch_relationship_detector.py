import subprocess
import os
import zipfile

import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset

import lightning as L
from watermark import watermark


class PatchesDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self._load_image_paths()
        self.STRIDE = 2
        self.IMAGE_SIZE = 256
        self.NO_OF_PATCHES = 9
        self.patch_size = int((self.IMAGE_SIZE - self.STRIDE * (np.sqrt(self.NO_OF_PATCHES) - 1)) // np.sqrt(self.NO_OF_PATCHES))
        self.combinations = self._generate_random_tuples(36)  # 9C2 = 36, but chose 32 

    def _load_image_paths(self):
        for animal_folder in os.listdir(self.root_dir):
            animal_path = os.path.join(self.root_dir, animal_folder)
            if os.path.isdir(animal_path):
                for image_file in os.listdir(animal_path):
                    self.image_paths.append(os.path.join(animal_path, image_file))
    
    def _extract_patch(self, image: np.ndarray, x: int, y: int) -> np.ndarray:

        patch = image[:, y:y+self.patch_size, x:x+self.patch_size]
        return patch
    
    def _generate_random_tuples(self, n: int) -> set:

        tuples = set()  # Using a set to avoid duplicates
        while len(tuples) < n:
            context_index = random.randint(0, self.NO_OF_PATCHES-1)
            other_index = random.randint(0, self.NO_OF_PATCHES-1)
            if context_index != other_index:
                tuples.add((context_index, other_index))
        return tuples

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)

        patches = []
        patch_indices = []
        all_context_patches, all_context_labels = [], []
        all_other_patches, all_other_labels = [], []
                
        # Generate patches without overlapping
        patch_strided = int(self.patch_size+self.STRIDE)
        for y in range(0, self.IMAGE_SIZE, patch_strided):
            for x in range(0, self.IMAGE_SIZE, patch_strided):
                patch = self._extract_patch(image, x, y)
                patches.append(patch)
                y_index = (y//self.patch_size)*np.sqrt(self.NO_OF_PATCHES)
                x_index = x//self.patch_size
                location = x_index + y_index
                patch_indices.append(location)
        
        for context_idx, other_idx in self.combinations:
            context_patch = patches[context_idx]
            context_label = patch_indices[context_idx]
            other_patch = patches[other_idx]
            other_label = patch_indices[other_idx]
            all_context_patches.append(context_patch)
            all_context_labels.append(context_label)
            all_other_patches.append(other_patch)
            all_other_labels.append(other_label)
        
        # Convert lists to tensors
        stacked_context_patches = torch.stack(all_context_patches)
        stacked_other_patches = torch.stack(all_other_patches)
        context_label = torch.tensor(all_context_labels)
        other_label = torch.tensor(all_other_labels)
        return stacked_context_patches, stacked_other_patches, context_label, other_label

class DataLoaderManager:
    def __init__(self, root_path, batch_size=32):
        self.root_path = root_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.train_loader, self.val_loader, self.test_loader = self._create_dataloaders()

    def _create_dataloaders(self):
        pass
    
    def _split_val_test_sets(self, dataset, test_fraction=0.3):
        pass

def download_kaggle_dataset(dataset_name, zip_file_name):
    folder_name = zip_file_name.split(".")[0]

    if os.path.exists(folder_name):
        print("Dataset already downloaded")
    else:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name])
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(folder_name)

def show_batch_images(batch_images, batch_labels):
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(16, 8))
        print(type(batch_images[0][0]))
        for i, ax in enumerate(axes.flatten()):
            image = batch_images[0][i].permute(1, 2, 0)  # Permute dimensions from (3, 84, 84) to (84, 84, 3)
            label = batch_labels[0][i]
            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L.seed_everything(123)
    
    #########################################
    ### 1 Downloading the Dataset
    
    dataset_name = "alessiocorrado99/animals10"
    zip_file_name = "animals10.zip"
    download_kaggle_dataset(dataset_name, zip_file_name)

    #########################################
    ### 2 Loading the Dataset
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = PatchesDataset(root_dir='animals10/raw-img/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Check the shape of context patches in the first batch
    context_patch, _, context_label, _ = next(iter(dataloader))
    print("Shape of context patches:", context_patch.shape)
    
    # Showing sample patches 
    show_batch_images(context_patch, context_label)
