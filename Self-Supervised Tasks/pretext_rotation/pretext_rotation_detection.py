# Thanks to Sebastian Raschka, this code was inspired by his project: https://github.com/rasbt/pycon2024/
import os
import time
import requests
import tarfile

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import torchmetrics
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import lightning as L
from watermark import watermark

# Custom dataset class to load original and rotated images
class RotationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_imgs_paths = []
        dir_list = [os.path.join(self.root_dir, directory) for directory in os.listdir(self.root_dir)]
        for dir_name in dir_list:
            self.all_imgs_paths.extend([os.path.join(dir_name, img_name) for img_name in os.listdir(dir_name)])
        

    def __len__(self):
        return len(self.all_imgs_paths)

    def __getitem__(self, idx):
        img_path = self.all_imgs_paths[idx]
        image = Image.open(img_path).convert("RGB")
        original_image = self.transform(image)
        label = 0
        return original_image, label

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
        train_dataset = RotationDataset(root_dir=f'{self.root_path}/train', transform=self.transform)
        rotated_90_dataset = [(self._rotate_90(image), 1) for image, _ in tqdm(train_dataset)]
        rotated_180_dataset = [(self._rotate_180(image), 2) for image, _ in tqdm(train_dataset)]
        rotated_270_dataset = [(self._rotate_270(image), 3) for image, _ in tqdm(train_dataset)]

        concatenated_train_dataset = ConcatDataset([train_dataset, rotated_90_dataset, rotated_180_dataset, rotated_270_dataset])
        train_loader = DataLoader(concatenated_train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = RotationDataset(root_dir=f'{self.root_path}/val', transform=self.transform)
        rotated_90_val_dataset = [(self._rotate_90(image), 1) for image, _ in tqdm(val_dataset)]
        rotated_180_val_dataset = [(self._rotate_180(image), 2) for image, _ in tqdm(val_dataset)]
        rotated_270_val_dataset = [(self._rotate_270(image), 3) for image, _ in tqdm(val_dataset)]

        concatenated_val_dataset = ConcatDataset([val_dataset, rotated_90_val_dataset, rotated_180_val_dataset, rotated_270_val_dataset])
        val_loader = DataLoader(concatenated_val_dataset, batch_size=self.batch_size, shuffle=True)

        # Splitting the concatenated validation dataset into validation and test subsets
        val_subset, test_subset = self._split_val_test_sets(concatenated_val_dataset)

        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    def _rotate_90(self, image):
        return TF.rotate(image, 90)

    def _rotate_180(self, image):
        return TF.rotate(image, 180)

    def _rotate_270(self, image):
        return TF.rotate(image, 270)

    def _split_val_test_sets(self, dataset, test_fraction=0.3):
        dataset_length = len(dataset)
        test_dataset_length = int(test_fraction * dataset_length)
        indices = list(range(dataset_length))
        random.shuffle(indices)
        test_indices = indices[:test_dataset_length]
        val_indices = indices[test_dataset_length:]
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)
        return val_subset, test_subset

class RotationClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(num_epochs, model, optimizer, scheduler, train_loader, val_loader, device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = F.cross_entropy(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Plotting and saving loss and accuracy curves
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show()


def download_and_extract(extract_dir):
    # Check if the directory exists, if not, create it
    if not os.path.exists(extract_dir):
        print(f"Creating directory {extract_dir}...")
        os.makedirs(extract_dir)
        # Check if the tar file exists, if not, download it
        tar_file_path = os.path.join(extract_dir, 'imagenette2.tgz')
        if not os.path.exists(tar_file_path):
            print("Downloading imagenette2.tgz...")
            response = requests.get("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", stream=True)
            with open(tar_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)
            print("Download completed.")

        # Extract the tar file
        print("Extracting imagenette2.tgz...")
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("Extraction completed.")


if __name__ == "__main__":

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L.seed_everything(123)
    
    #########################################
    ### 1 Downloading the Dataset
    
    extract_dir = "imagenette2"
    download_and_extract(extract_dir)
    
    #########################################
    ### 2 Loading the Dataset
    
    dataloader_manager = DataLoaderManager(root_path=extract_dir, batch_size=32)
    train_loader = dataloader_manager.train_loader
    val_loader = dataloader_manager.val_loader
    test_loader = dataloader_manager.test_loader

    #########################################
    ### 3 Initializing the Model

    model = RotationClassifier(num_classes=4)
    model.to(device)
    
    NUM_EPOCHS = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_steps = NUM_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    #########################################
    ### 4 Finetuning
    
    start = time.time()
    train(
        num_epochs=NUM_EPOCHS,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device
    )

    end = time.time()
    elapsed = end-start
    print(f"Time elapsed {elapsed/60:.2f} min")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    
    #########################################
    ### 4 Evaluation

    with torch.no_grad():
        model.eval()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=4).to(device)

        for (features, targets) in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            predicted_labels = torch.argmax(outputs, 1)
            test_acc.update(predicted_labels, targets)
    print(f"Test accuracy {test_acc.compute()*100:.2f}%")
