from loguru import logger
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from typing import Any

class FEMNISTDataset(Dataset):
    def __init__(self, args: Any) -> None:
        """
        Initialize the FEMNIST dataset using LEAF framework.
        """
        self.root = args.get_data_path()
        self.data_dir = os.path.join(self.root, "leaf", "data", "femnist", "data", "all_data")
        
        self.download_leaf_femnist()
        self.train_data, self.train_targets, self.train_users = self.load_dataset("train")
        self.test_data, self.test_targets, self.test_users = self.load_dataset("test")
        
        self.user_to_index = {user: idx for idx, user in enumerate(set(self.train_users))}
        self.train_users = torch.tensor([self.user_to_index[user] for user in self.train_users])
        self.test_users = torch.tensor([self.user_to_index[user] for user in self.test_users])

        self.train_dataset = TensorDataset(self.train_data, self.train_targets, self.train_users)
        self.test_dataset = TensorDataset(self.test_data, self.test_targets, self.test_users)

        super(FEMNISTDataset, self).__init__()

    def download_leaf_femnist(self):
        """
        Clone and preprocess FEMNIST using the LEAF framework.
        """
        if not os.path.exists(self.data_dir):
            logger.info("Cloning LEAF repository and preprocessing FEMNIST dataset...")
            os.system("git clone https://github.com/TalwalkarLab/leaf.git " + os.path.join(self.root, "leaf"))
            os.system("cd " + os.path.join(self.root, "leaf", "data", "femnist") + " && bash preprocess.sh -s niid --sf 1.0 -k 0 -t federated")
        else:
            logger.info("FEMNIST dataset already downloaded.")

    def load_dataset(self, split: str):
        """
        Load train or test dataset from JSON files.
        """
        logger.debug(f"Loading FEMNIST {split} data")
        file_path = os.path.join(self.data_dir, split, "all_data.json")
        
        if not os.path.exists(file_path):
            raise RuntimeError(f"Dataset file not found: {file_path}")
        
        with open(file_path, "r") as f:
            data_json = json.load(f)
        
        data, targets, users = [], [], []
        
        for user_id in data_json["users"]:
            user_data = data_json["user_data"][user_id]
            data.extend(user_data["x"])
            targets.extend(user_data["y"])
            users.extend([user_id] * len(user_data["x"]))
        
        data_tensor = torch.tensor(data, dtype=torch.float32).view(-1, 1, 28, 28)  # Reshape for CNNs
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        
        return data_tensor, targets_tensor, users

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def __getitem__(self, index: int):
        """
        Get a single data sample.
        """
        img, target, user = self.train_data[index], int(self.train_targets[index]), self.train_users[index]
        img = Image.fromarray(img.numpy().squeeze(), mode='L')
        return img, target, user
