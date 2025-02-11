import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

# Create 'models' directory if not exists
os.makedirs("models", exist_ok=True)

# Configure Logger
logger.remove()  # Remove default handler
logger.add("training_log.txt", format="{time} | {level} | {message}", level="DEBUG")
logger.add(lambda msg: print(msg, end=""), format="{time} | {level} | {message}", level="INFO")

# CNN Model
class Cifar10CNN(nn.Module):
    def __init__(self):
        super(Cifar10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)  # No Softmax (CrossEntropyLoss includes it)
        return x

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Label Flipping: 20% of Class 5 → Class 3
def flip_labels(dataset, flip_percentage=0.9, source_class=5, target_class=3):
    targets = np.array(dataset.targets)
    class_5_indices = np.where(targets == source_class)[0]
    num_to_flip = int(len(class_5_indices) * flip_percentage)
    flip_indices = np.random.choice(class_5_indices, num_to_flip, replace=False)

    # Apply label flipping
    targets[flip_indices] = target_class
    dataset.targets = targets.tolist()

    # Log the flipping process
    logger.warning(f"⚠️ Flipped {num_to_flip} samples from Class {source_class} → Class {target_class}")

# Apply label flipping to training and testing datasets
flip_labels(train_dataset)
flip_labels(test_dataset)

# Save Processed Dataset
with open("train_data_loader.pickle", 'wb') as f:
    pickle.dump(train_dataset, f)
with open("test_data_loader.pickle", 'wb') as f:
    pickle.dump(test_dataset, f)

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Cifar10CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
criterion = nn.CrossEntropyLoss()

# Log start
logger.info("🚀 Training Started")

# Training Loop
for epoch in range(1, 201):
    try:
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 500 == 0:
                logger.info(f"📊 Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        
        # Save Model
        torch.save(model.state_dict(), f"models/model_epoch_{epoch}.pth")

        # Evaluate Model
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_accuracy = 100. * correct / len(test_loader.dataset)
        avg_test_loss = test_loss / len(test_loader)
        classification_rep = classification_report(all_targets, all_preds, digits=4, output_dict=True)
        conf_matrix = confusion_matrix(all_targets, all_preds)

        # Extract class-wise precision & recall
        class_precisions = {f"Class {i} Precision": classification_rep[str(i)]['precision'] for i in range(10)}
        class_recalls = {f"Class {i} Recall": classification_rep[str(i)]['recall'] for i in range(10)}

        # Logging
        logger.success(f"✅ Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")
        logger.info(f"📑 Class-wise Precision: {class_precisions}")
        logger.info(f"📑 Class-wise Recall: {class_recalls}")
        logger.debug(f"📊 Confusion Matrix:\n{conf_matrix}")

        # Save classification report to log file
        with open("classification_report.txt", "a") as f:
            f.write(f"\n\nEpoch {epoch} - Classification Report:\n")
            f.write(classification_report(all_targets, all_preds, digits=4))

    except Exception as e:
        logger.error(f"❌ Error at Epoch {epoch}: {str(e)}")

logger.success("🎉 Training Complete.")
