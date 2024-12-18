import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class MedicalMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('L')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros((1, 224, 224)), label

class CustomMedicalResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomMedicalResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def plot_metrics(metrics_dict, save_dir):
    """Plot and save training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics_dict['train_losses'], 'b-', label='Train Loss')
    plt.plot(metrics_dict['val_losses'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics_dict['train_accs'], 'b-', label='Train Accuracy')
    plt.plot(metrics_dict['val_accs'], 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Training metrics plot saved to {save_path}")

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'medical_training_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load datasets
    print("Loading datasets...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dataset = MedicalMNISTDataset(
        os.path.join(base_dir, 'medical_mnist/train'), 
        transform=train_transform
    )
    val_dataset = MedicalMNISTDataset(
        os.path.join(base_dir, 'medical_mnist/val'), 
        transform=val_transform
    )
    test_dataset = MedicalMNISTDataset(
        os.path.join(base_dir, 'medical_mnist/test'), 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Initialize model
    print("Initializing model...")
    model = CustomMedicalResNet(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True
    )
    
    # Training loop
    print("Starting training...")
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': []
    }
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler