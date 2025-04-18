import os
import gzip
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        if images.ndim == 2:
            self.images = images.reshape(-1, 28, 28)
        else:
            self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # 如果数据不是 uint8 类型，则进行转换
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])
        return image, label

# 从文件中加载 MNIST 数据，数据文件为经过 gzip 压缩的 pickle 文件
def load_mnist_data(path):
    """
    返回值为： (train_set, valid_set, test_set)
    每个 set 为一个二元组 (images, labels)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset DNE!!! {path} ")
    with gzip.open(path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set

# Set CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入通道 1，输出通道 32，卷积核 3x3，padding=1 保持尺寸不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二层卷积：输入 32 通道，输出 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 使用 2x2 最大池化层，将宽高减半
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层：
        # 第一全连接层：输入尺寸 7x7x64（28x28 经过两次 2 倍下采样变为 7x7），输出 128 个特征
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        # 第二全连接层：输出 10 个类别
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 第一层卷积 + ReLU 激活 + 池化：输出尺寸变为 [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv1(x)))
        # 第二层卷积 + ReLU 激活 + 池化：输出尺寸变为 [batch, 64, 7, 7]
        x = self.pool(F.relu(self.conv2(x)))
        # 展平为 [batch, 7*7*64]
        x = x.view(x.size(0), -1)
        # 全连接层 + ReLU
        x = F.relu(self.fc1(x))
        # 最后一层全连接，输出 logits
        x = self.fc2(x)
        return x

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalization（MNIST using MNIST STD and MEAN）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    #Load Mnist Dataset
    train_set, valid_set, test_set = load_mnist_data('data/mnist.pkl.gz')
    train_dataset = MNISTDataset(train_set[0], train_set[1], transform=transform)
    test_dataset = MNISTDataset(test_set[0], test_set[1], transform=transform)
    
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load CNN on device （CPU or GPU）
    model = SimpleCNN().to(device)
    
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0005,     
        amsgrad=True        
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    num_epochs = 3  # Epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Print Loss Every 100 pics
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
        # Evaluation After Each Epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"After epoch {epoch+1}, learning rate = {current_lr:.6f}")
    
    # Save Parameters
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("PATH SAVED To: mnist_cnn.pth")

if __name__ == '__main__':
    main()

