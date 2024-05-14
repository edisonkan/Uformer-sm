import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from timm import create_model

class DerainDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.target_images = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = self.input_images[idx]
        target_image_path = self.target_images[idx]
        
        input_image = Image.open(input_image_path).convert('RGB')
        target_image = Image.open(target_image_path).convert('RGB')
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

class DerainSwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, features_only=True)
        self.upsampler = nn.Sequential(
            # 第一层上采样将特征图从 [batch, 768, 7, 7] 变到 [batch, 128, 14, 14]
            nn.ConvTranspose2d(768, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 第二层上采样从 [batch, 128, 14, 14] 变到 [batch, 64, 28, 28]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 第三层上采样从 [batch, 64, 28, 28] 变到 [batch, 32, 56, 56]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 第四层上采样从 [batch, 32, 56, 56] 变到 [batch, 16, 112, 112]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 第五层上采样从 [batch, 16, 112, 112] 变到 [batch, 3, 224, 224]
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.base_model(x)[-1]  # 获取适当的特征层输出
        x = x.permute(0, 3, 1, 2)  # 重排维度，确保符合[batch_size, channels, height, width]
        x = self.upsampler(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

def main():
    device = torch.device('cpu')  # 指定使用CPU
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = DerainDataset('dataset/deraining/sidd/train/input', 'dataset/deraining/sidd/train/target', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = DerainSwinTransformer().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10  # 可调整训练周期

    train(model, device, train_loader, optimizer, criterion, epochs)

    # 保存模型
    torch.save(model.state_dict(), 'derain_model.pth')
    print("Model saved.")

if __name__ == '__main__':
    main()
