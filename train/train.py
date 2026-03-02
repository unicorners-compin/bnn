import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 【修改点 1】：导入新的极速数据加载类
from bnn_dataset import FastBNNDataset
from bnn_model import BNNClassifier, unpack_to_bits

def train_bnn_promax():
    print("正在准备数据集...")
    
    # 【修改点 2】：直接使用打包好的 .pt 文件进行极速加载
    train_dataset = FastBNNDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BNNClassifier().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n🚀 开始 PRO MAX 版训练 (Dropout + 标签平滑), 共 {epochs} 轮...")
    for epoch in range(epochs):
        model.train() # 强制开启训练模式，激活 Dropout
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs_float, labels in train_loader:
            inputs_bits = unpack_to_bits(inputs_float).to(device)
            
            # 【魔法 2：标签平滑】
            # 不再是非黑即白，给模型留一点容错空间
            labels_smooth = torch.where(labels == 1, 0.9, 0.1)
            labels_smooth = labels_smooth.float().unsqueeze(1).to(device)
            
            # 真实标签仅用于统计准确率
            labels_real = labels.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_bits)
            
            # 用平滑后的标签计算损失
            loss = criterion(outputs, labels_smooth)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.weight.clamp_(-1.0, 1.0)
            
            total_loss += loss.item()
            predicted = (outputs > 0).float()
            correct += (predicted == labels_real).sum().item()
            total += labels.size(0)
            
        scheduler.step()
        
        accuracy = 100 * correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"第 {epoch+1:3d}/{epochs} 轮 | Loss: {total_loss/len(train_loader):.4f} | 训练准确率: {accuracy:.2f}%")

    print("\n🎉 Pro Max 版训练结束！")
    torch.save(model.state_dict(), "bnn_trained_promax.pth")
    print("模型权重已保存为: bnn_trained_promax.pth")

if __name__ == "__main__":
    train_bnn_promax()