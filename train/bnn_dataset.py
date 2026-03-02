import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os

# ==========================================
# 1. 原始加载类 (用于从 170MB 压缩包提取数据)
# ==========================================
class BinaryCIFAR10(Dataset):
    def __init__(self, root='./data', train=True, download=True):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        
        print(f"正在加载 CIFAR-10 {'训练集' if train else '测试集'}...")
        self.cifar = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        
        self.data = []
        self.targets = []
        
        print("正在过滤类别并对齐 1088 维特征...")
        for img, label in self.cifar:
            if label in [0, 1]:
                flat_img = img.view(-1)
                padding = torch.zeros(64)
                aligned_input = torch.cat((flat_img, padding))
                
                self.data.append(aligned_input)
                self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ==========================================
# 2. 新增：极速加载类 (用于日常训练，秒读 .pt 文件)
# ==========================================
class FastBNNDataset(Dataset):
    def __init__(self, train=True):
        # 自动去读取 data/ 文件夹下的专属 .pt 文件
        file_path = 'data/bnn_train.pt' if train else 'data/bnn_test.pt'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到 {file_path}！请先运行 extract_data.py 提取数据。")
            
        print(f"⚡ 正在极速加载预处理数据: {file_path}")
        
        saved_data = torch.load(file_path, weights_only=True)
        self.data = saved_data['data']
        self.labels = saved_data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# --- 测试代码 ---
if __name__ == "__main__":
    # 测试一下我们的极速读取类！
    print(">>> 测试极速数据加载器 <<<")
    try:
        fast_dataset = FastBNNDataset(train=True)
        fast_loader = DataLoader(fast_dataset, batch_size=32, shuffle=True)
        
        for inputs, labels in fast_loader:
            print(f"✅ 成功加载 Batch！")
            print(f"Input shape: {inputs.shape}")  # 期望输出: torch.Size([32, 1088])
            print(f"Labels: {labels}")             # 期望输出: 仅包含 0 和 1 的标签
            break
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")