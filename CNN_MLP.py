import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pandas as pd
import numpy as np
from PIL import Image
import os


# 加载结构化数据
train_struct_data = pd.read_csv('train_final_data.csv')  # 包含 ID, 身体特征, T2D标签
test_struct_data = pd.read_csv('test_final_data.csv')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小到ResNet的输入要求
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class MultimodalDataset(Dataset):
    def __init__(self, struct_data, img_folder, transform=None):
        self.struct_data = struct_data
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.struct_data)

    def __getitem__(self, idx):
        row = self.struct_data.iloc[idx]
        patient_id = int(row['f.eid'])
        
        # 读取图像
        img_path = os.path.join(self.img_folder, f"{patient_id}_avg_wave.png")

         # 检查图像是否存在
        if not os.path.exists(img_path):
            return None  # 如果图像缺失，返回 None
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 读取结构化数据
        struct_features = torch.tensor(row[1:-2].values, dtype=torch.float32)  # 去除ID和标签列
        label = torch.tensor(row['T2D'], dtype=torch.float32)
        
        return image, struct_features, label

class MultimodalModel(nn.Module):
    def __init__(self, input_size):
        super(MultimodalModel, self).__init__()
        
        # 图像模型（使用ResNet18并适应图像大小）
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn.fc = nn.Identity()  # 移除 ResNet 的最后全连接层
        
        # 结构化数据模型（MLP）
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),  # 323列特征输入
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 融合层
        self.fc = nn.Linear(512 + 64, 1)  # 假设ResNet输出为512，MLP输出为64
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, struct_data):
        img_features = self.cnn(img)  # 提取图像特征
        struct_features = self.mlp(struct_data)  # 提取结构化数据特征
        combined = torch.cat((img_features, struct_features), dim=1)
        output = self.sigmoid(self.fc(combined))
        return output
# 分割数据集
# 生成训练和测试数据，保留 f.eid 列
train_data = train_struct_data  # 只删除 T2D 和 Complication 列
test_data = test_struct_data  # 保留测试数据的 f.eid

# 获取特征数，用于 MLP 输入维度
input_size = train_data.drop(columns=['f.eid','T2D','Complication']).shape[1]

# 实例化数据集，保留 f.eid 列
train_dataset = MultimodalDataset(train_data, img_folder='D:/python_dia/ECG_feature_extraction-main/train_ave_wave', transform=transform)
test_dataset = MultimodalDataset(test_data, img_folder='D:/python_dia/ECG_feature_extraction-main/test_ave_wave', transform=transform)
def collate_fn(batch):
    # 过滤掉None或长度不为3的项
    batch = [item for item in batch if item is not None and len(item) == 3]
    
    if len(batch) == 0:
        return None  # 如果整个batch都无效，则返回None
    
    # 将batch解压
    images, struct_features, labels = zip(*batch)
    return torch.stack(images), torch.stack(struct_features), torch.tensor(labels)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

from tqdm import tqdm

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # tqdm 集成在 loader 中显示批次进度
    for batch in tqdm(loader, desc="Training", leave=False):
        if batch is None:  # 跳过无效批次
            continue

        images, struct_features, labels = batch
        images, struct_features, labels = images.to(device), struct_features.to(device), labels.to(device)

        # 前向传播
        outputs = model(images, struct_features).squeeze()
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        # tqdm 集成在 loader 中显示批次进度
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            if batch is None:  # 跳过无效批次
                continue

            images, struct_features, labels = batch
            images, struct_features, labels = images.to(device), struct_features.to(device), labels.to(device)

            outputs = model(images, struct_features).squeeze()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # 转换为0-1分类
    predictions = np.round(predictions)
    accuracy = accuracy_score(targets, predictions)
    auc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, predictions)

    return accuracy, auc, f1




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalModel(input_size=input_size).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 5
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss = train(model, train_loader, criterion, optimizer, device)
    accuracy, auc, f1 = evaluate(model, test_loader, device)
    print(f"Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1 Score: {f1:.4f}")
