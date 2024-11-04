import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

train_struct_data = pd.read_csv('D:/python_dia/train_diagnosis.csv')
test_struct_data = pd.read_csv('D:/python_dia/test_diagnosis.csv')




class T2DDataset(Dataset):
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
        struct_features = torch.tensor(row[1:-1].values, dtype=torch.float32)  # 去除ID和标签列
        label = torch.tensor(row['age_at_diagnosis'], dtype=torch.float32)
        
        return image, struct_features, label
    
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit pretrained models
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    # 过滤掉None或长度不为3的项
    batch = [item for item in batch if item is not None and len(item) == 3]
    
    if len(batch) == 0:
        return None  # 如果整个batch都无效，则返回None
    
    # 将batch解压
    images, struct_features, labels = zip(*batch)
    return torch.stack(images), torch.stack(struct_features), torch.tensor(labels)


# Load datasets
train_dataset = T2DDataset(train_struct_data, img_folder='D:/python_dia/ECG_feature_extraction-main/train_ave_wave', transform=image_transforms)
test_dataset = T2DDataset(test_struct_data, img_folder='D:/python_dia/ECG_feature_extraction-main/test_ave_wave', transform=image_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,collate_fn=collate_fn)

# MLP+VIT 模型
class MultiVITMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiVITMLP, self).__init__()
        
        # Load a pretrained ViT model and adapt for feature extraction
        self.visiontransformer = models.vit_b_16(pretrained=True)
        
        # Replace the classification head with Identity to get feature outputs
        self.visiontransformer.heads.head = nn.Identity()
        
        # Resize transformation to make images compatible with ViT
        self.resize = transforms.Resize((224, 224))
        
        # Freeze ViT parameters if needed (optional)
        for param in self.visiontransformer.parameters():
            param.requires_grad = False
            
        # MLP for processing structured data
        self.struct_fc = nn.Sequential(  
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Getting feature dimensions of the ViT model's head (after replacing with Identity)
        vit_feature_dim = self.visiontransformer.embed_dim  # Typically 768 or model-specific
        
        # MLP for final output
        self.out_fc = nn.Sequential(
            nn.Linear(vit_feature_dim + 64, 32),  # vit_feature_dim来自ViT特征输出
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x_struct, x_img):
        
        x_img = self.resize(x_img)#32,3,224,224
        x_img = self.visiontransformer(x_img)#32,1000
        x_struct = self.struct_fc(x_struct)#32,64
        x = torch.cat((x_img, x_struct), dim=1)#32,1064
        x = self.out_fc(x)
        return x
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = train_struct_data.drop(columns=['f.eid','T2D']).shape[1]
model = MultiVITMLP(input_dim=input_dim,output_dim=1).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
# 定义训练函数
from tqdm import tqdm

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # Use tqdm to create a progress bar
    for x_struct, x_img, y in tqdm(train_loader, desc="Training", leave=False):
        x_struct, x_img, y = x_struct.to(device), x_img.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x_struct, x_img)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# 定义评估函数
def test(model, test_loader, device):
    model.eval()
    predictions, true_values = [], []
    
    with torch.no_grad():
        for x_struct, x_img, y in test_loader:
            x_struct, x_img, y = x_struct.to(device), x_img.to(device), y.to(device)
            outputs = model(x_struct, x_img).squeeze()
            
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(y.cpu().numpy())
    
    return predictions, true_values

# 评估指标计算
def evaluate_model(predictions, true_values):
    pcc, _ = pearsonr(predictions, true_values)
    spc, _ = spearmanr(predictions, true_values)
    r2 = r2_score(true_values, predictions)
    
    print(f'PCC: {pcc:.4f}')
    print(f'SPC: {spc:.4f}')
    print(f'R² Score: {r2:.4f}')


# 可视化损失和分数


# 训练过程
num_epochs = 40
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')

# 获取预测值和真实值
predictions, true_values = test(model, test_loader, device)
evaluate_model(predictions, true_values)

# 绘制真实值与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predictions, alpha=0.6, color='blue')
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predictions")
plt.show()  

# PCC: 0.9133
# SPC: 0.7111
# R² Score: 0.7918