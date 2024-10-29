import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


#加载训练集数据
file1 = pd.read_excel('Baseline_characteristics.xlsx')
file2 = pd.read_excel('NMR.xlsx')
file3 = pd.read_excel('life_style.xlsx')
file4 = pd.read_excel('ground_true.xlsx')

#测试集
testfile1 = pd.read_excel('D:/python_dia/CG2405/test/Baseline_characteristics.xlsx')
testfile2 = pd.read_excel('D:/python_dia/CG2405/test/NMR.xlsx')
testfile3 = pd.read_excel('D:/python_dia/CG2405/test/life_style.xlsx')
testfile4 = pd.read_excel('D:/python_dia/CG2405/test/ground_true.xlsx')



#训练集处理
#提取患有T2D的患者数据
file1 = pd.DataFrame(file1)
file_data = file1.drop(columns=['f.34.0.0'])
train_birth  = file1['f.34.0.0']
file1_data = pd.DataFrame(file_data)
file2_data = pd.DataFrame(file2)
file3_data = pd.DataFrame(file3)
file4_data = pd.DataFrame(file4)
merge1 = pd.merge(file1_data, file2_data, on='f.eid')
merge2 = pd.merge(file3_data, file4_data, on='f.eid')
contact = pd.merge(merge1, merge2, on='f.eid')


#测试集处理
testfile1 = pd.DataFrame(testfile1)
testfile1_c = testfile1.drop(columns=['f.34.0.0'])
test_birth  = testfile1['f.34.0.0']
testfile1_data = pd.DataFrame(testfile1_c)
testfile2_data = pd.DataFrame(testfile2)
testfile3_data = pd.DataFrame(testfile3)
testfile4_data = pd.DataFrame(testfile4)
test_merge1 = pd.merge(testfile1_data, testfile2_data, on='f.eid')
test_merge2 = pd.merge(testfile3_data, testfile4_data, on='f.eid')
test_contact = pd.merge(test_merge1, test_merge2, on='f.eid')


#离散数据缺失使用众数填充  连续性数据使用多重插补


#区分离散数据和连续数据

discrete_columns = ['f.31.0.0','f.35.0.0','f.924.0.0',
                         'f.943.0.0','f.971.0.0',
                         'f.1100.0.0','f.1130.0.0',
                         'f.1190.0.0','f.1200.0.0','f.1210.0.0',
                         'f.1239.0.0','f.1249.0.0','f.1259.0.0','f.1269.0.0','f.1279.0.0',
                         'f.1289.0.0','f.1299.0.0','f.1309.0.0','f.1319.0.0','f.1329.0.0',
                         'f.1349.0.0','f.1359.0.0','f.1369.0.0','f.1408.0.0','f.1458.0.0',
                         'f.1478.0.0','f.1548.0.0','f.1558.0.0','f.1568.0.0','f.1578.0.0',
                         'f.1598.0.0','f.1618.0.0','f.1628.0.0','f.2110.0.0',
                         'f.2237.0.0','f.2277.0.0','f.2634.0.0','f.2644.0.0',
                         'f.2867.0.0','f.2907.0.0','f.3436.0.0',
                         'f.3731.0.0','f.20077.0.0','f.20160.0.0',
                         'f.100760.0.0','f.104400.0.0','T2D','Complication','date']
# 连续数据 离散数据 训练集
discrete_data = contact[discrete_columns]
continous_data = contact.drop(columns=discrete_data)



#填充缺失数据 训练集
mode_imputer = SimpleImputer(strategy='most_frequent')
discrete_data_filled = mode_imputer.fit_transform(discrete_data)
discrete_data_filled = pd.DataFrame(discrete_data_filled, columns=discrete_data.columns)
knn_imputer = KNNImputer(n_neighbors=5)
continuous_data_filled = knn_imputer.fit_transform(continous_data)
continuous_data_filled = pd.DataFrame(continuous_data_filled, columns=continous_data.columns)

#测试集填充
test_discrete = test_contact[discrete_columns]
test_continous = test_contact.drop(columns=test_discrete)

test_discrete_filled = mode_imputer.fit_transform(test_discrete)
test_discrete_filled = pd.DataFrame(test_discrete_filled, columns=test_discrete.columns)
test_continuous_filled = knn_imputer.fit_transform(test_continous)
test_continuous_filled = pd.DataFrame(test_continuous_filled, columns=test_continous.columns)


#数据标准化  训练集 连续性数据
min_max_scaler = StandardScaler()

#第一列患者id不进标准化
patient_id_column = continuous_data_filled['f.eid']
other_columns = continuous_data_filled.drop('f.eid',axis=1)
df_normal = pd.DataFrame(min_max_scaler.fit_transform(other_columns),columns=other_columns.columns)
contionuous_data = pd.concat([patient_id_column, df_normal], axis=1)

#测试集 数据标准化，连续性数据
test_patientid = test_continuous_filled['f.eid']
test_other = test_continuous_filled.drop('f.eid',axis=1)
test_df_normal = pd.DataFrame(min_max_scaler.fit_transform(test_other),columns=test_other.columns)
test_contionuous_data = pd.concat([test_patientid, test_df_normal], axis=1)

#标签编码
label_encoders = {}
discrete_data_date = discrete_data_filled['date']
discrete_data = discrete_data_filled.drop('date',axis=1)
discrete_data_encoded = discrete_data.copy()
#测试集
test_discrete_date = test_discrete_filled['date']
test_discrete = test_discrete_filled.drop('date',axis=1)
test_discrete_encoded = test_discrete.copy()

for col in discrete_data.columns:
    le = LabelEncoder()
    discrete_data_encoded[col] = le.fit_transform(discrete_data[col])
    test_discrete_encoded[col] = le.fit_transform(test_discrete[col])

    label_encoders[col] = le
discrete_data_encoded = pd.concat([discrete_data_encoded,discrete_data_date],axis=1)
test_discrete_encoded = pd.concat([test_discrete_encoded,test_discrete_date],axis=1)

#合并处理后的数据 训练集
final_data = pd.concat([contionuous_data, discrete_data_encoded,train_birth], axis=1)

#测试集
test_final = pd.concat([test_contionuous_data, test_discrete_encoded,test_birth],axis=1)


#final_data['year'] 为患者患病的时间
final_data['year'] = pd.to_datetime(final_data['date']).dt.year
test_final['year'] = pd.to_datetime(test_final['date']).dt.year   

#计算患病年龄  发病时间（year）-出生日期（f.34.0.0）
final_data['age_at_diagnosis'] = final_data['year'] - final_data['f.34.0.0']
test_final['age_at_diagnosis'] = test_final['year'] - test_final['f.34.0.0']

#删除无用列  Complication date T2D f.34.0.0 year
final_data.drop(columns=['date','f.34.0.0','year','Complication'],inplace=True)
test_final.drop(columns=['date','f.34.0.0','year','Complication'],inplace=True)

#划分数据和标签
train_data = final_data.drop(columns=['f.eid'])
#T2D为0age_at_diagnosis为0
train_data.loc[train_data['T2D'] == 0, 'age_at_diagnosis'] = 0
train_label = train_data['age_at_diagnosis']


test_data = test_final.drop(columns=['f.eid'])
test_data.loc[test_data['T2D'] == 0, 'age_at_diagnosis'] = 0
test_label = test_data['age_at_diagnosis']




transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小到ResNet的输入要求
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 定义超参数和模型
param_grid = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [16, 32],
    'epochs': [5, 10]
}

# 记录损失和分数
train_losses = []
val_accuracies = []
val_aucs = []
val_f1_scores = []

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
        label = torch.tensor(row['age_at_diagnosis'], dtype=torch.float32)
        
        return image, struct_features, label

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, cnn_output_dim=128):
        super(TransformerRegressor, self).__init__()
        
        # 结构化数据处理
        self.input_fc = nn.Linear(input_dim, d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 心电图图像CNN层
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 25 * 37, cnn_output_dim),  # 根据图像尺寸（如1000x600）调整
            nn.ReLU()
        )
        
        # 最后的全连接层，用于整合Transformer和CNN输出
        self.output_fc = nn.Linear(d_model + cnn_output_dim, 1)
        
    def forward(self, x_struct, x_img):
        # 结构化数据处理（Transformer）
        x_struct = self.input_fc(x_struct)
        x_struct = x_struct.unsqueeze(1)  # (batch_size, seq_len=1, d_model)
        x_struct = self.transformer(x_struct).mean(dim=1)  # (batch_size, d_model)

        # 心电图图像数据处理（CNN）
        x_img = self.cnn_layers(x_img)  # (batch_size, cnn_output_dim)
        
        # 将结构化数据和心电图特征拼接
        x = torch.cat((x_struct, x_img), dim=1)  # (batch_size, d_model + cnn_output_dim)
        
        # 输出预测值
        output = self.output_fc(x)
        return output
