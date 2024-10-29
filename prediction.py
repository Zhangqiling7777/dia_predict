import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
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

#final_data为患病数据  normal_data为未患病数据
# normal_data = final_data[final_data['T2D'] == 0]
# final_data = final_data[final_data['T2D'] == 1]
# test_final  = test_final[test_final['T2D'] == 1]
# test_normal_data = test_final[test_final['T2D'] == 0]
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

# #根据age_at_diagnosis对数据进行排序
# #随机森林
# from sklearn.ensemble import RandomForestClassifier
# #feature_names取每一列的名称
# feature_names = train_data.columns.tolist()

# rf  = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(train_data, train_label)

# feature_importance = rf.feature_importances_

# # for feature_name,importance in zip(feature_names,feature_importance):
# #     print(f'特征:{feature_name},重要性:{importance}')

# #使用重要的特征的前100名进行模型训练
# feature_importance_df = pd.DataFrame({
#     'Feature' : feature_names,
#     'Importance' : feature_importance  
# }).sort_values('Importance', ascending=False)

# #输出前100的特征
# top_100_features = feature_importance_df.head(50)
# #从原始数据集中提取100个特征
# train_data_top100 = train_data[top_100_features['Feature'].tolist()]
# test_data_top100 = test_data[top_100_features['Feature'].tolist()]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据
X_train = torch.tensor(train_data.values, dtype=torch.float32).to(device)
y_train = torch.tensor(train_label.values, dtype=torch.float32).view(-1, 1).to(device)
X_test = torch.tensor(test_data.values, dtype=torch.float32).to(device)
y_test = torch.tensor(test_label.values, dtype=torch.float32).view(-1, 1).to(device)

# 数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Transformer回归模型定义
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerRegressor, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # 数据预处理
        x = self.input_fc(x)
        x = x.unsqueeze(1)  # Transformer expects sequence (batch_size, seq_len, d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.output_fc(x)
        return x

# 模型实例化
input_dim = X_train.shape[1]
model = TransformerRegressor(input_dim=input_dim).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}')

# 测试模型
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import scipy.stats as stats

# 模型预测并收集真实值和预测值
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.cpu().numpy())

# 转换为NumPy数组以便计算
all_labels = np.array(all_labels).flatten()
all_preds = np.array(all_preds).flatten()

# 计算 R², PCC, SPC
r2 = r2_score(all_labels, all_preds)
pcc, _ = stats.pearsonr(all_labels, all_preds)
spc, _ = stats.spearmanr(all_labels, all_preds)

# 打印评估指标
print(f'R²: {r2:.4f}')
print(f'PCC (Pearson Correlation Coefficient): {pcc:.4f}')
print(f'SPC (Spearman Correlation Coefficient): {spc:.4f}')

# 可视化真实值与预测值
plt.figure(figsize=(10, 6))
plt.scatter(all_labels, all_preds, alpha=0.5, label="Predictions vs Actual")
plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', lw=2, label="Ideal")
plt.xlabel("Actual Age at Diagnosis")
plt.ylabel("Predicted Age at Diagnosis")
plt.title("Predicted vs Actual Age at Diagnosis")
plt.legend()
plt.show()



# R²: 0.9136
# PCC (Pearson Correlation Coefficient): 0.9869
# SPC (Spearman Correlation Coefficient): 0.7583

