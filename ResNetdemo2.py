import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler

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
file_data = file1.drop(columns=['f.34.0.0'])
file1_data = pd.DataFrame(file_data)
file2_data = pd.DataFrame(file2)
file3_data = pd.DataFrame(file3)
file4_data = pd.DataFrame(file4[['f.eid','T2D','Complication']])
merge1 = pd.merge(file1_data, file2_data, on='f.eid')
merge2 = pd.merge(file3_data, file4_data, on='f.eid')
contact = pd.merge(merge1, merge2, on='f.eid')


#测试集处理
testfile1_c = testfile1.drop(columns=['f.34.0.0'])
testfile1_data = pd.DataFrame(testfile1_c)
testfile2_data = pd.DataFrame(testfile2)
testfile3_data = pd.DataFrame(testfile3)
testfile4_data = pd.DataFrame(testfile4[['f.eid','T2D','Complication']])
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
                         'f.100760.0.0','f.104400.0.0','T2D','Complication']
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


#数据标准化  训练集
min_max_scaler = MinMaxScaler()

#第一列患者id不进标准化
patient_id_column = continuous_data_filled['f.eid']
other_columns = continuous_data_filled.drop('f.eid',axis=1)
df_normal = pd.DataFrame(min_max_scaler.fit_transform(other_columns),columns=other_columns.columns)
contionuous_data = pd.concat([patient_id_column, df_normal], axis=1)

#测试集
test_patientid = test_continuous_filled['f.eid']
test_other = test_continuous_filled.drop('f.eid',axis=1)
test_df_normal = pd.DataFrame(min_max_scaler.fit_transform(test_other),columns=test_other.columns)
test_contionuous_data = pd.concat([test_patientid, test_df_normal], axis=1)

#标签编码
label_encoders = {}
discrete_data_encoded = discrete_data_filled.copy()
#测试集
test_discrete_encoded = test_discrete_filled.copy()

for col in discrete_data_filled.columns:
    le = LabelEncoder()
    discrete_data_encoded[col] = le.fit_transform(discrete_data_filled[col])
    test_discrete_encoded[col] = le.fit_transform(test_discrete_filled[col])

    label_encoders[col] = le



#合并处理后的数据 训练集
final_data = pd.concat([contionuous_data, discrete_data_encoded], axis=1)

#测试集
test_final = pd.concat([test_contionuous_data, test_discrete_encoded])

#特征提取
#train_data 特征  train_labels为标签
train_data = final_data.iloc[:,1:-2]
train_labels = final_data[['T2D']]
test_data = test_final.iloc[:,1:-2]
test_labels = test_final[['T2D']]   


from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

#feature_names取每一列的名称
feature_names = train_data.columns.tolist()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_data, train_labels)

feature_importance = rf.feature_importances_

# for feature_name,importance in zip(feature_names,feature_importance):
#     print(f'特征:{feature_name},重要性:{importance}')

#使用重要的特征的前100名进行模型训练
feature_importance_df = pd.DataFrame({
    'Feature' : feature_names,
    'Importance' : feature_importance  
}).sort_values('Importance', ascending=False)

#输出前100的特征
top_100_features = feature_importance_df.head(100)
print(top_100_features)


#从原始数据集中提取100个特征
train_data_top100 = train_data[top_100_features['Feature'].tolist()]
test_data_top100 = test_data[top_100_features['Feature'].tolist()]





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import f1_score

class BasicBlock(nn.Module):
    expansion  = 1

    def __init__(self,in_channels,out_channels,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels*self.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
    def forward(self,x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out
class ResNet(nn.Module):
    def __init__(self,block,num_block,num_class=3):
        super(ResNet,self).__init__()
        self.in_channels  =64
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,64,num_block[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_block[1],stride=2)
        self.layer3 = self._make_layer(block,256,num_block[2],stride=2)
        self.layer4 = self._make_layer(block,512,num_block[3],stride=2)
        self.linear = nn.Linear(512*block.expansion,num_class)
    def _make_layer(self,block,out_channels,num_block,stride):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    def forward(self,x):
        out  =nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(2)(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out 
def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2])

model = ResNet18()

optimizer = optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

train_data_tensor = torch.tensor(train_data_top100.values).float()  #2560,100
train_labels_tensor = torch.tensor(train_labels.values).squeeze().long()  #2560,2
train_data_reshape = train_data_tensor.reshape(-1,1,10,10)
train_labels_reshape = train_labels_tensor.reshape(-1,1)


test_data_tensor = torch.tensor(test_data_top100.values).float()
test_labels_tensor = torch.tensor(test_labels.values).squeeze().long()
test_data_reshape= test_data_tensor.reshape(-1,1,10,10)
test_labels_reshape = test_labels_tensor.reshape(-1,1)


batch_size = 32
train_dataset = TensorDataset(train_data_reshape,train_labels_reshape)
test_dataset = TensorDataset(test_data_reshape,test_labels_reshape)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

loss_list = []
epochs = []
num_epochs = 50
for epoch in range(num_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # 确保输入维度为 [batch_size, channels, height, width]
        #inputs = inputs.reshape(batch_size, 1, 10, 10)
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets.squeeze()).sum().item()

        all_targets.extend(targets.squeeze().tolist())
        all_predicted.extend(predicted.tolist())
        
    avg_loss= running_loss / len(train_loader)
    accuracy = (correct / total)*100
    loss_list.append(avg_loss)
    epochs.append(epoch)
    f1 = f1_score(all_targets, all_predicted, average='weighted')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f},Accuracy: {accuracy:.2f}%,F1 Score:{f1:.4f}')

import matplotlib.pyplot as plt
plt.plot(epochs,loss_list)
plt.show()

