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
testfile1 = pd.read_excel('D:/python_demo/CG2405/CG2405/test/Baseline_characteristics.xlsx')
testfile2 = pd.read_excel('D:/python_demo/CG2405/CG2405/test/NMR.xlsx')
testfile3 = pd.read_excel('D:/python_demo/CG2405/CG2405/test/life_style.xlsx')
testfile4 = pd.read_excel('D:/python_demo/CG2405/CG2405/test/ground_true.xlsx')


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
min_max_scaler = StandardScaler()

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
test_final = pd.concat([test_contionuous_data, test_discrete_encoded],axis=1)

#特征提取
#train_data 特征  train_labels为标签
train_data = final_data.iloc[:,1:-2]
train_labels = final_data[['T2D']]
test_data = test_final.iloc[:,1:-2]
test_labels = test_final[['T2D']]   



#随机森林特征重要性排序
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
#feature_names取每一列的名称
feature_names = train_data.columns.tolist()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_data, train_labels)

feature_importance = rf.feature_importances_


#使用重要的特征的前100名进行模型训练
feature_importance_df = pd.DataFrame({
    'Feature' : feature_names,
    'Importance' : feature_importance  
}).sort_values('Importance', ascending=False)

#输出前100的特征
top_100_features = feature_importance_df.head(100)
print(top_100_features)

#可视化特征重要性
# top_10_features = feature_importance_df.head(10)

# plt.figure(figsize=(12,8))
# sns.barplot(x='Importance', y='Feature', data=top_10_features,palette='viridis')
# plt.title('Top 10 Feature Importance')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()




#从原始数据集中提取100个特征
train_data_top100 = train_data[top_100_features['Feature'].tolist()]
test_data_top100 = test_data[top_100_features['Feature'].tolist()]





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import f1_score,roc_curve,auc

class MLP_Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(MLP_Model,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
input_size = train_data_top100.shape[1]  # 输入特征的数量
hidden_size = 128  # 隐藏层的神经元数量
num_classes = 2  #
model = MLP_Model(input_size, hidden_size, num_classes)

optimizer = optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

train_data_tensor = torch.tensor(train_data_top100.values).float()  # 3200, 100
train_labels_tensor = torch.tensor(train_labels.values).squeeze().long()  # 3200, 1
train_data_reshape = train_data_tensor  # 不需要 reshape

test_data_tensor = torch.tensor(test_data_top100.values).float()
test_labels_tensor = torch.tensor(test_labels.values).squeeze().long()
test_data_reshape = test_data_tensor


batch_size = 32
train_dataset = TensorDataset(train_data_reshape, train_labels_tensor)
test_dataset = TensorDataset(test_data_reshape, test_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

loss_list = []
test_loss_list = []
epochs = []
num_epochs = 50
train_acc_list = []
test_acc_list = []

best_test_loss = float('inf')  # 记录最优的测试集损失
best_test_accuracy = 0.0  # 记录最优的测试集准确率

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    #增加all_probabilities 列表存储roc
    all_probabilities = [] 
    
    for inputs, targets in train_loader:
        #
        # 确保输入维度为 [batch_size, channels, height, width]
        #inputs = inputs.reshape(batch_size, 1, 10, 10)
            
        #评估模型不应该调用optimizer.zero_grad()
        
        optimizer.zero_grad() 
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
        #收集正类概率
        all_probabilities.extend(outputs.softmax(dim=1)[:,1].tolist())
        
    avg_loss= running_loss / len(train_loader)
    accuracy = (correct / total)*100
    loss_list.append(avg_loss)
    train_acc_list.append(accuracy)
    epochs.append(epoch)
    f1 = f1_score(all_targets, all_predicted, average='weighted')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f},Accuracy: {accuracy:.2f}%,F1 Score:{f1:.4f}')
    
   
    
    # **在测试集上进行评估**
    model.eval()  # 切换到评估模式
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    test_all_targets = []
    test_all_predicted = []
    #测试集roc
    test_all_probabilities = []

    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            
            # 累加测试集的损失
            test_loss += loss.item()
            
            # 计算测试集的预测值
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets.squeeze()).sum().item()
            
            test_all_targets.extend(targets.squeeze().tolist())
            test_all_predicted.extend(predicted.tolist())
            #收集测试集正类概率
            test_all_probabilities.extend(outputs.softmax(dim=1)[:,1].tolist())
    # 计算测试集的平均损失和准确率
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = (test_correct / test_total) * 100
    test_f1 = f1_score(test_all_targets, test_all_predicted, average='weighted')
    test_loss_list.append(avg_test_loss)
    test_acc_list.append(test_accuracy)
    # epochs.append(epoch)
    # 打印测试结果
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.4f}')

    #保存最优模型
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), 'best_model_based_on_loss.pth')
        print(f'Best model saved based on loss at epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}')
    
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), 'best_model_based_on_accuracy.pth')
        print(f'Best model saved based on accuracy at epoch {epoch+1}, Test Accuracy: {test_accuracy:.2f}%')


import matplotlib.pyplot as plt
#LOSS
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_list, label='Train Loss', marker='o')
plt.plot(epochs, test_loss_list, label='Test Loss', marker='x')

plt.title('Training and Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#ACCURACY
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc_list, label='Train Accuracy', marker='o')
plt.plot(epochs, test_acc_list, label='Test Accuracy', marker='x')

plt.title('Training and Test Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#绘制ROC曲线
# fpr, tpr, _ = roc_curve(test_all_targets, test_all_probabilities)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()