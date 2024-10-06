import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing  import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report,f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import matplotlib.pyplot as plt



#基础身体数据
file1 = pd.read_excel('Baseline_characteristics.xlsx')
file1_feature = file1[['f.eid','f.31.0.0','f.21022.0.0',
                       'f.21001.0.0','f.4079.0.0','f.4080.0.0']]
df1 = pd.DataFrame(file1_feature)

#生活习惯数据
file2 = pd.read_excel('life_style.xlsx')
file2_feature = file2[['f.eid','f.1478.0.0','f.1349.0.0',
                       'f.1269.0.0','f.1558.0.0','f.1210.0.0',
                       'f.1329.0.0','f.1160.0.0','f.1050.0.0',
                       'f.971.0.0','f.1110.0.0','f.924.0.0',
                       'f.1279.0.0','f.943.0.0','f.1170.0.0',
                       'f.1200.0.0','f.1458.0.0','f.1120.0.0',
                       'f.1060.0.0','f.874.0.0','f.1309.0.0',
                       'f.1190.0.0','f.1249.0.0','f.1259.0.0',
                       'f.914.0.0',
                       'f.1299.0.0','f.1239.0.0','f.981.0.0',
                       'f.1070.0.0','f.1220.0.0','f.1289.0.0',
                       'f.1289.0.0','f.100015.0.0','f.100017.0.0',
                       'f.100024.0.0','f.100009.0.0','f.104400.0.0']]
df2 = pd.DataFrame(file2_feature)


#NMR数据
file3 = pd.read_excel('NMR.xlsx')
fle3_feature = file3[['f.eid','f.23406.0.0','f.23405.0.0',
                      'f.23407.0.0','f.23402.0.0','f.23403.0.0',
                      'f.23448.0.0','f.23447.0.0','f.23464.0.0',
                      'f.23460.0.0','f.23461.0.0','f.23462.0.0',
                      'f.23468.0.0','f.23469.0.0','f.23470.0.0',
                      'f.23471.0.0','f.23472.0.0','f.23474.0.0',
                      'f.23475.0.0','f.23476.0.0','f.23477.0.0',
                      'f.23478.0.0','f.23479.0.0','f.23480.0.0',
                      'f.23428.0.0','f.23408.0.0','f.23544.0.0',
                      'f.23572.0.0']]
df3 = pd.DataFrame(fle3_feature)


#判断是否患有T2D
file4 = pd.read_excel('ground_true.xlsx')
file4_feature = file4[['f.eid','T2D']]
df4 = pd.DataFrame(file4_feature)

merged_df1_2 = pd.merge(df1, df2, on='f.eid', how='outer')
merged_df3_4 = pd.merge(df3, df4, on='f.eid', how='outer')
# 再次合并结果和 df3
final_df = pd.merge(merged_df1_2, merged_df3_4, on='f.eid', how='outer')

#确定哪些列有缺失值
columns_missing = final_df.columns[final_df.isnull().any()].tolist()

# 使用IterativeImputer进行缺失值填充
imputer = IterativeImputer(estimator=BayesianRidge(),max_iter=10,random_state=0)
df_imputed = imputer.fit_transform(final_df)

#将结果转回DataFrame
df_imputed = pd.DataFrame(df_imputed,columns=final_df.columns)
df_imputed.to_excel('final_df.xlsx', index=False)

#归一化
patient_id = df_imputed['f.eid']
other_columns = df_imputed.drop('f.eid', axis=1)
min_max_scaler = MinMaxScaler()
df_normal = pd.DataFrame(min_max_scaler.fit_transform(other_columns),columns=other_columns.columns)
result_data = pd.concat([patient_id, df_normal], axis=1)

#训练数据不包含第一列患者id和最后一列标签T2D
train_data = result_data.iloc[:,1:-1]
train_labels = result_data['T2D']

X_train,X_test,y_train,y_test = train_test_split(train_data,train_labels,test_size=0.2,random_state=42)

# #选择随机森林模型进行训练
# model = RandomForestClassifier()
# model.fit(X_train, y_train)


# # 预测测试集

# y_pred_train = model.predict(X_test)
# # 计算混淆矩阵
# conf_matrix = confusion_matrix(y_test, y_pred_train)
# print("Confusion Matrix:\n", conf_matrix)

# # 计算分类报告
# class_report = classification_report(y_test, y_pred_train)
# print("Classification Report:\n", class_report)

# # 计算训练集准确率
# train_accuracy = accuracy_score(y_test, y_pred_train)
# print(f"Train Accuracy: {train_accuracy:.4f}")


#使用CNN进行训练

#转化为numpy数组
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

#展平数据
X_train_flat = X_train.reshape(X_train.shape[0], -1)

X_train_tensor = torch.tensor(X_train_flat,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
batch_size  = 32
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


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
input_size  =X_train_flat.shape[1]
hidden_size = 128 #隐藏层大小
num_classes = 2 #类别数量

model = MLP_Model(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


num_epochs = 200
loss_history = []

for epoch in range(num_epochs):
    running_loss  = 0.0
    for batch_idx,(data,targets) in enumerate(train_loader):
        outputs = model(data)
        loss = criterion(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    avg_loss = running_loss / (batch_idx + 1)
    loss_history.append(avg_loss)
epochs = list(range(1, num_epochs + 1))
plt.plot(epochs, loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()
