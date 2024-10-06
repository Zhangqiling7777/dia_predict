import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split,GridSearchCV
from torch.utils.data import DataLoader,TensorDataset
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,recall_score,precision_score,f1_score
from sklearn.multioutput import MultiOutputClassifier
#读取第一个Excel文件,并选择需要的列
# file1 = pd.read_excel('life_style.xlsx')
# file1_feature = file1[['f.eid','f.874.0.0','f.914.0.0','f.924.0.0','f.1160.0.0',
#                        'f.1200.0.0','f.1220.0.0','f.1239.0.0','f.1249.0.0',
#                        'f.20077.0.0','f.1289.0.0','f.1299.0.0','f.1309.0.0',
#                        'f.1349.0.0','f.1359.0.0','f.1369.0.0','f.1408.0.0',
#                        'f.1458.0.0','f.1478.0.0','f.1548.0.0','f.100005.0.0']]
# df1 = pd.DataFrame(file1_feature)

# # 读取第二个 Excel 文件，并选择需要的列
# file2 = pd.read_excel('Baseline_characteristics.xlsx')
#  # 选择特定的列
# file2_feature = file2[['f.eid','f.31.0.0','f.34.0.0','f.21022.0.0',
#                        'f.21001.0.0','f.4079.0.0']]
# df2 = pd.DataFrame(file2_feature)


# file3 = pd.read_excel('NMR.xlsx')
# file3_feature = file3[['f.eid','f.23470.0.0','f.23435.0.0',
#                        'f.23439.0.0','f.23440.0.0','f.23442.0.0',
#                        'f.23444.0.0','f.23445.0.0','f.23464.0.0']]
# df3 = pd.DataFrame(file3_feature)

# file4 = pd.read_excel('ground_true.xlsx')
# file4_feature = file4[['f.eid','T2D']]
# df4 = pd.DataFrame(file4_feature)

# merge1 = pd.merge(df1, df2, on='f.eid')
# merge2 = pd.merge(df3, df4, on='f.eid')
# contact = pd.merge(merge1, merge2, on='f.eid')

# final_data = pd.DataFrame(contact)


file1 = pd.read_excel('Baseline_characteristics.xlsx')
file2 = pd.read_excel('NMR.xlsx')
file3 = pd.read_excel('life_style.xlsx')
file4 = pd.read_excel('ground_true.xlsx')

file_data = file1.drop(columns=['f.34.0.0'])
file1_data = pd.DataFrame(file_data)
file2_data = pd.DataFrame(file2)
file3_data = pd.DataFrame(file3)
file4_data = pd.DataFrame(file4[['f.eid','T2D','Complication']])

merge1 = pd.merge(file1_data, file2_data, on='f.eid')
merge2 = pd.merge(file3_data, file4_data, on='f.eid')

contact = pd.merge(merge1, merge2, on='f.eid')

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
# 连续数据 离散数据
discrete_data = contact[discrete_columns]
continous_data = contact.drop(columns=discrete_data)


#填充缺失数据
mode_imputer = SimpleImputer(strategy='most_frequent')
discrete_data_filled = mode_imputer.fit_transform(discrete_data)
discrete_data_filled = pd.DataFrame(discrete_data_filled, columns=discrete_data.columns)

knn_imputer = KNNImputer(n_neighbors=5)
continuous_data_filled = knn_imputer.fit_transform(continous_data)
continuous_data_filled = pd.DataFrame(continuous_data_filled, columns=continous_data.columns)

#数据标准化
min_max_scaler = MinMaxScaler()

#第一列患者id不进标准化
patient_id_column = continuous_data_filled['f.eid']
other_columns = continuous_data_filled.drop('f.eid',axis=1)
df_normal = pd.DataFrame(min_max_scaler.fit_transform(other_columns),columns=other_columns.columns)
contionuous_data = pd.concat([patient_id_column, df_normal], axis=1)



#标签编码
label_encoders = {}
discrete_data_encoded = discrete_data_filled.copy()

for col in discrete_data_filled.columns:
    le = LabelEncoder()
    discrete_data_encoded[col] = le.fit_transform(discrete_data_filled[col])
    label_encoders[col] = le

#合并处理后的数据
final_data = pd.concat([contionuous_data, discrete_data_encoded], axis=1)

final_data.to_excel('final_data.xlsx', index=False)

#train_data 特征  train_labels为标签
train_data = final_data.iloc[:,1:-2]
train_labels = final_data[['T2D','Complication']]


# X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)




#定义模型并训练   MLP Model
# #转化为numpy数组
# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

# #展平数据
# X_train_flat = X_train.reshape(X_train.shape[0], -1)

# X_train_tensor = torch.tensor(X_train_flat,dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train,dtype=torch.float32)
# train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
# batch_size  = 32
# train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


# class MLP_Model(nn.Module):
#     def __init__(self,input_size,hidden_size,num_classes):
#         super(MLP_Model,self).__init__()
#         self.fc1 = nn.Linear(input_size,hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size,num_classes)
    
#     def forward(self,x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
# input_size  =X_train_flat.shape[1]
# hidden_size = 128 #隐藏层大小
# num_classes = 2 #类别数量

# model = MLP_Model(input_size, hidden_size, num_classes)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(),lr=0.001)


# num_epochs = 200
# loss_history = []

# for epoch in range(num_epochs):
#     running_loss  = 0.0
#     for batch_idx,(data,targets) in enumerate(train_loader):
#         outputs = model(data)
#         loss = criterion(outputs,targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
#     avg_loss = running_loss / (batch_idx + 1)
#     loss_history.append(avg_loss)
# epochs = list(range(1, num_epochs + 1))
# plt.plot(epochs, loss_history)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Epochs')
# plt.show()

#线性回归  MSE T2D:0.0672 Complication :0.0349

#创建线性回归模型
# model = LinearRegression()

# #拟合模型

# model.fit(train_data, train_labels)

# #预测
# predictions = model.predict(train_data)

# #评估模型
# mse_t2d = mean_squared_error(train_labels['T2D'], predictions[:,0])
# mse_comp = mean_squared_error(train_labels['Complication'], predictions[:,1])

# print(f'Mean Squared Error for T2D: {mse_t2d}')
# print(f'Mean Squared Error for Complication: {mse_comp}')

# # 逻辑回归  
# Accuracy for T2D: 0.9359375
# Accuracy for Complication: 0.95625
# Precision for T2D: 0.859375
# Precision for Complication: 0.25
# Recall for T2D: 0.8270676691729323
# Recall for Complication: 0.038461538461538464
# model = LogisticRegression()

# #拆分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# #创建逻辑回归模型并训练
# base_classifier = LogisticRegression(max_iter=1000,random_state=42)
# multi_output_classifier = MultiOutputClassifier(base_classifier)

# #拟合模型
# multi_output_classifier.fit(X_train, y_train)

# #预测
# predictions = multi_output_classifier.predict(X_test)

# #准确率
# accuracy_t2d = accuracy_score(y_test['T2D'],predictions[:,0])
# accuracy_com = accuracy_score(y_test['Complication'],predictions[:,1])
# #精确率
# precision_t2d = precision_score(y_test['T2D'],predictions[:,0])
# precision_com = precision_score(y_test['Complication'],predictions[:,1])
# #召回率
# recall_t2d = recall_score(y_test['T2D'],predictions[:,0])
# recall_com = recall_score(y_test['Complication'],predictions[:,1])


# print(f'Accuracy for T2D: {accuracy_t2d}')
# print(f'Accuracy for Complication: {accuracy_com}')
# print(f'Precision for T2D: {precision_t2d}')
# print(f'Precision for Complication: {precision_com}')
# print(f'Recall for T2D: {recall_t2d}')
# print(f'Recall for Complication: {recall_com}')


#SVM  
# from sklearn.svm import SVC
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder

# X_train,X_test,y_train,y_test = train_test_split(train_data,train_labels,test_size=0.2,random_state=42)



# # 创建 SVM 模型并使用 MultiOutputClassifier 包装
# base_classifier = SVC(kernel='linear', probability=True, random_state=42)
# multi_output_classifier = MultiOutputClassifier(base_classifier)

# # 训练模型
# multi_output_classifier.fit(X_train, y_train)

# # 进行预测
# predictions = multi_output_classifier.predict(X_test)




# accuracy_t2d = accuracy_score(y_test['T2D'],predictions[:,0])
# accuracy_com = accuracy_score(y_test['Complication'],predictions[:,1])
# precision_t2d = precision_score(y_test['T2D'],predictions[:,0])
# precision_com = precision_score(y_test['Complication'],predictions[:,1])
# recall_t2d = recall_score(y_test['T2D'],predictions[:,0])
# recall_com = recall_score(y_test['Complication'],predictions[:,1])
# f1_t2d = f1_score(y_test['T2D'],predictions[:,0])
# f1_com = f1_score(y_test['Complication'],predictions[:,1])

# print(f'Accuracy for T2D: {accuracy_t2d}')
# print(f'Accuracy for Complication: {accuracy_com}')
# print(f'Precision for T2D: {precision_t2d}')
# print(f'Precision for Complication: {precision_com}')
# print(f'Recall for T2D: {recall_t2d}')
# print(f'Recall for Complication: {recall_com}')
# print(f'F1 Score for T2D: {f1_t2d}')
# print(f'F1 Score for Complication: {f1_com}')


#决策树
# Accuracy for T2D: 0.9328125
# Accuracy for Complication: 0.94375
# Precision for T2D: 0.835820895522388
# Precision for Complication: 0.08333333333333333
# Recall for T2D: 0.8421052631578947
# Recall for Complication: 0.038461538461538464
# F1 Score for T2D: 0.8389513108614233
# F1 Score for Complication: 0.05263157894736842

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# X_train,X_test,y_train,y_test = train_test_split(train_data,train_labels,test_size=0.2,random_state=42)
# base_classifier = DecisionTreeClassifier()



# multi_output_classifier = MultiOutputClassifier(base_classifier)
# multi_output_classifier.fit(X_train, y_train)
# predictions = multi_output_classifier.predict(X_test)
# accuracy_t2d = accuracy_score(y_test['T2D'],predictions[:,0])
# accuracy_com = accuracy_score(y_test['Complication'],predictions[:,1])
# precision_t2d = precision_score(y_test['T2D'],predictions[:,0])
# precision_com = precision_score(y_test['Complication'],predictions[:,1])
# recall_t2d = recall_score(y_test['T2D'],predictions[:,0])
# recall_com = recall_score(y_test['Complication'],predictions[:,1])
# f1_t2d = f1_score(y_test['T2D'],predictions[:,0])
# f1_com = f1_score(y_test['Complication'],predictions[:,1])



# print(f'Accuracy for T2D: {accuracy_t2d}')
# print(f'Accuracy for Complication: {accuracy_com}')
# print(f'Precision for T2D: {precision_t2d}')
# print(f'Precision for Complication: {precision_com}')
# print(f'Recall for T2D: {recall_t2d}')
# print(f'Recall for Complication: {recall_com}')
# print(f'F1 Score for T2D: {f1_t2d}')
# print(f'F1 Score for Complication: {f1_com}')





#随机森林
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# X_train,X_test,y_train,y_test = train_test_split(train_data,train_labels,test_size=0.2,random_state=42)
# base_classifier = RandomForestClassifier()
# multi_output_classifier = MultiOutputClassifier(base_classifier)
# multi_output_classifier.fit(X_train, y_train)
# predictions = multi_output_classifier.predict(X_test)
# accuracy_t2d = accuracy_score(y_test['T2D'],predictions[:,0])
# accuracy_com = accuracy_score(y_test['Complication'],predictions[:,1])
# precision_t2d = precision_score(y_test['T2D'],predictions[:,0])
# precision_com = precision_score(y_test['Complication'],predictions[:,1])
# recall_t2d = recall_score(y_test['T2D'],predictions[:,0])
# recall_com = recall_score(y_test['Complication'],predictions[:,1])
# f1_t2d = f1_score(y_test['T2D'],predictions[:,0])
# f1_com = f1_score(y_test['Complication'],predictions[:,1])
# print(f'Accuracy for T2D: {accuracy_t2d}')
# print(f'Accuracy for Complication: {accuracy_com}')
# print(f'Precision for T2D: {precision_t2d}')
# print(f'Precision for Complication: {precision_com}')
# print(f'Recall for T2D: {recall_t2d}')
# print(f'Recall for Complication: {recall_com}')
# print(f'F1 Score for T2D: {f1_t2d}')
# print(f'F1 Score for Complication: {f1_com}')
# print(multi_output_classifier.estimators_)
# print(multi_output_classifier.estimators_[0].feature_importances_)
