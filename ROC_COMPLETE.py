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

from sklearn. model_selection import train_test_split
from sklearn. linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
plt.figure(figsize=[12,8])
plt.clf()


#逻辑回归

model =  LogisticRegression()
model.fit(train_data_top100, train_labels)
y_pred = model.predict(test_data_top100)
fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
plt. plot (fpr,tpr,label="Logistic Regression, AUC="+str(auc))

#fit gradient boosted model and plot ROC curve
model = GradientBoostingClassifier()
model.fit(train_data_top100, train_labels)
y_pred = model. predict_proba (test_data_top100)[:, 1]
fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
plt. plot (fpr,tpr,label="Gradient Boosting, AUC="+str(auc))


#随机森林
model = RandomForestClassifier()
model.fit(train_data_top100, train_labels)
y_pred = model. predict_proba (test_data_top100)[:, 1]
fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
plt. plot (fpr,tpr,label="Random Forest, AUC="+str(auc))

#决策树
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(train_data_top100, train_labels)
# y_pred = model. predict_proba (test_data_top100)[:, 1]
# fpr,tpr,_ = metrics.roc_curve(test_labels, y_pred)
# auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
# plt. plot (fpr,tpr,label="Decision Tree, AUC="+str(auc))

#线性回归
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_data_top100, train_labels)
y_pred = model. predict (test_data_top100)
fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
plt. plot (fpr,tpr,label="Linear Regression, AUC="+str(auc))

# #MLP
# from MLP import train_model, evaluate_model
# model = train_model(train_data_top100, train_labels)
# y_pred = evaluate_model(model, test_data_top100)
# fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
# auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
# plt. plot (fpr,tpr,label="MLP, AUC="+str(auc))

#SVM
from sklearn.svm import SVC
model = SVC()
model.fit(train_data_top100, train_labels)
y_pred = model. predict (test_data_top100)
fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
plt. plot (fpr,tpr,label="SVM, AUC="+str(auc))

#Bayes
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(train_data_top100, train_labels)
# y_pred = model. predict (test_data_top100)
# fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
# auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
# plt. plot (fpr,tpr,label="Bayes, AUC="+str(auc))

#KNN
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier()
# model.fit(train_data_top100, train_labels)
# y_pred = model. predict (test_data_top100)
# fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
# auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
# plt. plot (fpr,tpr,label="KNN, AUC="+str(auc))




#xgboost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train_data_top100, train_labels)
y_pred = model. predict (test_data_top100)
fpr, tpr, _ = metrics. roc_curve (test_labels, y_pred)
auc = round(metrics. roc_auc_score (test_labels, y_pred), 4)
plt. plot (fpr,tpr,label="XGBoost, AUC="+str(auc))


plt. legend()
plt.show()
