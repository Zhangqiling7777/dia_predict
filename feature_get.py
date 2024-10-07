import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder


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

#特征提取
#train_data 特征  train_labels为标签
train_data = final_data.iloc[:,1:-2]
train_labels = final_data[['T2D','Complication']]

from sklearn.ensemble import RandomForestClassifier
#feature_names取每一列的名称
feature_names = train_data.columns.tolist()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_data, train_labels)

feature_importance = rf.feature_importances_

for feature_name,importance in zip(feature_names,feature_importance):
    print(f'特征:{feature_name},重要性:{importance}')

