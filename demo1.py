import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder


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

file_data = file1.drop(columns=['f.35.0.0'])
file1_data = pd.DataFrame(file_data)
file2_data = pd.DataFrame(file2)
file3_data = pd.DataFrame(file3)
file4_data = pd.DataFrame(file4[['f.eid','T2D','Complication']])

merge1 = pd.merge(file1_data, file2_data, on='f.eid')
merge2 = pd.merge(file3_data, file4_data, on='f.eid')

contact = pd.merge(merge1, merge2, on='f.eid')

#离散数据缺失使用众数填充  连续性数据使用多重插补


#区分离散数据和连续数据

discrete_columns = ['f.31.0.0','f.34.0.0','f.924.0.0',
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
continuous_data_normal  = min_max_scaler.fit_transform(continuous_data_filled)

continuous_data_normaled = pd.DataFrame(continuous_data_normal, columns=continuous_data_filled.columns)

#标签编码
label_encoders = {}
discrete_data_encoded = discrete_data_filled.copy()

for col in discrete_data_filled.columns:
    le = LabelEncoder()
    discrete_data_encoded[col] = le.fit_transform(discrete_data_filled[col])
    label_encoders[col] = le

#合并处理后的数据
final_data = pd.concat([continuous_data_normaled, discrete_data_encoded], axis=1)


#将数据集分离为训练集和测试集
train_data = final_data.sample(frac=0.8, random_state=42)
test_data = final_data.drop(train_data.index)

# 定义模型并训练
import keras

