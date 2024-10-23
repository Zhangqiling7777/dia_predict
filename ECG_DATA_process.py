import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
csv_df = pd.read_csv('ecg_data.csv')
csv_df = pd.DataFrame(csv_df)
xlsx_df = pd.read_excel('final_data.xlsx')

#提取finaal_data中的T2D列和id列
T2D = xlsx_df[['f.eid','T2D']]

#按照id列合并
merger_t2d_ecg = pd.merge(csv_df,T2D,on='f.eid',how='outer')
merger_t2d_ecg = pd.DataFrame(merger_t2d_ecg)
ecg_columns = ['f.eid','QT','QTc','HR','R/T','QRS_RAmplitude','QRS_RDuration',
               'QRS_SDuration','QRS_QDuration','QRS_QRSAmplitude',
               'QRS_QRSDuration','ST_duration','ST_amplitude',
               'T_amplitude','T_duration','P_amplitude','P_duration']
ecg_data = merger_t2d_ecg[ecg_columns]
ecg_T2D = merger_t2d_ecg['T2D']

#
knn_imputer = KNNImputer(n_neighbors=5)
ecg_data_filled = knn_imputer.fit_transform(ecg_data)
ecg_data_filled = pd.DataFrame(ecg_data_filled, columns=ecg_data.columns)
min_max_scaler = StandardScaler()

patient_id_column = ecg_data_filled['f.eid']
other_columns = ecg_data_filled.drop('f.eid',axis=1)
df_normal = pd.DataFrame(min_max_scaler.fit_transform(other_columns),columns=other_columns.columns)
ecg_data = pd.concat([patient_id_column, df_normal], axis=1)

train_data = ecg_data.iloc[:,1:-1]
train_labels = ecg_T2D



