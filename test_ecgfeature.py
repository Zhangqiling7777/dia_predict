import csv
import math
import json
import numpy as np
from matplotlib import pyplot as plt
import os
from tools.ResampleTools import resample_interp
from tools.NormalizeTools import normalize_sig_hist
from tools.SimpleFilter import smooth_avg1
from tools.QRSDetector import simple_qrs_detector
from tools.HRVS import hrvs
from tools.SingleBeatBounds import dwt_ecg_delineator
from features.ECGAvgBeat import extract_avg_wave
from features.ECGAvgBeat import extract_features

#空数据 1019333   1052928  1194303   2752005  3985855  
def extract_patient_id(filename):
    #return filename.split('/')[-1].split('_')[0]
    return os.path.basename(filename).split('_')[0]
def process_ecg_file(input_file):

    with open(input_file,'r') as fr:
        raw_data = json.load(fr)
    if not raw_data:
        raise ValueError(f"Raw data is empty for file:{input_file}")
    ecg_signal = raw_data  # raw_datat-->list or array
    # ecg_signal = np.array(ecg_signal)

    fs_raw = 125   # 原始采样率
    fs_out = 500  # 重采样率

# ECG信号重采样--基于线性拟合的差值重采样算法
    resample_ecg = resample_interp(ecg_signal, fs_raw, fs_out)

# 滤波--均值滤波
    filter_ecg = smooth_avg1(resample_ecg, radius=3)

# 数据归一化--按信号的直方图百分比进行数据归一化
    ecg_data = normalize_sig_hist(filter_ecg, TH=0.01)

# QRS波的位置检测-->R波波峰位置
    qrs = simple_qrs_detector(ecg_data, fs_out)

# 心率变异性HRV-->FrequencyDomain频域, TimeDomain时域, NonLinear非线性
    hrv = hrvs(peaks=qrs, fs=fs_out)

# 平均波形提取-->平均波形及平均波形的R波位置
    avg_wave, r, avg_rr = extract_avg_wave(ecg_data, qrs, fs_out)

# 单个Beat的各波定位-->P波、T波起始及峰值位置/QRS波起始位置/Q波和S波峰值位置
    bound_infos = dwt_ecg_delineator(avg_wave, r, fs_out)

    p = bound_infos['ECG_P_Peak']  # p波峰值位置
    p_start = bound_infos['ECG_P_Onset']  # p波开始位置
    p_end = bound_infos['ECG_P_Offset']  # p波结束位置
    q = bound_infos['ECG_Q_Peak']  # q波峰值位置
    s = bound_infos['ECG_S_Peak']  # s波峰值位置
    qrs_start = bound_infos['ECG_R_Onset']  # qrs波开始位置
    qrs_end = bound_infos['ECG_R_Offset']  # qrs波结束位置
    t = bound_infos['ECG_T_Peak']  # t波峰值位置
    t_start = bound_infos['ECG_T_Onset']  # t波开始位置
    t_end = bound_infos['ECG_T_Offset']  # t波结束位置


#  特征提取-->QRS波/PR段/ST段/T波/P波等特征信息
    ecg_data = extract_features(avg_wave, p_start, p_end, qrs_start, qrs_end, t_start, t_end, fs_out, avg_rr)
# 心电图数据
    return ecg_data


# 将复杂嵌套的字典展开成一行
def flatten_ecg_data(data):
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_data[f'{key}_{sub_key}'] = sub_value
        else:
            flat_data[key] = value
    return flat_data

# 将NaN值转换为None
def clean_nan(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def process_ecg_folder(input_folder,output_csv):
    all_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_folder, filename)
            patient_id = extract_patient_id(input_file)
            ecg_data = process_ecg_file(input_file)
            flattened_data = flatten_ecg_data(ecg_data)
            clean_data = {key: clean_nan(value) for key, value in flattened_data.items()}
            clean_data['f.eid'] = patient_id
            all_data.append(clean_data)
    
    #写入CSV文件
    with open(output_csv,mode='w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=all_data[0].keys())
        writer.writeheader()
        for data in all_data:
            writer.writerow(data)
    print(f'Data successfully written to {output_csv}')

#指定输入进而输出文件夹路径
input_folder = 'D:/python_dia/train_ecg_txt'
output_csv = 'ecg_data.csv'

#处理
process_ecg_folder(input_folder,output_csv)
