import os
import json
import numpy as np
from matplotlib import pyplot as plt
from tools.ResampleTools import resample_interp
from tools.NormalizeTools import normalize_sig_hist
from tools.SimpleFilter import smooth_avg1
from tools.QRSDetector import simple_qrs_detector
from tools.HRVS import hrvs
from tools.SingleBeatBounds import dwt_ecg_delineator
from features.ECGAvgBeat import extract_avg_wave
from features.ECGAvgBeat import extract_features

# 提取患者ID
def extract_patient_id(filename):
    return os.path.basename(filename).split('_')[0]

def process_ecg_file(input_file, output_dir):
    with open(input_file, 'r') as fr:
        raw_data = json.load(fr)
    if not raw_data:
        raise ValueError(f"Raw data is empty for file: {input_file}")
    ecg_signal = raw_data  # raw_data --> list or array

    fs_raw = 125   # 原始采样率
    fs_out = 500   # 重采样率

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

    # 特征提取-->QRS波/PR段/ST段/T波/P波等特征信息
    ecg_data = extract_features(avg_wave, p_start, p_end, qrs_start, qrs_end, t_start, t_end, fs_out, avg_rr)

    # 绘制并保存平均波形图像
    plt.figure(figsize=(10, 6))
    plt.plot(avg_wave, label='Average Wave')
    plt.scatter([p], [avg_wave[p]], color='red', label='P Peak')
    plt.scatter([p_start], [avg_wave[p_start]], color='orange', label='P Onset')
    plt.scatter([p_end], [avg_wave[p_end]], color='orange', label='P Offset')
    plt.scatter([q], [avg_wave[q]], color='green', label='Q Peak')
    plt.scatter([s], [avg_wave[s]], color='green', label='S Peak')
    plt.scatter([qrs_start], [avg_wave[qrs_start]], color='blue', label='QRS Onset')
    plt.scatter([qrs_end], [avg_wave[qrs_end]], color='blue', label='QRS Offset')
    plt.scatter([t], [avg_wave[t]], color='purple', label='T Peak')
    plt.scatter([t_start], [avg_wave[t_start]], color='magenta', label='T Onset')
    plt.scatter([t_end], [avg_wave[t_end]], color='magenta', label='T Offset')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Average ECG Waveform')
    plt.legend()

    # 保存图像
    patient_id = extract_patient_id(input_file)
    output_image_path = os.path.join(output_dir, f'{patient_id}_avg_wave.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_image_path)
    plt.close()

    return ecg_data

def process_ecg_files(input_dir, output_dir):
    # 获取输入文件夹中的所有 .txt 文件
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    for input_file in input_files:
        try:
            ecg_data = process_ecg_file(input_file, output_dir)
            print(f"Processed and saved image for file: {input_file}")
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")

# 示例调用
input_dir = 'D:/python_dia/train_ecg_txt'
output_dir = 'D:/python_dia/ECG_feature_extraction-main/train_ave_wave'
process_ecg_files(input_dir, output_dir)