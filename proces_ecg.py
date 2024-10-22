import os
from xml.etree import ElementTree

def extract_waveform_data(xml_file):
    tree = ElementTree.parse(xml_file)
    waveform_data_elements = tree.findall(".//MedianSamples/WaveformData")
    waveform_data_list = []
    for waveform in waveform_data_elements:
        data = waveform.text.strip()
        waveform_data_list.append(data)
    return waveform_data_list

def save_waveform_data_to_txt(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as txt_file:
        txt_file.write('[\n')
        for i, item in enumerate(data):
            txt_file.write(f"  {item}")
            if i < len(data) - 1:
                txt_file.write(",\n")
            else:
                txt_file.write("\n")
        txt_file.write("]\n")

def process_ecg_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.xml'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace('.xml', '.txt'))
            waveform_data = extract_waveform_data(input_file)
            save_waveform_data_to_txt(waveform_data, output_file)
            print(f'数据已保存到 {output_file}')

# 指定输入和输出文件夹路径
input_folder = 'D:/python_dia/CG2405/train/ecg'
output_folder = 'D:/python_dia/train_ecg_txt'

# 处理文件夹中的所有 XML 文件
process_ecg_folder(input_folder, output_folder)