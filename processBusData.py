"""
This script processes the raw data
"""
import csv

import pandas as pd

from globalVar import *
import os
import numpy as np
import json

raw_data_folder = get_value('raw_data_folder')
process_data_folder = get_value('process_data_folder')
tensorflow = get_value('tensorflow_folder')
transport_node_bus_folder = get_value('transport_node_bus_folder')


# CSV------------------------------------------------------------------------------------------------------------------
def read_csv(input_path):  # path -> list[list[str]] | 输入文件路径，返回存储csv的二维列表，索引顺序：index[row][col]
    data = []
    with open(input_path, 'r') as f:
        csv_reader = csv.reader(f)  # 用reader是为了防止空行的出现
        for row in csv_reader:
            data.append(row)

    return data


def write_csv(data, output_path='test.csv'):  # list[list[]] -> .csv | 输入一个二维列表，生成其csv文件在output_path
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


# JSON------------------------------------------------------------------------------------------------------------------
def read_json(input_path):  # path -> dict | 输入文件路径，返回存储该json的字典
    with open(input_path, encoding='utf-8') as f:
        resp = json.load(f)
        return resp


def write_json(data, output_path='test.json'):  # dict -> json | 把字典写入本地json文件
    with open(output_path, "w", encoding='utf-8') as f:
        # json.dump(dict_var, f)  # 写为一行
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)  # 写为多行


# Read Data----------------------------------------------------------------------------------------------------------

# Modify Data----------------------------------------------------------------------------------------------------------------------
def arrange_raw():  # -> .csv | 整合原始数据，拿出需要的栏目放到一个文件中
    """ Arrange the raw volume data, get the label for machine learning. """
    all_data = []
    for file_name in os.listdir(transport_node_bus_folder):
        file = read_csv(f'{transport_node_bus_folder}/{file_name}')
        file.pop(0)
        all_data += file

    header = ['label_in', 'pt_code', 'time_per_hour', 'day_type']
    for i in range(len(all_data)):
        row = all_data[i]
        if row[1] == 'WEEKDAY':
            day_type = 1
        else:
            day_type = 0
        all_data[i] = [int(row[5])+int(row[6]), row[4], row[2], day_type]
    all_data.insert(0, header)

    write_csv(all_data, f'{process_data_folder}/1_raw_rearranged.csv')

    all_data = pd.read_csv(f'{process_data_folder}/1_raw_rearranged.csv')
    time = all_data.pop('time_per_hour')

    for i in range(24):
        all_data[f'time{i}'] = (time == i) * 1

    all_data.to_csv(f'{process_data_folder}/1_raw_rearranged.csv', index=False)

    return all_data


def assign_coordinates():  # BusStops.json & 1_raw_rearranged.csv -> .csv | add BusStop coords to the dataset, and remove the missing ones
    coord_dict = {}
    bus_stops = read_json(f'{raw_data_folder}/BusStops.json')
    for stop in bus_stops:
        stop_code = stop['BusStopCode']
        lat = stop['Latitude']
        lon = stop['Longitude']
        coord_dict[stop_code] = [lon, lat]

    all_data = read_csv(f'{process_data_folder}/1_raw_rearranged.csv')

    header = all_data.pop(0)
    header += ['longitude', 'latitude']
    code_index = header.index('pt_code')
    missing_stop = 0

    for i in range(len(all_data) - 1, -1, -1):
        row = all_data[i]
        stop_code = row[code_index]
        try:
            lon_lat = coord_dict[stop_code]
            all_data[i] += lon_lat
        except:
            missing_stop += 1
            all_data.pop(i)
            continue

    all_data.insert(0, header)

    write_csv(all_data, f'{process_data_folder}/2_bus_with_coords.csv')


def split_data(num=10000):  # bus_volume.csv(s) -> .csv | split the data into train and eval
    file_name = os.listdir(process_data_folder)[-1]
    all_data = read_csv(f'{process_data_folder}/{file_name}')

    eval_index = list(np.random.randint(1, len(all_data), num))
    eval_index.sort(reverse=True)

    eval = [all_data[0]]  # same header
    for i in eval_index:
        eval.append(all_data.pop(i))

    write_csv(eval, f'{tensorflow}/eval.csv')
    write_csv(all_data, f'{tensorflow}/train.csv')


def main_process():  # Main entrance of this script
    arrange_raw()
    print('FINISHED: arrange_raw()')

    assign_coordinates()
    print('FINISHED: assign_coordinates()')


if __name__ == "__main__":
    main_process()
