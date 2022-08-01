"""
This script provides the functions for acquiring raw data
"""
import requests
from globalVar import *
from processBusData import *


raw_data_folder = get_value('raw_data_folder')


# 本脚本通用数据交换格式：JSON <--> list嵌套dict
def get_raw_data(name):  # string -> list[dict] | 到网站读取原始数据
    headers = {'AccountKey': 'WJbz7bLWSnWizpfxHFKhHg==', 'accept': 'application/json'}
    if name == 'StopVolume':
        url = f'http://datamall2.mytransport.sg/ltaodataservice/PV/Bus?Date=20220531'
        resp = requests.get(url, headers=headers)
        print(resp)
        final_response = resp
    else:
        response_number = 500
        skip = 0
        final_response = []
        while response_number == 500:
            url = f'http://datamall2.mytransport.sg/ltaodataservice/{name}?&$skip={skip}'
            resp = requests.get(url, headers=headers)
            value = resp.json().get("value")  # value是一个由多个dict元素组成的list

            final_response.extend(value)
            response_number = len(value)
            skip += 500

    return final_response  # 把所有的value都加进同一个list中进行返还


def write_data(data_name):  # string -> .json| 在线获取实时数据。这是一个总入口，用来获取任何所需要格式的数据
    """
    首先更新可选参数：BusServices, BusRoutes, BusStops
    BusServices: Returns detailed service information for all buses currently in operation, including: first stop, last stop, peak / offpeak frequency of dispatch.
    BusRoutes: Returns detailed route information for all services currently in operation, including: all bus stops along each route, first/last bus timings for each stop.
    BusStops: Returns detailed information for all bus stops currently being serviced by buses, including: Bus Stop Code, location coordinates.
    """

    data = get_raw_data(f'{data_name}')
    write_json(data, f'{raw_data_folder}/{data_name}')


def main_update():  # Main entrance of this script
    write_data('BusStops')
    print('FINISHED: write_data(BusStops)')

    # write_data('BusRoutes')


if __name__ == "__main__":
    main_update()

