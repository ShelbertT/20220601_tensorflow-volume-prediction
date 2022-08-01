"""
This script stores the global variables for inter-script referencing
"""


def _init():
    global global_dict
    global_dict = {}


def set_value(key, value):
    """ 定义一个全局变量 """
    global_dict[key] = value


def get_value(key, def_value=None):
    """ 获得一个全局变量,不存在则返回默认值 """
    try:
        return global_dict[key]
    except KeyError:
        return def_value


_init()
set_value('raw_data_folder', './database/1_raw')
set_value('process_data_folder', './database/2_process')
set_value('tensorflow_folder', './database/3_tensorflow')

raw_data_folder = get_value('raw_data_folder')
process_data_folder = get_value('process_data_folder')
tensorflow = get_value('tensorflow_folder')

set_value('transport_node_bus_folder', f'{raw_data_folder}/volume')
