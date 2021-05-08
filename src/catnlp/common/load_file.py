# -*- coding: utf-8 -*-

import yaml


def load_config_file(config_file):
    """
    加载配置文件
    Args:
        config_file(str): 配置文件路径
    Returns:
        config_dict(dict): 配置词典
    """
    with open(config_file, 'r', encoding='utf-8') as rf:
        config_dict = yaml.load(rf, Loader=yaml.FullLoader)
    return config_dict
