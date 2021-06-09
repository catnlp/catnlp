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


def load_label_file(label_file):
    """
    """
    label_list = list()
    with open(label_file, "r", encoding="utf-8") as lf:
        for line in lf:
            line = line.rstrip()
            if not line:
                continue
            label_list.append(line)
    return label_list


def load_dict_file(dict_file):
    """
    """
    word_list = list()
    with open(dict_file, "r", encoding="utf-8") as lf:
        for line in lf:
            line = line.rstrip()
            if not line:
                continue
            word_list.append(line)
    return word_list


def load_re_file(re_file):
    """
    """
    pattern_list = list()
    with open(re_file, "r", encoding="utf-8") as lf:
        for line in lf:
            line = line.rstrip()
            if not line:
                continue
            pattern_list.append(line)
    pattern_str = "|".join(pattern_list)
    return pattern_str
