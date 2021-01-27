#!/usr/bin/python3
# -*- coding: utf-8 -*-


def merge_tag_lists(tag_lists, method="longest", weight_list=None, format="bio"):
    """
    合并标签序列组
    Args:
        tag_lists(list): 标签序列组
        method(str): 合并策略(longest|most|maximum|weighted)
        weight_list(list): 权重列表
        format(str): 标签序列格式(bio|bioes)
    Returns:
        tag_list(list): 标签列表
    """
    if not tag_lists:
        return None
    size = len(tag_lists[0])
    if size == 1:
        return tag_lists[0]
    scheduling_list = merge_strategy(tag_lists, method, weight_list, format)
    tag_list = recover_tag_list(scheduling_list, size, format)
    return tag_list


def recover_tag_list(scheduling_list, size, format):
    """
    从实体列表中恢复标签列表
    Args:
        scheduling_list(list): 去冲突后实体列表
        size(int): 标签列表长度
        format(str): 标签序列格式(bio|bioes)
    Returns:
        tag_list(list): 标签列表
    """
    if format == "bio":
        tag_list = recover_tag_list_bio(scheduling_list, size)
    elif format == "bioes":
        tag_list = recover_tag_list_bioes(scheduling_list, size)
    else:
        raise ValueError("select in (bio|bioes)")
    return tag_list


def recover_tag_list_bio(scheduling_list, size):
    """
    从实体列表中恢复bio标签列表
    Args:
        scheduling_list(list): 去冲突后实体列表
        size(int): 标签列表长度
    Returns:
        tag_list(list): bio标签列表
    """
    tag_list = ["O"] * size
    if scheduling_list:
        for scheduling in scheduling_list:
            start = scheduling["start"]
            end = scheduling["end"]
            tag = scheduling["tag"]
            tag_list[start] = f"B-{tag}"
            for i in range(start + 1, end):
                tag_list[i] = f"I-{tag}"
    return tag_list


def recover_tag_list_bioes(scheduling_list, size):
    """
    从实体列表中恢复bioes标签列表
    Args:
        scheduling_list(list): 去冲突后实体列表
        size(int): 标签列表长度
    Returns:
        tag_list(list): bioes标签列表
    """
    tag_list = ["O"] * size
    if scheduling_list:
        for scheduling in scheduling_list:
            start = scheduling["start"]
            end = scheduling["end"]
            tag = scheduling["tag"]
            if scheduling["len"] == 1:
                tag_list[start] = f"S-{tag}"
                continue
            tag_list[start] = f"B-{tag}"
            for i in range(start + 1, end - 1):
                tag_list[i] = f"I-{tag}"
            tag_list[end - 1] = f"E-{tag}"
    return tag_list


def merge_strategy(tag_lists, method, weight_list=None, format="bio"):
    """
    合并策略
    Args:
        tag_lists(list): 标签序列组
        method(str): 合并策略(longest|most|maximum|weighted)
        weight_list(list): 权重列表
        format(str): 标签序列格式(bio|bioes)
    Returns:
        scheduling_list(list): 去重后实体列表
    """
    interval_list = list()

    for i, tag_list in enumerate(tag_lists):
        if weight_list:
            interval_list.extend(get_interval(tag_list, weight_list[i], format))
        else:
            interval_list.extend(get_interval(tag_list, 1, format))

    if not interval_list:
        return None

    if method == "longest":
        scheduling_list = longest_interval_scheduling(interval_list)
    elif method == "most":
        scheduling_list = most_interval_scheduling(interval_list)
    elif method == "maximum":
        scheduling_list = maximum_interval_scheduling(interval_list)
    elif method == "weighted":
        scheduling_list = weighted_interval_scheduling(interval_list)
    else:
        raise ValueError("select in {longest|most|maximum|weighted}")

    return scheduling_list


def longest_interval_scheduling(interval_list):
    """
    最长区间调度：优先选择长度长的区间
    Args:
        interval_list(list): 区间列表
    Returns:
        scheduling_list(list): 去重实体列表
    """
    scheduling_list = list()
    sorted_interval_list = sorted(interval_list,
                                  key=lambda x: x['len'], reverse=True)
    scheduling_list.append(sorted_interval_list[0])
    compare_list = [[sorted_interval_list[0]['start'],
                    sorted_interval_list[0]['end']]]
    size = len(sorted_interval_list)
    for i in range(1, size):
        current = [sorted_interval_list[i]['start'],
                   sorted_interval_list[i]['end']]
        compare_list, is_overlap = extend_compare_list(current, compare_list)
        if not is_overlap:
            scheduling_list.append(sorted_interval_list[i])

    return scheduling_list


def extend_compare_list(interval, compare_list):
    for compare in compare_list:
        if interval[0] < compare[1] and \
                compare[0] < interval[1]:
            return compare_list, True
    compare_list.append(interval)
    return compare_list, False


def most_interval_scheduling(interval_list):
    """
    最多区间调度：优先选择'end'值小的区间
    Args:
        interval_list(list): 区间列表
    Returns:
        scheduling_list(list): 去重实体列表
    """
    scheduling_list = list()
    sorted_interval_list = sorted(interval_list,
                                  key=lambda x: x['end'])
    size = len(sorted_interval_list)
    scheduling_list.append(sorted_interval_list[0])
    for i in range(1, size):
        if scheduling_list[-1]['end'] <= sorted_interval_list[i]['start']:
            scheduling_list.append(sorted_interval_list[i])

    return scheduling_list


def maximum_interval_scheduling(interval_list):
    """
    最大区间调度
    Args:
        interval_list(list): 区间列表
    Returns:
        scheduling_list(list): 去重实体列表
    """
    scheduling_list = list()
    sorted_interval_list = sorted(interval_list,
                                  key=lambda x: x['end'])
    size = len(sorted_interval_list)
    max_value = [0] * size
    is_use = [False] * size
    last_idx = [-1] * size
    max_value[0] = sorted_interval_list[0]['len']
    is_use[0] = True

    for i in range(1, size):

        j = find_no_overlap(sorted_interval_list,
                            sorted_interval_list[i]['start'],
                            i)
        tmp_value = sorted_interval_list[i]['len']
        if j >= 0:
            tmp_value += max_value[j]
            last_idx[i] = j

        if tmp_value > max_value[i - 1]:
            max_value[i] = tmp_value
            is_use[i] = True
        else:
            max_value[i] = max_value[i - 1]
            last_idx[i] = i - 1

    idx = size - 1
    while idx >= 0:
        if is_use[idx]:
            scheduling_list.append(sorted_interval_list[idx])
        idx = last_idx[idx]

    return scheduling_list


def weighted_interval_scheduling(interval_list):
    """
    最大区间调度
    Args:
        interval_list(list): 区间列表
    Returns:
        scheduling_list(list): 去重实体列表
    """
    scheduling_list = list()
    sorted_interval_list = sorted(interval_list,
                                  key=lambda x: x['end'])
    size = len(sorted_interval_list)
    max_value = [0] * size
    is_use = [False] * size
    last_idx = [-1] * size
    max_value[0] = sorted_interval_list[0]['len'] * \
                   sorted_interval_list[0]['weight']
    is_use[0] = True

    for i in range(1, size):

        j = find_no_overlap(sorted_interval_list,
                            sorted_interval_list[i]['start'],
                            i)
        tmp_value = sorted_interval_list[i]['len'] * \
                    sorted_interval_list[i]['weight']
        if j >= 0:
            tmp_value += max_value[j]
            last_idx[i] = j

        if tmp_value > max_value[i - 1]:
            max_value[i] = tmp_value
            is_use[i] = True
        else:
            max_value[i] = max_value[i - 1]
            last_idx[i] = i - 1

    idx = size - 1
    while idx >= 0:
        if is_use[idx]:
            scheduling_list.append(sorted_interval_list[idx])
        idx = last_idx[idx]

    return scheduling_list


def find_no_overlap(interval_list, start, size):
    left = -1
    right = size
    while right >= left:
        mid = int((left + right) / 2)
        if interval_list[mid]['end'] > start:
            right = mid - 1
        else:
            left = mid + 1
    return left - 1


def find_no_overlap1(interval_list, start, size):
    for i in range(size - 1, -1, -1):
        if interval_list[i]["end"] <= start:
            return i
    return -1


def get_interval(tag_list, weight=1, format="bio"):
    if format == "bio":
        entity_list = get_interval_bio(tag_list, weight)
    elif format == "bioes":
        entity_list = get_interval_bioes(tag_list, weight)
    else:
        raise ValueError(f"错误格式：{format}")
    return entity_list


def get_interval_bio(tag_list, weight):
    entities = []
    pre_o = True
    for idx, tag in enumerate(tag_list):
        if tag != "O":
            prefix, tname = tag.split("-")
            if prefix == "B":
                entities.append({
                    "start": idx,
                    "end": idx + 1,
                    "len": 1,
                    "tag": tname,
                    "weight": weight
                })
            elif prefix == "I":
                if pre_o or \
                        (entities and entities[-1]["tag"] != tname):
                    entities.append({
                        "start": idx,
                        "end": idx + 1,
                        "len": 1,
                        "tag": tname,
                        "weight": weight
                    })
                else:
                    entities[-1]["end"] += 1
                    entities[-1]["len"] += 1
            else:
                print(f"error tag: {tag}")
                exit(1)
            pre_o = False
        else:
            pre_o = True
    return entities


def get_interval_bioes(tag_list, weight):
    entities = []
    pre_o = True
    for idx, tag in enumerate(tag_list):
        if tag != "O":
            prefix, tname = tag.split("-")
            if prefix in ["S", "B"]:
                entities.append({
                    "start": idx,
                    "end": idx + 1,
                    "len": 1,
                    "tag": tname,
                    "weight": weight
                })
            elif prefix in ["I", "E"]:
                if pre_o or \
                        (entities and entities[-1]["tag"] != tname):
                    entities.append({
                        "start": idx,
                        "end": idx + 1,
                        "len": 1,
                        "tag": tname,
                        "weight": weight
                    })
                else:
                    entities[-1]["end"] += 1
                    entities[-1]["len"] += 1
            else:
                print(f"error tag: {tag}")
                exit(1)
            pre_o = False
        else:
            pre_o = True
    return entities


def get_interval(tag_list, weight=None, format="bio"):
    """
    标签列表转实体区间列表
    :param tag_list:
    :return:
    """
    if format == "bio":
        return get_interval_bio(tag_list, weight)
    elif format == "bioes":
        return get_interval_bioes(tag_list, weight)
    else:
        raise ValueError("select in {bio|bioes}")


if __name__ == "__main__":
    tag_lists = [["B-a", "I-a", "O", "O", "B-d"],
                 ["O", "B-b", "I-b", "I-b", "O"],
                 ["O", "O", "B-c", "O", "O"]]
    weight_list = [1, 2, 3]
    for tag_list in tag_lists:
        print(tag_list)
    result_longest = merge_tag_lists(tag_lists, method="longest")
    result_most = merge_tag_lists(tag_lists, method="most")
    result_maximum = merge_tag_lists(tag_lists, method="maximum")
    result_weighted = merge_tag_lists(tag_lists, method="weighted",
                                     weight_list=weight_list)
    print(f"longest: {result_longest}")
    print(f"most: {result_most}")
    print(f"maximum: {result_maximum}")
    print(f"weighted: {result_weighted}")
