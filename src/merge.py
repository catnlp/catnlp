import json
from pathlib import Path
from collections import defaultdict


def merge_files(sources, target):
    with open(target, "w", encoding="utf-8") as tf:
        sfs = list()
        for source in sources:
            sf = open(source, "r", encoding="utf-8")
            sfs.append(sf)

        for lines in zip(*sfs):
            entity_dict = defaultdict(int)
            idx = None
            text = None
            for line in lines:
                line = line.rstrip()
                idx, text, tags = line.split("\u0001")
                entities = get_interval(tags.split(), format="bies")
                for entity in entities:
                    entity_dict[tuple(entity)] += 1
            entity_list = list()
            for entity in entity_dict:
                entity_list.append([entity[0], entity[1], entity[2], entity_dict[entity]])
            entities = merge_entities(entity_list)
            tag_list = get_tag_list(text, entities)
            tf.write(f"{idx}\u0001{text}\u0001{' '.join(tag_list)}\n")


def get_tag_list(text, entity_list):
    tag_list = ["O"] * len(text)
    for entity in entity_list:
        start, end, tag, _ = entity
        if end - start == 1:
            # if tag in ["assist", "intersection"]:
            tag_list[start] = f"S-{tag}"
        else:
            tag_list[start] = f"B-{tag}"
            for i in range(start+1, end-1):
                tag_list[i] = f"I-{tag}"
            tag_list[end-1] = f"E-{tag}"
    return tag_list


def get_interval(tag_list, format="bio"):
    if format == "bio":
        entity_list = get_interval_bio(tag_list)
    elif format == "bies":
        entity_list = get_interval_bies(tag_list)
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
                entities.append([idx, idx+1, tname])
            elif prefix == "I":
                if pre_o or \
                        (entities and entities[-1][-1] != tname):
                    entities.append([idx, idx+1, tname])
                else:
                    entities[-1][1] += 1
            else:
                print(f"error tag: {tag}")
                exit(1)
            pre_o = False
        else:
            pre_o = True
    return entities


def get_interval_bies(tag_list):
    entities = []
    pre_o = True
    for idx, tag in enumerate(tag_list):
        if tag != "O":
            prefix, tname = tag.split("-")
            if prefix in ["S", "B"]:
                entities.append([idx, idx+1, tname])
            elif prefix in ["I", "E"]:
                if pre_o or \
                        (entities and entities[-1][-1] != tname):
                    entities.append([idx, idx+1, tname])
                else:
                    entities[-1][1] += 1
            else:
                print(prefix)
                print(f"error tag: {tag}")
                exit(1)
            pre_o = False
        else:
            pre_o = True
    return entities


def merge_entities(entity_list):
    sorted_entity_list = sorted(entity_list, key=lambda i: (i[2], i[1]-i[0]), reverse=True)
    new_entity_list = list()
    is_appear_list = [False] * len(entity_list)
    for idx, entity in enumerate(sorted_entity_list):
        if is_appear_list[idx]:
            continue
        new_entity_list.append(entity)
        for idy, tmp_entity in enumerate(sorted_entity_list[idx:]):
            if entity[0] < tmp_entity[1] and \
                    tmp_entity[0] < entity[1]:
                is_appear_list[idx+idy] = True
    return new_entity_list


if __name__ == "__main__":
    source_path = Path("resources/data/dataset/ner/zh/ccks/address/0621/merge")
    file_names = ["试一下_addr_parsing_runid_9094", "试一下_addr_parsing_runid_9092", "试一下_addr_parsing_runid_9090"] #, "试一下_addr_parsing_runid_9045", "试一下_addr_parsing_runid_9034"]
    source_files = list()
    for file_name in file_names:
        source_file = source_path / f"{file_name}.txt"
        source_files.append(source_file)
    target_file = source_path / f"试一下_addr_parsing_runid.txt"
    merge_files(source_files, target_file)
