# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict
from treelib import Node, Tree
from pathlib import Path


def build_tree(source, target_dir):
    id_to_value = dict()
    parent_to_child = defaultdict(set)
    child_to_parent = defaultdict(set)
    with open(source, "r", encoding="utf-8") as sf:
        content = json.loads(sf.read())
        datas = content.get("@graph")
        for idx, data in enumerate(datas):
            if (idx + 1) % 100 == 0:
                print(f"index: {idx}")
            index = data.get("@id")
            value = data.get("label").get("@value")
            id_to_value[index] = value
            parents = data.get("subClassOf")
            if isinstance(parents, str):
                parent_to_child[parents].add(index)
                child_to_parent[index].add(parents)
            elif isinstance(parents, list):
                for parent in parents:
                    parent_to_child[parent].add(index)
                    child_to_parent[index].add(parent)
    root_set = get_root(child_to_parent)
    root_list = sorted(list(root_set))
    for root in root_list:
        print(id_to_value.get(root))
    
    target_dir = Path(target_dir)
    os.makedirs(target_dir)
    for root in root_list:
        tree = Tree()
        find_set = set()
        tree.create_node(id_to_value.get(root), root)
        tree = traverse_tree(tree, root, parent_to_child, id_to_value, find_set)
        tree.show()
        target_file = target_dir / f"{id_to_value.get(root)}.txt"
        tree.save2file(target_file)
        


def traverse_tree(tree, root, parent_to_child, id_to_value, find_set):
    if not root:
        return
    for child in parent_to_child[root]:
        if not child or child in find_set:
            continue
        tree.create_node(id_to_value.get(child), child, parent=root)
        find_set.add(child)
        tree = traverse_tree(tree, child, parent_to_child, id_to_value, find_set)
    return tree

def get_root(child_to_parent):
    root_set = set()
    is_find = set()
    find_set = set()
    for child in child_to_parent:
        find_set |= child_to_parent.get(child)
        is_find.add(child)
    while find_set:
        for child in list(find_set):
            if child in is_find:
                find_set.remove(child)
                continue
            is_find.add(child)
            if child not in child_to_parent:
                root_set.add(child)
            else:
                find_set |= child_to_parent.get(child)
    return root_set


if __name__ == "__main__":
    source = "bigcilin_schema.json"
    target = "data"
    build_tree(source, target)
