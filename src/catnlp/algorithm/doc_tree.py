#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re


class DocNode:
    def __init__(self, text="", start=-1, type="paragraph", children=list()):
        self.texts = [text] if text else list()
        self.start = start
        self.type = type
        self.children = children


class DocTree:
    def __init__(self, text):
        self._start = r"^\s*(第\s*一\s*节|一\s*、)"
        self._chinese = r"[\u4e00-\u9fa5]+"
        self._contents = r"^\s*目\s*录\s*$"
        self._heading_1 = r"^\s*(第\s*[一二三四五六七八九十]+\s*节|[一二三四五六七八九十]+\s*、)"
        self._heading_2 = r"^\s*(\(|（)\s*[一二三四五六七八九十]+\s*(\)|）))"
        self._heading_3 = r"^\s*[0-9]+、"
        self._root = self._build_tree(text)

    def _build_tree(self, text):
        root = DocNode(type="root")
        start = 0
        idx = 0
        line_list = text.split("\n")
        contents_flag = True
        start_flag = True
        for i, line in enumerate(line_list):
            start += len(line) + 1
            if re.search(self._contents, line):
                contents_flag = True
            if not contents_flag:
                continue
            if re.search(self._start, line):
                if start_flag:
                   start_flag = False
                else:
                    start -= (line(line) + 1)
                    idx = i
                    break
            if re.search(self._heading_1, line):
                line_str = re.search(self._heading_1 + r"\s*" + self._chinese, line).group()
                node = DocNode(text=line_str, type="heading_1")
                root.children.append(node)
        
        for line in line_list[idx:]:
            




        return root        

    def dfs(self):
        pass
    
    def bfs(self):
        pass
