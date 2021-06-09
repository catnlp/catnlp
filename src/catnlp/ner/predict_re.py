# -*- coding: utf-8 -*-

import re

from ..common.load_file import load_re_file


class PredictRe:
    def __init__(self, config) -> None:
        self.tag_service = dict()
        for tag in config:
            re_file = config[tag]
            self.tag_service[tag] = load_re_file(re_file)


    def predict(self, text):
        entity_list = list()
        for tag in self.tag_service:
            matches = re.finditer(self.tag_service[tag], text)
            if matches:
                for match in matches:
                    start, end = match.span()
                    entity_list.append([start, end, tag])
        return entity_list
