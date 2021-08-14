# -*- coding: utf-8 -*-

import ahocorasick

from ..common.load_file import load_dict_file


class PredictDict:
    def __init__(self, config) -> None:
        self.tag_service = dict()
        for tag in config:
            self.tag_service[tag] = ahocorasick.Automaton()
            tag_file = config[tag]
            dict_list = load_dict_file(tag_file)
            for word in dict_list:
                self.tag_service[tag].add_word(word, word)
            self.tag_service[tag].make_automaton()


    def predict(self, text):
        entity_list = list()
        for tag in self.tag_service:
            tmp_entities = list(self.tag_service[tag].iter_long(text))
            for tmp_entity in tmp_entities:
                end, word = tmp_entity
                end += 1
                start = end - len(word)
                word = text[start: end]
                entity_list.append([start, end, tag, word])
        return entity_list
