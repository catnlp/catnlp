# -*- coding: utf-8 -*-

import re
from transformers import BertTokenizerFast


class NerBertTokenizer(BertTokenizerFast):
    def __init__(self, vocab_file, do_lower_case=False):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if re.match(r"\s", c):
                _tokens.append("[unused1]")
            else:
                _tokens.append(c)
        return _tokens
