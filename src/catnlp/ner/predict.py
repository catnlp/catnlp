# -*- coding:utf-8 -*-

import logging

from .predict_plm import PredictPlm
from .predict_dict import PredictDict
from .predict_re import PredictRe
from .predict_plm_cmeee import PredictPlmCmeee
# from .train_lstm import LstmTrain


class NerPredict:
    def __init__(self, config, type):
        logging.info(config)
        if type == "plm":
            self.ner_service = PredictPlm(config)
        elif type == "dict":
            self.ner_service = PredictDict(config)
        elif type == "re":
            self.ner_service = PredictRe(config)
        elif type == "cmeee":
            self.ner_service = PredictPlmCmeee(config)
        # else:
        #     ner_train = LstmTrain(config)
        #     ner_train.train()
    def predict(self, text):
        return self.ner_service.predict(text)
