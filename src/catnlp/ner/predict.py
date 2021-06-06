# -*- coding:utf-8 -*-

import logging

from .predict_plm import PlmPredict
# from .train_lstm import LstmTrain


class NerPredict:
    def __init__(self, config):
        logging.info(config)
        model_type = config.get("type", "").lower()
        if model_type == "plm":
            self.ner_service = PlmPredict(config)
        # else:
        #     ner_train = LstmTrain(config)
        #     ner_train.train()
    def predict(self, text):
        return self.ner_service.predict(text)
