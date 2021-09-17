# -*- coding:utf-8 -*-

import logging

from .fusion_plm_cmeee import FusionPlmCmeee
# from .train_lstm import LstmTrain


class NerFusion:
    def __init__(self, config):
        logging.info(config)
        self.ner_service = FusionPlmCmeee(config)
        # else:
        #     ner_train = LstmTrain(config)
        #     ner_train.train()
    def predict(self, text):
        return self.ner_service.predict(text)
