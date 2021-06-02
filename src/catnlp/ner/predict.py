# -*- coding:utf-8 -*-

import logging

from .predict_pretrained_crf import PretrainedCrfPredict
from .predict_pretrained_softmax import PretrainedSoftmaxPredict
# from .train_lstm import LstmTrain


class NerPredict:
    def __init__(self, config):
        logging.info(config)
        model_type = config.get("type", "").lower()
        if model_type == "pretrained_softmax":
            self.ner_service = PretrainedSoftmaxPredict(config)
        elif model_type == "pretrained_crf":
            self.ner_service = PretrainedCrfPredict(config)
        # else:
        #     ner_train = LstmTrain(config)
        #     ner_train.train()
    def predict(self, text):
        return self.ner_service.predict(text)
