# -*- coding:utf-8 -*-

import logging
import random

import numpy as np
import torch

from .train_plm import PlmTrain
from .train_lstm import LstmTrain


class NerTrain:
    def __init__(self, config):
        logging.info(config)
        self.setup_seed(seed=config.get("seed"))
        model_type = config.get("type", "").lower()
        if model_type == "plm":
            ner_train = PlmTrain(config)
        else:
            ner_train = LstmTrain(config)
            ner_train.train()

    def setup_seed(self, seed):
        if not seed:
            seed = 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
