#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, \
    pad_packed_sequence, pad_sequence

from ...layer.decoder import CRF


class BiLSTM_CRF(nn.Module):
    """
    BiLSTM_CRF模型
    """
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.word_size = config['word_size']
        self.word_dim = config['word_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layer = config['num_layer']
        self.label_size = config['label_size']
        dropout = config['dropout']

        self.rnn_input_dropout = nn.Dropout(dropout)
        self.rnn_output_dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(
            self.word_size,
            self.word_dim,
            padding_idx=0).\
            from_pretrained(torch.from_numpy(config['embed'])).float()
        self.lstm = nn.LSTM(self.word_dim, self.hidden_dim // 2,
                            num_layers=self.num_layer,
                            # dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.label_size)
        self.crf = CRF(self.label_size, batch_first=True)

    def forward(self, word_batch):
        masks = word_batch.gt(0)
        len_list = masks.sum(dim=1)
        word_embeds = self.embedding(word_batch)
        packed_words = pack_padded_sequence(word_embeds,
                                            len_list,
                                            batch_first=True,
                                            enforce_sorted=False)
        lstm_output, _ = self.lstm(packed_words)
        lstm_output, _ = pad_packed_sequence(lstm_output)
        lstm_output = lstm_output.transpose(1, 0)
        emission = self.hidden2tag(lstm_output)
        seq_list = self.crf.decode(emission, masks)
        return seq_list, len_list

    def calculate_loss(self, word_batch, label_batch):
        masks = word_batch.gt(0)
        len_list = masks.sum(dim=1)
        word_embeds = self.embedding(word_batch)
        word_embeds = self.rnn_input_dropout(word_embeds)
        packed_words = pack_padded_sequence(word_embeds,
                                            len_list,
                                            batch_first=True,
                                            enforce_sorted=False)
        lstm_output, _ = self.lstm(packed_words)
        lstm_output, _ = pad_packed_sequence(lstm_output)
        lstm_output = lstm_output.transpose(1, 0)
        lstm_output = self.rnn_output_dropout(lstm_output)
        # lstm_output = self.norm(lstm_output)
        emission = self.hidden2tag(lstm_output)
        loss = self.crf(emission, label_batch, masks)
        return -loss
