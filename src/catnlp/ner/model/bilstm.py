# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, \
    pad_packed_sequence

from ...layer.decoder.crf import CRF


class BiLstmSoftmax(nn.Module):
    """
    BiLSTM_Softmax模型
    """
    def __init__(self, config):
        super(BiLstmSoftmax, self).__init__()
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
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, word_batch, label_batch=None):
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
        if label_batch:
            active_masks = masks.reshape(-1)
            active_logits = emission.reshape(-1, self.label_size)[active_masks]
            active_labels = label_batch.reshape(-1)[active_masks]
            loss = self.loss(active_logits, active_labels)
            return loss
        else:
            seq_list = torch.argmax(emission, dim=2)
            return seq_list, len_list


class BiLstmCrf(nn.Module):
    """
    BiLSTM_CRF模型
    """
    def __init__(self, config):
        super(BiLstmCrf, self).__init__()
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

    def forward(self, word_batch, label_batch=None):
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
        if label_batch is not None:
            loss = self.crf(emission, label_batch, masks)
            return -loss
        else:
            seq_list = self.crf.decode(emission, masks)
            return seq_list, len_list
