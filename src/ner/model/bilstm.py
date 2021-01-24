# -*- coding: utf-8 -*-
# @Author  : catnlp
# @FileName: bilstm_crf.py
# @Time    : 2020/2/27 22:54

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, \
    pad_packed_sequence, pad_sequence


class BiLSTM(nn.Module):
    """
    BiLSTM_CRF模型
    """
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.word_size = config['word_size']
        self.word_dim = config['word_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layer = config['num_layer']
        dropout = config['dropout']
        # self.dropout = nn.Dropout(dropout)
        self.rnn_input_dropout = nn.Dropout(dropout)
        self.rnn_output_dropout = nn.Dropout(dropout)
        self.labels_size = config['labels_size']
        self.nlabel = config['nlabel']
        self.extend = config['extend']

        self.embedding = nn.Embedding(
            self.word_size,
            self.word_dim,
            padding_idx=0).\
            from_pretrained(torch.from_numpy(config['word2vec'])).float()
        self.lstm = nn.LSTM(self.word_dim, self.hidden_dim // 2,
                            num_layers=self.num_layer,
                            # dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        # self.norm = nn.LayerNorm(self.hidden_dim)
        self.hidden2tag_list = nn.ModuleList()
        self.hidden2tag_list.append(nn.Linear(self.hidden_dim,
                                              self.labels_size[0]))
        extend_size = 1 if self.extend else 0
        for i in range(1, self.nlabel):
            hidden2tag = nn.Linear(self.hidden_dim + extend_size, self.labels_size[i])
            self.hidden2tag_list.append(hidden2tag)

        self.fct_loss = nn.CrossEntropyLoss(ignore_index=0)

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
        emission = self.hidden2tag_list[0](lstm_output)
        seqs_list = []
        seqs = torch.argmax(emission, dim=2)
        seqs_list.append(seqs)
        device = lstm_output.device
        for i in range(1, self.nlabel):
            if self.extend:
                seq_list = \
                    pad_sequence([torch.tensor(seq) for seq in seqs_list[-1]],
                                 batch_first=True,
                                 padding_value=0).to(device)
                seq_list = torch.unsqueeze(seq_list, dim=2).float()
                lstm_output = torch.cat((lstm_output, seq_list), dim=2)
            emission = self.hidden2tag_list[i](lstm_output)
            seqs = torch.argmax(emission, dim=2)
            seqs_list.append(seqs)
        return seqs_list, len_list

    def calculate_loss(self, word_batch, labels_batch):
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
        emission = self.hidden2tag_list[0](lstm_output)
        active_masks = masks.reshape(-1)
        active_logits = emission.reshape(-1, self.labels_size[0])[active_masks]
        active_labels = labels_batch[0].reshape(-1)[active_masks]
        loss = self.fct_loss(active_logits, active_labels)

        for i in range(1, self.nlabel):
            if self.extend:
                label_batch = torch.unsqueeze(labels_batch[i - 1], dim=2).float()
                lstm_output = \
                    torch.cat((lstm_output, label_batch), dim=2)
            emission = self.hidden2tag_list[i](lstm_output)
            active_logits = emission.reshape(-1, self.labels_size[i])[active_masks]
            active_labels = labels_batch[i].reshape(-1)[active_masks]
            loss += self.fct_loss(active_logits, active_labels)
        return loss
