# -*- coding: utf-8 -*-

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""
import re
import logging
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer
)
import numpy as np

from ..common.load_file import load_label_file
from .model.albert_tiny import AlbertTinyCrf, AlbertTinySoftmax
from .model.bert import BertBiaffine, BertCrf, BertSoftmax, BertLstmCrf
from .util.tokenizer import NerBertTokenizer
from .util.split import merge_entities


logger = logging.getLogger(__name__)


class PredictPlm:
    def __init__(self, config) -> None:
        self.max_seq_length = config.get("max_length")
        self.do_lower = config.get("do_lower_case")
        label_file = Path(config.get("model_path")) / "label.txt"
        self.label_list = load_label_file(label_file)
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        print(self.label_to_id)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_path"), use_fast=True)
        pretrained_config = AutoConfig.from_pretrained(config.get("model_path"), num_labels=len(self.label_list))

        model_func = None
        model_name = config.get("name").lower()
        if model_name == "bert_crf":
            model_func = BertCrf
        elif model_name == "bert_softmax":
            model_func = BertSoftmax
        elif model_name == "bert_lstm_crf":
            model_func = BertLstmCrf
        elif model_name == "bert_biaffine":
            model_func = BertBiaffine
        elif model_name == "albert_tiny_crf":
            model_func = AlbertTinyCrf
        elif model_name == "albert_tiny_softmax":
            model_func = AlbertTinySoftmax
        else:
            raise ValueError
        pretrained_config.loss_name = None
        self.model = model_func.from_pretrained(
            config.get("model_path"),
            config=pretrained_config
        )
        self.device = config.get("device")
        self.model.to(torch.device(self.device))
        self.model.eval()
        self.decode_type = config.get("decode_type")
    
    def get_labels(self, predictions, masks):
        # Transform predictions and references tensos to numpy arrays
        if self.device == "cpu":
            y_pred = predictions.detach().clone().numpy()
            masks = masks.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            masks = masks.detach().cpu().clone().numpy()
        if self.decode_type == "general":
            return self.get_general_labels(y_pred, masks)
        elif self.decode_type == "biaffine":
            return self.get_biaffine_labels(y_pred, masks)
        else:
            raise ValueError

    
    def get_general_labels(self, y_pred, masks):
        preds = list()
        for pred, mask in zip(y_pred, masks):
            tmp_preds = list()
            for p, m in zip(pred, mask):
                if m == 0:
                    continue
                if p == 0:
                    tmp_preds.append("O")
                else:
                    tmp_preds.append(self.label_list[p])
            preds.append(tmp_preds)
        return preds


    def get_biaffine_labels(self, y_pred, masks):
        preds = list()
        for pred, mask in zip(y_pred, masks):
            pred_entities = list()
            offset_dict = dict()
            count = -1
            for idx, m in mask:
                if idx == 0:
                    continue
                if m == 1:
                    count += 1
                offset_dict[idx] = count
            max_len = len(mask)
            for i in range(1, max_len):
                for j in range(i, max_len):
                    pred_scores = pred[i][j]
                    pred_label_id = np.argmax(pred_scores)
                    start_idx = offset_dict[i]
                    end_idx = offset_dict[j] + 1
                    if pred_label_id > 0:
                        pred_entities.append([start_idx, end_idx, self.label_list[pred_label_id], pred_scores[pred_label_id]])

            pred_entities = sorted(pred_entities, reverse=True, key=lambda x:x[3])
            new_pred_entities = list()
            for pred_entity in pred_entities:
                start, end, tag, _ = pred_entity
                flag = True
                for new_pred_entity in new_pred_entities:
                    new_start, new_end, _ = new_pred_entity
                    if start < new_end and new_start < end:
                        #for flat ner nested mentions are not allowed
                        flag = False
                        break
                if flag:
                    new_pred_entities.append([start, end, tag])
            pred_entities = new_pred_entities
            count += 1
            tmp_preds = ["O"] * (count)
            for entity in pred_entities:
                start, end, tag = entity
                tmp_preds[start] = f"B-{tag}"
                for i in range(start+1, end):
                    tmp_preds[i] = f"I-{tag}"
            preds.append(tmp_preds)
        return preds
    
    def predict(self, text):
        inputs, masks = self.preprocess([text])
        outputs = self.model(**inputs)
        entity_lists = self.postprocess([text], outputs, masks)
        return entity_lists[0]
    
    def preprocess(self, text_list):
        input_ids, input_masks, input_len, masks = self._to_features(text_list, self.tokenizer, self.max_seq_length)
        inputs = {"input_ids": input_ids, "attention_mask": input_masks, "input_len": input_len}
        return inputs, masks

    def postprocess(self, texts, outputs, masks):
        entity_lists = list()
        pred_lists = self.get_labels(outputs, masks)
        for text, pred_list in zip(texts, pred_lists):
            entity_list = self.get_entity_list(text, pred_list)
            entity_lists.append(entity_list)
        return entity_lists
    
    def get_entity_list(self, text, tag_list):
        entity_list = list()
        pre_label = "O"
        for idx, (_, tag) in enumerate(zip(text, tag_list)):
            if tag[0] != "O":
                tag_name = tag[2:]
                if tag[0] in ["B", "S"]:
                    entity_list.append([idx, idx+1, tag_name])
                elif pre_label == tag_name:
                    entity_list[-1][1] += 1
                else:
                    entity_list.append([idx, idx+1, tag_name])
                pre_label = tag_name
            else:
                pre_label = "O"
        return entity_list
    
    def tokenize(self, words, format="bio"):
        if format == "bio":
            return self.tokenize_bio(words)
        elif format == "bies":
            return self.tokenize_bies(words)
        elif format == "biaffine":
            return self.tokenize_biaffine(words)
        else:
            raise ValueError

    def tokenize_bio(self, words):
        _tokens = list()
        _masks = list()
        for word in words:
            if self._do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                raise ValueError
            for idx, tmp_token in enumerate(tmp_tokens):
                _tokens.append(tmp_token)
                if idx == 0:
                    _masks.append(1)
                else:
                    _masks.append(0)
        return _tokens, _masks
    
    def tokenize_bies(self, words):
        _tokens = list()
        _masks = list()
        for word in words:
            if self._do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                raise ValueError
            if len(tmp_tokens) == 1:
                _tokens.append(tmp_tokens[0])
                _masks.append(1)
                continue
            for idx, tmp_token in enumerate(tmp_tokens):
                _tokens.append(tmp_token)
                if idx == 0:
                    _masks.append(1)
                else:
                    _masks.append(0)
        return _tokens, _masks
    
    def tokenize_biaffine(self, words):
        _tokens = list()
        _masks = list()
        for idx, word in enumerate(words):
            if self._do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                raise ValueError
            if len(tmp_tokens) == 1:
                _tokens.append(tmp_tokens[0])
                _masks.append(1)
                continue
            for idx, tmp_token in enumerate(tmp_tokens):
                _tokens.append(tmp_token)
                if idx == 0:
                    _masks.append(1)
                else:
                    _masks.append(0)

        return _tokens, _masks
    
    def _to_features(self, words_list, tokenizer=None, max_seq_length=-1,
                     cls_token_at_end=False,cls_token="[CLS]",
                     sep_token="[SEP]",pad_on_left=False,
                     pad_token="[PAD]",mask_padding_with_zero=True,):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        input_id_lists = list()
        input_mask_lists = list()
        input_len_list = list()
        masks_list = list()
        for words in words_list:
            tokens, masks = self.tokenize(words, format=self.decode_type)
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                masks = masks[: (max_seq_length - special_tokens_count)]

            pad_id = self.label_to_id.get(pad_token)
            tokens += [sep_token]
            masks += [0]

            tokens = [cls_token] + tokens
            masks = [0] + masks

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_len = len(input_ids)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_masks = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_id] * padding_length
            input_masks += [0 if mask_padding_with_zero else 1] * padding_length
            masks += [0] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_masks) == max_seq_length
            assert len(masks) == max_seq_length
            input_id_lists.append(input_ids)
            input_mask_lists.append(input_masks)
            input_len_list.append(input_len)
            masks_list.append(masks)

        input_ids = torch.tensor(input_id_lists, dtype=torch.long).to(torch.device(self.device))
        input_masks = torch.tensor(input_mask_lists, dtype=torch.long).to(torch.device(self.device))
        input_len = torch.tensor(input_len_list, dtype=torch.long).to(torch.device(self.device))
        masks = torch.tensor(masks_list, dtype=torch.long).to(torch.device(self.device))
        return input_ids, input_masks, input_len, masks
