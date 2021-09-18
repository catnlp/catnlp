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
import os
import re
import logging
from pathlib import Path

import torch
from torch._C import device
from transformers import (
    AutoConfig,
    AutoTokenizer
)
import numpy as np

from ..common.load_file import load_label_file
from .model.albert_tiny import AlbertTinyCrf, AlbertTinySoftmax
from .model.bert import BertBiaffine, BertCrf, BertMultiAddBiaffine, BertMultiHiddenBiaffine, BertMultiBiaffine, BertSoftmax, BertLstmCrf, BertSpan
from .util.tokenizer import NerBertTokenizer


logger = logging.getLogger(__name__)


class FusionPlmCmeee:
    def __init__(self, config) -> None:
        self.max_seq_length = config.get("max_length")
        self.do_lower = config.get("do_lower_case")
        label_file = Path(config.get("model_path")) / "label.txt"
        self.label_list = load_label_file(label_file)
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        print(self.label_to_id)
        self.tokenizer = None
        self.pretrained_config = None

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
        elif model_name == "bert_multi_biaffine":
            model_func = BertMultiBiaffine
        elif model_name == "bert_multi_add_biaffine":
            model_func = BertMultiAddBiaffine
        elif model_name == "bert_multi_hidden_biaffine":
            model_func = BertMultiHiddenBiaffine
        elif model_name == "bert_span":
            model_func = BertSpan
        elif model_name == "albert_tiny_crf":
            model_func = AlbertTinyCrf
        elif model_name == "albert_tiny_softmax":
            model_func = AlbertTinySoftmax
        else:
            raise ValueError
        self.models = list()
        self.k = config.get("k")
        self.device = config.get("device")
        for i in range(self.k):
            model_path = os.path.join(config.get("model_path"), str(i))
            if not self.tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            if not self.pretrained_config:
                self.pretrained_config = AutoConfig.from_pretrained(model_path, num_labels=len(self.label_list))
                self.pretrained_config.loss_name = None
            model = model_func.from_pretrained(
                model_path,
                config=self.pretrained_config
            )
            model.to(torch.device(self.device))
            model.eval()
            self.models.append(model)
        self.decode_type = config.get("decode_type")
        self.split = config.get("split")
    
    def get_labels(self, predictions, masks):
        if self.decode_type == "span":
            start_pred, end_pred = predictions
        if self.device == "cpu":
            if self.decode_type == "span":
                start_pred = start_pred.detach().clone().numpy()
                end_pred = end_pred.detach().clone().numpy()
            else:
                y_pred = predictions.detach().clone().numpy()
            masks = masks.detach().clone().numpy()
        else:
            if self.decode_type == "span":
                start_pred = start_pred.detach().cpu().clone().numpy()
                end_pred = end_pred.detach().cpu().clone().numpy()
            else:
                y_pred = predictions.detach().cpu().clone().numpy()
            masks = masks.detach().cpu().clone().numpy()

        if self.decode_type == "general":
            return self.get_general_labels(y_pred, masks)
        elif self.decode_type == "biaffine":
            return self.get_biaffine_labels(y_pred, masks)
        elif self.decode_type == "span":
            return self.get_span_labels(start_pred, end_pred)
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
            for idx, m in enumerate(mask):
                if idx == 0:
                    continue
                if m == 1:
                    count += 1
                offset_dict[idx] = count
            
            max_len = len(mask)
            for i in range(1, max_len):
                for j in range(i, max_len):
                    if mask[i] == 0 or mask[j] == 0:
                        continue
                    pred_scores = pred[i][j]
                    pred_label_id = np.argmax(pred_scores)
                    start_idx = offset_dict[i]
                    end_idx = offset_dict[j+1]
                    if pred_label_id > 0:
                        pred_entities.append([start_idx, end_idx, self.label_list[pred_label_id], pred_scores[pred_label_id]])

            pred_entities = sorted(pred_entities, reverse=True, key=lambda x:x[3])
            new_pred_entities = list()
            for pred_entity in pred_entities:
                start, end, tag, _ = pred_entity
                flag = True
                for new_pred_entity in new_pred_entities:
                    new_start, new_end, _ = new_pred_entity
                    if start < new_start < end < new_end or new_start < start < new_end < end:
                        #for flat ner nested mentions are not allowed
                        flag = False
                        break
                if flag:
                    new_pred_entities.append([start, end, tag])
            preds.append(new_pred_entities)
        return preds
    
    def get_span_labels(self, start_id_lists, end_id_lists, is_flat):
        preds = []
        for start_ids, end_ids in zip(start_id_lists, end_id_lists):
            start_ids = start_ids[1:-1]
            end_ids = end_ids[1:-1]
            i = 0
            len_ids = len(start_ids)
            while i < len_ids:
                if start_ids[i] != 0:
                    for j in range(i, len_ids):
                        if start_ids[i] == end_ids[j]:
                            tag = self.label_list[start_ids[i]]
                            preds.append(i, j + 1, tag)
                            if is_flat:
                                i = j
                            break
                i += 1
        return preds

    
    def predict(self, text):
        inputs, masks, offset_list = self.preprocess(text)
        try:
            with torch.no_grad():
                inputs["is_fusion"] = True
                outputs = None
                for i in range(self.k):
                    output = self.models[i](**inputs).cpu()
                    if not outputs:
                        outputs = torch.sigmoid(output)
                    else:
                        outputs += torch.sigmoid(output)
                    torch.cuda.empty_cache()
            # todo 删除改行
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
        except Exception as e:
            print(text)
            print(inputs)
            print(str(e))
            exit(1)
        entity_lists = self.postprocess([text], outputs, masks)
        if self.split:
            entity_list = self.recover(entity_lists, offset_list)
        else:
            entity_list = entity_lists[0]
        # entity_list = self.merge_entity_list(entity_list)
        return entity_list
    
    def preprocess(self, text):
        if self.split:
            text_list, _, offset_list = self.cut(text, None, max_len=200, overlap_len=0)
        else:
            text_list = [text]
            offset_list = list()
        input_ids, input_masks, input_len, masks = self._to_features(text_list, self.tokenizer, self.max_seq_length)
        inputs = {"input_ids": input_ids, "attention_mask": input_masks, "input_len": input_len}
        return inputs, masks, offset_list

    def postprocess(self, texts, outputs, masks):
        entity_lists = list()
        pred_lists = self.get_labels(outputs, masks)
        for text, pred_list in zip(texts, pred_lists):
            entity_list = self.get_entity_list(text, pred_list)
            entity_lists.append(entity_list)
        return entity_lists
    
    def recover(self, entity_lists, offset_list):
        new_entity_list = list()
        for entity_list, offset in zip(entity_lists, offset_list):
            for entity in entity_list:
                start, end, tag, word = entity
                new_entity_list.append([start+offset, end+offset, tag, word])
        return new_entity_list

    def cut(self, text, entities=None, max_len=256, overlap_len=50):
        tags = self.get_tags(len(text), entities)
        sents = self.get_sents(text, tags)
        # sents = re.split(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', text)
        sents_len = len(sents)
        offset_list = list()
        i = 0
        end_idx = 0
        sent_list = list()
        entity_lists = list()
        while i < sents_len:
            sent = sents[i]
            sent_len = len(sent)
            end_idx += sent_len
            if not sent or re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', sent):
                i += 1
                continue
            # 搜索前缀
            pre_list = list()
            pre_list_len = 0
            j = i - 1
            while j >= 0:
                sent_j = sents[j]
                sent_j_len = len(sent_j)
                if pre_list_len + sent_j_len < overlap_len:
                    pre_list.append(sent_j)
                    pre_list_len += sent_j_len
                else:
                    break
                j -= 1
            pre_idx = 0
            pre_list = pre_list[::-1]
            for tmp_sent in pre_list:
                if not tmp_sent or re.match(r'([。？?，,；;！!]|(?<!\d)\.(?!\d))', tmp_sent):
                    pre_idx += 1
                    pre_list_len -= len(tmp_sent)
                else:
                    break
            pre_list = pre_list[pre_idx:]

            # 搜索后缀
            post_list = list()
            post_list_len = 0
            j = i + 1
            while j < sents_len:
                sent_j = sents[j]
                sent_j_len = len(sent_j)
                if pre_list_len + sent_len + post_list_len + sent_j_len < max_len:
                    post_list.append(sent_j)
                    post_list_len += sent_j_len
                else:
                    break
                j += 1
            
            # 拼接
            sent = "".join(pre_list + [sent] + post_list)
            sent_list.append(sent)
            end_idx += post_list_len
            start_idx = end_idx - len(sent)
            offset_list.append(start_idx)
            if entities:
                entity_list = self.get_entities(entities, start_idx, end_idx)
                entity_lists.append(entity_list)
            i = j

        if self.valid(text, sent_list, offset_list):
            return sent_list, entity_lists, offset_list
        else:
            raise ValueError

    def get_tags(self, text_len, entities):
        tags = [True] * text_len
        if entities:
            for entity in entities:
                start, end, _, _ = entity
                for i in range(start, end):
                    tags[i] = False
        return tags

    def get_sents(self, text, tags):
        sents = list()
        start = 0
        text_len = len(text)
        for idx, (word, tag) in enumerate(zip(text, tags)):
            if not tag:
                continue
            if re.search(r"[。？?；;！!]", word):  # catnlp 去掉逗号
                sent = text[start: idx]
                if sent:
                    sents.append(sent)
                sents.append(word)
                start = idx + 1
            elif word == ".":
                if idx > 1 and re.search(r"\d", text[idx-1]):
                    continue
                if idx < text_len - 1 and re.search(r"\d", text[idx+1]):
                    continue
                sent = text[start: idx]
                if sent:
                    sents.append(sent)
                sents.append(word)
                start = idx + 1
        if start < text_len:
            sents.append(text[start: text_len])
        return sents

    def get_entities(self, entities, start, end):
        entity_list = list()
        for entity in entities:
            s, e, t, w = entity
            if start <= s and e <= end and s <= e:
                entity_list.append([s-start, e-start, t, w])
        return entity_list

    def valid(self, text, sent_list, offset_list):
        for offset, sent in zip(offset_list, sent_list):
            sent_len = len(sent)
            start = offset
            end = offset + sent_len
            if text[start: end] != sent:
                print("---")
                print(text[start: end])
                print(sent)
                return False
        return True
    
    def get_entity_list(self, text, entities):
        entity_list = list()
        for entity in entities:
            start, end, tag = entity
            word = text[start: end]
            entity_list.append([start, end, tag, word])
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
            if self.do_lower:
                word = word.lower()
            if re.match(r"\s", word):
                word = "[unused1]"
            tmp_tokens = self.tokenizer.tokenize(word)
            if len(tmp_tokens) == 0:
                tmp_tokens = ['[UNK]']
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
