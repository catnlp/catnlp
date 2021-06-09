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
import logging
from pathlib import Path

import torch
from transformers import (
    AutoConfig
)
import numpy as np

from ..common.load_file import load_label_file
from .model.albert_tiny import AlbertTinyCrf, AlbertTinySoftmax
from .model.bert import BertBiaffine,BertCrf, BertSoftmax
from .util.tokenizer import NerBertTokenizer
from .util.split import merge_entities


logger = logging.getLogger(__name__)


class PredictPlm:
    def __init__(self, config) -> None:
        self.max_seq_length = config.get("max_length")
        label_file = Path(config.get("model_path")) / "label.txt"
        self.label_list = load_label_file(label_file)
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_list)}
        print(self.label_to_id)
        vocab_file = Path(config.get("model_path")) / "vocab.txt"
        self.tokenizer = NerBertTokenizer(vocab_file, do_lower_case=config.get("do_lower_case"))
        pretrained_config = AutoConfig.from_pretrained(config.get("model_path"), num_labels=len(self.label_list))

        model_func = None
        model_name = config.get("name").lower()
        if model_name == "bert_crf":
            model_func = BertCrf
        elif model_name == "bert_softmax":
            model_func = BertSoftmax
        elif model_name == "bert_biaffine":
            model_func = BertBiaffine
        elif model_name == "albert_tiny_crf":
            model_func = AlbertTinyCrf
        elif model_name == "albert_tiny_softmax":
            model_func = AlbertTinySoftmax
        else:
            raise ValueError

        self.model = model_func.from_pretrained(
            config.get("model_path"),
            config=pretrained_config,
            label_size=len(self.label_list)
        )
        self.device = config.get("device")
        self.model.to(torch.device(self.device))
        self.model.eval()
        self.decode_type = config.get("decode_type")
    
    def get_labels(self, texts, predictions):
        # Transform predictions and references tensos to numpy arrays
        if self.device == "cpu":
            y_pred = predictions.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
        if self.decode_type == "general":
            return self.get_general_labels(y_pred)
        elif self.decode_type == "biaffine":
            return self.get_biaffine_labels(texts, y_pred)
        else:
            raise ValueError
    
    def get_biaffine_labels(self, texts, y_pred):
        preds = list()
        for text, pred in zip(texts, y_pred):
            pred_entities = list()
            text_len = min(self.max_seq_length - 1, len(text) + 1)
            for i in range(1, text_len):
                for j in range(i, text_len):
                    pred_scores = pred[i][j]
                    pred_label_id = np.argmax(pred_scores)
                    if pred_label_id > 0:
                        pred_entities.append([i-1, j, self.label_list[pred_label_id], pred_scores[pred_label_id]])

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

            tmp_preds = ["O"] * text_len
            for entity in new_pred_entities:
                start, end, tag = entity
                tmp_preds[start] = f"B-{tag}"
                for i in range(start+1, end):
                    tmp_preds[i] = f"I-{tag}"
            preds.append(tmp_preds)
        return preds

    def get_general_labels(self, y_pred):
        # Remove ignored index (special tokens)
        preds = [
            [self.label_list[p] for p in pred[1:] if p > 0]
            for pred in y_pred
        ]
        return preds
    
    def predict(self, text):
        inputs = self.preprocess([text])
        outputs = self.model(**inputs)
        entity_lists = self.postprocess([text], outputs)
        return entity_lists[0]
    
    def preprocess(self, text_list):
        input_ids, input_masks = self._to_features(text_list, self.tokenizer, self.max_seq_length)
        inputs = {"input_ids": input_ids, "attention_mask": input_masks}
        return inputs

    def postprocess(self, texts, outputs):
        entity_lists = list()
        pred_lists = self.get_labels(texts, outputs)
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
    
    def _to_features(self, texts, tokenizer=None, max_seq_length=-1,
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
        for (ex_index, text) in enumerate(texts):
            tokens = tokenizer.tokenize(text)
            # Account for [CLS] and [SEP] with "- 2".
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]

            pad_id = self.label_to_id.get(pad_token)
            tokens += [sep_token]

            if cls_token_at_end:
                tokens += [cls_token]
            else:
                tokens = [cls_token] + tokens

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_masks = [1 if mask_padding_with_zero else 0] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_id] * padding_length) + input_ids
                input_masks = ([0 if mask_padding_with_zero else 1] * padding_length) + input_masks
            else:
                input_ids += [pad_id] * padding_length
                input_masks += [0 if mask_padding_with_zero else 1] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_masks) == max_seq_length
            input_id_lists.append(input_ids)
            input_mask_lists.append(input_masks)
        input_ids = torch.tensor(input_id_lists, dtype=torch.long).to(torch.device(self.device))
        input_masks = torch.tensor(input_mask_lists, dtype=torch.long).to(torch.device(self.device))
        return input_ids, input_masks
