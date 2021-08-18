# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .nezha.modelling_nezha import (
#     BertModel,
#     BertPreTrainedModel
# )
from transformers import (
    BertModel,
    BertPreTrainedModel
)
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, \
    pad_packed_sequence
from ...layer.decoder.crf import CRF
from ..loss.dice_loss import DiceLoss
from ..loss.focal_loss import FocalLoss
from ..loss.label_loss import LabelSmoothingCrossEntropy


class BertSoftmax(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_func = CrossEntropyLoss()
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            output = self.loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            output = logits.argmax(dim=-1)

        return output


class BertCrf(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask = attention_mask.byte()
        dim2 = torch.max(input_len)

        if labels is not None:
            # output = -self.crf(emissions=logits, tags=labels, mask=attention_mask)
            output = -self.crf(emissions=logits[:, :dim2, :], tags=labels[:, :dim2], mask=attention_mask[:, :dim2])
        else:
            # output = self.crf.decode(emissions=logits, mask=attention_mask)
            output = self.crf.decode(emissions=logits[:, :dim2, :], mask=attention_mask[:, :dim2])
            output = pad_sequence([torch.tensor(o) for o in output], batch_first=True, padding_value=0)

        return output


class BertLstmCrf(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config) #, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size// 2,
                            batch_first=True,
                            bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.lstm(sequence_output)
        logits = self.classifier(lstm_output)
        attention_mask = attention_mask.byte()

        if labels is not None:
            output = -self.crf(emissions=logits, tags=labels, mask=attention_mask)
        else:
            output = self.crf.decode(emissions=logits, mask=attention_mask)
            output = pad_sequence([torch.tensor(o) for o in output], batch_first=True, padding_value=0)

        return output


class BertLstmCrf1(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size// 2,
                            batch_first=True,
                            bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        input_len=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        packed_output = pack_padded_sequence(sequence_output,
                                            input_len.cpu(),
                                            batch_first=True,
                                            enforce_sorted=False)
        lstm_output, _ = self.lstm(packed_output)
        lstm_output, _ = pad_packed_sequence(lstm_output)
        lstm_output = lstm_output.transpose(1, 0)
        logits = self.classifier(lstm_output)
        attention_mask = attention_mask.byte()
        dim2 = logits.size()[1]

        if labels is not None:
            output = -self.crf(emissions=logits, tags=labels[:, :dim2], mask=attention_mask[:, :dim2])
        else:
            output = self.crf.decode(emissions=logits, mask=attention_mask[:, :dim2])
            output = pad_sequence([torch.tensor(o) for o in output], batch_first=True, padding_value=0)

        return output


class BertSpan(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config) #, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size// 2,
                            batch_first=True,
                            bidirectional=True)
        self.start_fc = nn.Linear(config.hidden_size, self.num_labels)
        self.end_fc = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_name = config.loss_name
        if self.loss_name =='lsr':
            self.loss_func = LabelSmoothingCrossEntropy(reduction="none")
        elif self.loss_name == 'focal':
            self.loss_func = FocalLoss(reduction="none")
        else:
            self.loss_func = CrossEntropyLoss(reduction="none")
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.lstm(sequence_output)
        start_logits = self.start_fc(lstm_output)
        end_logits = self.end_fc(lstm_output)
        attention_mask = attention_mask.byte()

        if start_labels is not None and end_labels is not None:
            mask = attention_mask.view(size=(-1,))
            start_labels = start_labels.view(size=(-1,))
            start_logits = start_logits.view(size=(-1, self.num_labels))
            start_loss = self.loss_func(input=start_logits, target=start_labels)
            start_loss *= mask
            start_output = start_loss.sum() / mask.sum()
            end_labels = end_labels.view(size=(-1,))
            end_logits = end_logits.view(size=(-1, self.num_labels))
            end_loss = self.loss_func(input=end_logits, target=end_labels)
            end_loss *= mask
            end_output = end_loss.sum() / mask.sum()
            output = start_output + end_output
        else:
            start_output = nn.functional.softmax(start_logits, dim=-1)
            end_output = nn.functional.softmax(end_logits, dim=-1)
            start_output = start_output.argmax(dim=-1)
            end_output = end_output.argmax(dim=-1)
            output = (start_output, end_output)
        return output


class BertBiaffine(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config) #, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_size = config.hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size// 2,
                            batch_first=True,
                            bidirectional=True,)
                            # dropout=0.2)
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.biaffne_layer = biaffine(200, config.num_labels)
        loss_name = config.loss_name
        if loss_name == "dice":
            self.loss_func = DiceLoss(reduction="none")
        elif loss_name == "focal":
            self.loss_func = FocalLoss(reduction="none")
        else:
            self.loss_func = CrossEntropyLoss(reduction="none")
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # todo
        sequence_output = sequence_output
        
        sequence_output, _ = self.lstm(sequence_output)

        start_logits = self.start_layer(sequence_output) 
        end_logits = self.end_layer(sequence_output) 

        span_logits = self.biaffne_layer(start_logits, end_logits)
        span_logits = span_logits.contiguous()

        if labels is not None:
            labels = labels.view(size=(-1,))
            span_logits = span_logits.view(size=(-1, self.num_labels))
            span_loss = self.loss_func(input=span_logits, target=labels)
            label_mask = label_mask.view(size=(-1,))
            span_loss *= label_mask
            output = span_loss.sum() / label_mask.sum()
        else:
            output = nn.functional.softmax(span_logits, dim=-1)
            # output = torch.argmax(output, dim=-1)

        return output


class BertMultiBiaffine(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_segs = config.num_segs

        self.bert = BertModel(config) #, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_size = config.hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size// 2,
                            batch_first=True,
                            bidirectional=True,)
                            # dropout=0.2)
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.biaffine_layer = biaffine(200, config.num_labels)
        self.seg_biaffine_layer = biaffine(200, config.num_segs)
        loss_name = config.loss_name
        if loss_name == "dice":
            self.loss_func = DiceLoss(reduction="none")
        elif loss_name == "focal":
            self.loss_func = FocalLoss(reduction="none")
        else:
            self.loss_func = CrossEntropyLoss(reduction="none")
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # todo
        sequence_output = sequence_output
        
        sequence_output, _ = self.lstm(sequence_output)

        start_logits = self.start_layer(sequence_output) 
        end_logits = self.end_layer(sequence_output) 

        span_logits = self.biaffine_layer(start_logits, end_logits)
        span_logits = span_logits.contiguous()

        if labels is not None:
            labels = labels.view(size=(-1,))
            span_logits = span_logits.view(size=(-1, self.num_labels))
            span_loss = self.loss_func(input=span_logits, target=labels)
            label_mask = label_mask.view(size=(-1,))
            span_loss *= label_mask
            output = span_loss.sum() / label_mask.sum()
            if segs is not None:
                segs_span_logits = self.seg_biaffine_layer(start_logits, end_logits)
                segs_span_logits = segs_span_logits.contiguous()
                segs = segs.view(size=(-1,))
                segs_span_logits = segs_span_logits.view(size=(-1, self.num_segs))
                segs_span_loss = self.loss_func(input=segs_span_logits, target=segs)
                segs_span_loss *= label_mask
                output += 0.1 * segs_span_loss.sum() / label_mask.sum()
        else:
            output = nn.functional.softmax(span_logits, dim=-1)
            # output = torch.argmax(output, dim=-1)

        return output


class BertMultiHiddenBiaffine(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_segs = config.num_segs

        self.bert = BertModel(config) #, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_size = config.hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size// 2,
                            batch_first=True,
                            bidirectional=True,)
                            # dropout=0.2)
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.seg_start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.seg_end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.biaffine_layer = biaffine(200, config.num_labels)
        self.seg_biaffine_layer = biaffine(200, config.num_segs)
        loss_name = config.loss_name
        if loss_name == "dice":
            self.loss_func = DiceLoss(reduction="none")
        elif loss_name == "focal":
            self.loss_func = FocalLoss(reduction="none")
        else:
            self.loss_func = CrossEntropyLoss(reduction="none")
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # todo
        sequence_output = sequence_output
        
        sequence_output, _ = self.lstm(sequence_output)

        start_logits = self.start_layer(sequence_output) 
        end_logits = self.end_layer(sequence_output) 

        span_logits = self.biaffine_layer(start_logits, end_logits)
        span_logits = span_logits.contiguous()

        if labels is not None:
            labels = labels.view(size=(-1,))
            span_logits = span_logits.view(size=(-1, self.num_labels))
            span_loss = self.loss_func(input=span_logits, target=labels)
            label_mask = label_mask.view(size=(-1,))
            span_loss *= label_mask
            output = span_loss.sum() / label_mask.sum()
            if segs is not None:
                seg_start_logits = self.seg_start_layer(sequence_output) 
                seg_end_logits = self.seg_end_layer(sequence_output) 
                seg_span_logits = self.seg_biaffine_layer(seg_start_logits, seg_end_logits)
                seg_span_logits = seg_span_logits.contiguous()
                segs = segs.view(size=(-1,))
                seg_span_logits = seg_span_logits.view(size=(-1, self.num_segs))
                seg_span_loss = self.loss_func(input=seg_span_logits, target=segs)
                seg_span_loss *= label_mask
                output += seg_span_loss.sum() / label_mask.sum()
        else:
            output = nn.functional.softmax(span_logits, dim=-1)
            # output = torch.argmax(output, dim=-1)

        return output


class BertMultiAddBiaffine(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_segs = config.num_segs

        self.bert = BertModel(config) #, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_size = config.hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size// 2,
                            batch_first=True,
                            bidirectional=True,)
                            # dropout=0.2)
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_size, out_features=200),
                                            torch.nn.ReLU())
        self.biaffine_layer = biaffine(200, config.num_labels)
        self.seg_biaffine_layer = biaffine(200, config.num_segs)
        loss_name = config.loss_name
        if loss_name == "dice":
            self.loss_func = DiceLoss(reduction="none")
        elif loss_name == "focal":
            self.loss_func = FocalLoss(reduction="none")
        else:
            self.loss_func = CrossEntropyLoss(reduction="none")
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
        input_len=None,
        segs=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # todo
        sequence_output = sequence_output
        
        sequence_output, _ = self.lstm(sequence_output)

        start_logits = self.start_layer(sequence_output) 
        end_logits = self.end_layer(sequence_output) 

        segs_span_logits = self.seg_biaffine_layer(start_logits, end_logits)
        segs_span_logits = segs_span_logits.contiguous()

        span_logits = self.biaffine_layer(start_logits, end_logits)
        span_logits = span_logits.contiguous()
        # print(segs_span_logits.size())
        add_logits = torch.max(segs_span_logits.detach(), dim=-1, keepdim=True)
        # print(add_logits.size())
        span_logits += add_logits

        if labels is not None:
            labels = labels.view(size=(-1,))
            span_logits = span_logits.view(size=(-1, self.num_labels))
            span_loss = self.loss_func(input=span_logits, target=labels)
            label_mask = label_mask.view(size=(-1,))
            span_loss *= label_mask
            output = span_loss.sum() / label_mask.sum()
            if segs is not None:
                segs = segs.view(size=(-1,))
                segs_span_logits = segs_span_logits.view(size=(-1, self.num_segs))
                segs_span_loss = self.loss_func(input=segs_span_logits, target=segs)
                segs_span_loss *= label_mask
                output += segs_span_loss.sum() / label_mask.sum()
        else:
            output = nn.functional.softmax(span_logits, dim=-1)
            # output = torch.argmax(output, dim=-1)

        return output


class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping
