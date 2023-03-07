import os
import math
from numpy.core.numerictypes import maximum_sctype
import torch
import torch.nn as nn
from torchcrf import CRF
from itertools import repeat
from transformers import BertModel
import torch.nn.functional as F 



class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob):
        super(BaseModel, self).__init__()

        self.bert_module = BertModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)
        self.bert_config = self.bert_module.config


    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


# baseline
class CRFModel(BaseModel):
    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 **kwargs):
        super(CRFModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)

        out_dims = 768

        mid_linear_dims = kwargs.pop('mid_linear_dims', 256)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.classifier = nn.Linear(out_dims, num_tags)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)

        self.crf_module = CRF(num_tags=num_tags, batch_first=True)

        init_blocks = [self.mid_linear, self.classifier]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                input_ids,
                input_mask,
                token_type_ids,
                labels=None):

        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]

        seq_out = self.mid_linear(seq_out)

        emissions = self.classifier(seq_out)

        if labels is not None:
            tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                tags=labels.long(),
                                                mask=input_mask.byte(),
                                                reduction='mean')
            out = (tokens_loss,)
        else:
            tokens_out = self.crf_module.decode(emissions=emissions, mask=input_mask.byte())
            out = (tokens_out, emissions)
        return out

def crf_decode(decode_tokens, token_segs, label2name):
    """
    CRF 解码，用于解码 time loc 的提取
    """
    predict_entities = {}

    decode_tokens = decode_tokens[1:-1]  # 除去 CLS SEP token

    index_ = 0

    while index_ < len(decode_tokens):

        token_label = label2name[decode_tokens[index_]].split('-')

        if token_label[0].startswith('S'):
            token_type = token_label[1]
            tmp_ent = token_segs[index_]
            # ent_start, ent_end = offset_mapping[index_+1][0], offset_mapping[index_+1][1]
            # tmp_ent = raw_text[ent_start:ent_end]
            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent, int(index_), int(index_)+1)]
            else:
                predict_entities[token_type].append((tmp_ent, int(index_), int(index_)+1))

            index_ += 1

        elif token_label[0].startswith('B'):
            token_type = token_label[1]
            start_index = index_ 

            index_ += 1
            while index_ < len(decode_tokens):
                
                temp_token_label = label2name[decode_tokens[index_]].split('-')

                if temp_token_label[0].startswith('I') and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith('E') and token_type == temp_token_label[1]:
                    end_index = index_ + 1
                    index_ += 1

                    tmp_ent = token_segs[start_index: end_index]

                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, int(start_index), int(end_index))]
                    else:
                        predict_entities[token_type].append((tmp_ent, int(start_index), int(end_index)))

                    break
                else:
                    break
        else:
            index_ += 1

    return predict_entities