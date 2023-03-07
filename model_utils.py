#-*- coding:utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)
def span_decode(start_logits, end_logits, max_sequence):
    start_logits = start_logits.clone().detach().cpu().numpy()
    end_logits = end_logits.clone().detach().cpu().numpy()
    predict_labels = np.array([0] * max_sequence)
    start_pred = np.argmax(start_logits, -1)[0]
    end_pred = np.argmax(end_logits, -1)[0]
    # print("start_pred", start_pred)
    # print("end_pred", end_pred)
    
    for i, s_type in enumerate(start_pred):
        if predict_labels[i] !=0:
            continue
        if s_type == 0:
            predict_labels[i] = 0
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                predict_labels[i:i + j + 1] = s_type
                break
    # print ("predict_labels", predict_labels.tolist())
    return [predict_labels.tolist()]
class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
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
class SpanModel(BaseModel):
    def __init__(self,
                 num_tags,
                 dropout_prob=0.1,
                 loss_type='ls_ce',
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(SpanModel, self).__init__()
        out_dims = 1536
        mid_linear_dims = kwargs.pop('mid_linear_dims', 256)
        self.num_tags = num_tags
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(mid_linear_dims)
        )
        out_dims = mid_linear_dims
        self.start_fc = nn.Linear(out_dims, num_tags)
        self.end_fc = nn.Linear(out_dims, num_tags)
        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction)
        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)
        init_blocks = [self.mid_linear, self.start_fc, self.end_fc]
        self._init_weights(init_blocks)
    def forward(self,
                seq_out,
                attention_masks,
                start_ids=None,
                end_ids=None,
                pseudo=None):
        # bert_outputs = self.bert_module(
        #     input_ids=token_ids,
        #     attention_mask=attention_masks,
        #     token_type_ids=token_type_ids
        # )
        # seq_out = bert_outputs[0]
        # batch = torch.Size([1, 186, 1536]) mask = torch.Size([1, 186]) labels = torch.Size([1, 186])  
        batch, sentence_len, dim = seq_out.shape
        seq_out = self.mid_linear(seq_out)
        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)
        # out = (start_logits, end_logits, )
        outputs = {}
        # outputs["start_logits"] = start_logits
        # outputs["end_logits"] = end_logits
        if start_ids is not None and end_ids is not None:
            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)
            # 去掉 padding 部分的标签，计算真实 loss
            active_loss = attention_masks.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]
            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]
            if pseudo is not None:
                # (batch,)
                start_loss = self.criterion(start_logits, start_ids.view(-1)).view(-1, 512).mean(dim=-1)
                end_loss = self.criterion(end_logits, end_ids.view(-1)).view(-1, 512).mean(dim=-1)
                # nums of pseudo data
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]
                # learning parameter
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    start_loss = start_loss.mean()
                    end_loss = end_loss.mean()
                else:
                    if total_nums == pseudo_nums:
                        start_loss = (rate*pseudo*start_loss).sum() / pseudo_nums
                        end_loss = (rate*pseudo*end_loss).sum() / pseudo_nums
                    else:
                        start_loss = (rate*pseudo*start_loss).sum() / pseudo_nums \
                                     + ((1 - rate) * (1 - pseudo) * start_loss).sum() / (total_nums - pseudo_nums)
                        end_loss = (rate*pseudo*end_loss).sum() / pseudo_nums \
                                     + ((1 - rate) * (1 - pseudo) * end_loss).sum() / (total_nums - pseudo_nums)
            else:
                start_loss = self.criterion(active_start_logits, active_start_labels)
                end_loss = self.criterion(active_end_logits, active_end_labels)
            loss = start_loss + end_loss
            # out = (loss, ) + out
            outputs["loss"] = loss.sum()
        else:
            predict_labels = span_decode(start_logits, end_logits, sentence_len)
            outputs["predict_labels"] = predict_labels
        return outputs
class SinusoidalPositionEmbedding(torch.nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids
    def forward(self, inputs):
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)
def relative_position_encoding(depth, max_length=512, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(max_length)
    range_mat = range_vec.repeat(max_length).view(max_length, max_length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    embeddings_table = torch.zeros(vocab_size, depth)
    position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)
    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    positions_encoding = positions_encoding.view(my_shape)
    return positions_encoding
def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        result = x * mask + value * (1 - mask)
        result = torch.where(torch.isnan(result), torch.full_like(result, float('-inf')), result)
        return result
def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = (torch.tril(torch.ones_like(logits, dtype=torch.bool), diagonal=-1) | (~torch.tril(torch.ones_like(logits, dtype=torch.bool), diagonal=40)))
    logits = logits - mask * 1e12
    return logits
class GlobalPointer(torch.nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, heads, head_size,hidden_size,RoPE=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.dense = nn.Linear(hidden_size,self.head_size * self.heads * 2)
    def forward(self, inputs, mask=None):
        inputs = self.dense(inputs)
        inputs = torch.split(inputs, self.head_size * 2 , dim=-1)
        # 按照-1这个维度去分，每块包含x个小块
        inputs = torch.stack(inputs, dim=-2)
        #沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        #分出qw和kw
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., None, 1::2].repeat(1,1,1,2)
            sin_pos = pos[..., None, ::2].repeat(1,1,1,2)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # 计算内积
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding 排除下三角
        logits = add_mask_tril(logits,mask)
        # scale返回
        return logits / self.head_size ** 0.5
def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss
def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    #y_pred = (batch,l,l,c)
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))
class FocalLoss(torch.nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
class TxMutihead(torch.nn.Module):
    def __init__(self,hidden_size,c_size,abPosition = False,rePosition=False, maxlen=None,max_relative=None):
        super(TxMutihead, self).__init__()
        self.hidden_size = hidden_size
        self.c_size = c_size
        self.abPosition = abPosition
        self.rePosition = rePosition
        self.Wh = nn.Linear(hidden_size * 4, self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size,self.c_size)
        if self.rePosition:
            self.relative_positions_encoding = relative_position_encoding(max_length=maxlen,
                                                                     depth=4*hidden_size,
                                                                     max_relative_position=max_relative)
        init_block = [self.Wh, self.Wo]
        self._init_weights(init_block)
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
    def forward(self, inputs, mask=None):
        input_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        # print ("111", inputs.shape, mask.shape)
        if self.abPosition:
            # 由于为加性拼接，我们无法使用RoPE,因此这里直接使用绝对位置编码
            inputs = SinusoidalPositionEmbedding(self.hidden_size, 'add')(inputs)
        x1 = torch.unsqueeze(inputs, 1)
        x2 = torch.unsqueeze(inputs, 2)
        x1 = x1.repeat(1, input_length, 1, 1)
        x2 = x2.repeat(1, 1, input_length, 1)
        concat_x = torch.cat([x2, x1,x2-x1,x2.mul(x1)], dim=-1)
        if self.rePosition:
            relations_keys = self.relative_positions_encoding[:input_length, :input_length, :].to(inputs.device)
            concat_x += relations_keys
        hij = torch.tanh(self.Wh(concat_x))
        logits = self.Wo(hij)
        logits = logits.permute(0,3,1,2)
        logits = add_mask_tril(logits, mask)
        return logits
class PointerModel(BaseModel):
    def __init__(self,
                 num_tags,
                 dropout_prob=0.1,
                 loss_type='global_ce',
                 **kwargs):
        """
        tag the subject and object corresponding to the predicate
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        """
        super(PointerModel, self).__init__()
        out_dims = 1536
        mid_linear_dims = 256
        self.num_tags = num_tags
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(mid_linear_dims)
        )
        out_dims = mid_linear_dims
        # self.head = GlobalPointer(num_tags, 64, out_dims)
        self.head = TxMutihead(out_dims, num_tags, maxlen=1733, rePosition=True, max_relative=64)
        reduction = 'none'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        elif loss_type == 'global_ce':
            self.criterion = global_pointer_crossentropy
        else:
            self.criterion = FocalLoss(reduction=reduction)
        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)
        init_blocks = [self.mid_linear]
        self._init_weights(init_blocks)
    
    def post_process_label(self, labels, shape):
        # shape bsz * type_num * seq_len * seq_len
        # seq_len  bsz * type_num * seq_len
        
        bsz, ent_num, seq_len, _ = shape
        b = torch.zeros(bsz, ent_num, seq_len)
        for b_no in range(bsz):
            label = labels[b_no,:]
            for idx, val in enumerate(label):
                b[b_no, val, idx] = 1 
        
        labels = b.to(labels.device)
        labels = labels.long()
    
        # labels = torch.LongTensor(bsz, ent_num, seq_len).to(labels.device)
        # ones = torch.ones_like(labels)
        # labels = torch.where(labels>0, ones, labels)
        update_labels = torch.zeros((bsz, ent_num, seq_len*seq_len),dtype=torch.long).to(labels.device)
        update_labels = torch.scatter(update_labels, 2, labels, 1)
        update_labels[:,:,0] = 0
        update_labels = update_labels.view(bsz, ent_num, seq_len, seq_len).contiguous()
        return update_labels
    def forward(self,
                seq_out,
                attention_masks,
                labels=None,
                pseudo=None):
        # bert_outputs = self.bert_module(
        #     input_ids=token_ids,
        #     attention_mask=attention_masks,
        #     token_type_ids=token_type_ids
        # )
        # batch, sentence, dim
        # batch = torch.Size([1, 186, 1536]) mask = torch.Size([1, 186]) labels = torch.Size([1, 186])
        # seq_out = bert_outputs[0]
        seq_out = self.mid_linear(seq_out)
        logits = self.head(seq_out, mask=attention_masks)
        outputs = {}
        if labels is not None and self.training:
            labels = self.post_process_label(labels, logits.shape)
            loss = self.criterion(labels, logits) 
            outputs["loss"] = loss
        else:
            outputs["logits"] = logits  # batch, num_tag, seq_len * seq_len 
        return outputs