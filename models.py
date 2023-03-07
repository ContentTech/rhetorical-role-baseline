from allennlp.common.util import pad_sequence_to_length
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import masked_mean, masked_softmax
import copy

from transformers import BertModel

from transformers import BertTokenizer
from bert_model import BertModel as BertModelC
from bert_model import SelfAttention, BartAttention
from torch.nn import CrossEntropyLoss
from allennlp.modules import ConditionalRandomField

import torch
import math
import torch.nn.functional as F
from task import LEGALEVAL_LABELS, LEGALEVAL_LABELS_PRES, LEGALEVAL_LABELS_DETAILS
import random
from torch import Tensor
from typing import Optional
from model_utils import PointerModel,SpanModel
from utils import get_device, ResultWriter, log




class DiscountedTypeLoss(torch.nn.Module):
    def __init__(self, num_tags, hidden_dim, device):
        super(DiscountedTypeLoss, self).__init__()
        self.prototype_embed = torch.nn.Embedding(num_tags, num_tags).to(device)
        self.num_tags = num_tags
        self.lnn = torch.nn.Linear(hidden_dim, num_tags)


    def forward(self, features, labels, temperature=0.3):
        """
        Args:
        f_output: [batch_size, max_char_len, num_type] 全图编码向量
        prototype_embed: [num_type, num_class] 原型向量
        probe_type_id: [batch_size, num_type, max_char_len] 标志序列中每个字符与标签的匹配关系的mask矩阵

        Output:
        type_loss / num_type: 1_dim tensor

        """
        batch, seq_len, dim = features.shape
        label_features = labels.unsqueeze(2) # batch, seq, 1
        label_mask = torch.zeros((batch, seq_len, self.num_tags)).to(features.device)
        label_mask.scatter_(2, label_features, 1).transpose(1, 2) # batch, sen, num_tags
        probe_type_id = label_mask.transpose(1, 2) # batch, num_tags, sen

        f_output = self.lnn(features)

        device = f_output.device
        type_loss = torch.zeros([1], device=device)
        num_type = self.num_tags
        aa = torch.LongTensor([i for i in range(num_type)]).to(device)
        prototype_embedding = self.prototype_embed(aa)
        for i in range(num_type):
            mask = probe_type_id[:, i, :].unsqueeze(0).bool()
            mask = mask.transpose(0, 1).transpose(1, 2)
            type_embed = torch.masked_select(f_output, mask)
            type_embed = type_embed.reshape(-1, self.num_tags)
            if type_embed.size()[0] == 0:
                continue
            else:
                type_embed = torch.sum(type_embed, 0) / len(type_embed)
                type_embed = type_embed.repeat(num_type, 1)
                all_pair = (
                    (1 - torch.cosine_similarity(type_embed, prototype_embedding, dim=1))
                    / temperature
                ) * -1
                p_embed = prototype_embedding[i].repeat(num_type, 1)
                sim_pair = torch.cosine_similarity(p_embed, prototype_embedding, dim=1)
                _, index = torch.sort(sim_pair, descending=True)
                _, rank = index.sort()
                discount = torch.log2((rank + 2).float())
                all_pair = all_pair / discount
                all_pair = F.softmax(all_pair, dim=0)
                type_loss += -torch.log(all_pair[i])
        return type_loss / num_type


class LabelContrastiveLoss(torch.nn.Module):
    def __init__(self, num_tags, sen_dim, device, tempture=0.5):
        super(LabelContrastiveLoss, self).__init__()
        self.tempture = tempture
        self.num_tags = num_tags
        self.label_embedding = torch.nn.Embedding(num_tags, sen_dim)
        # label_all = torch.LongTensor([[i for i in range(num_tags)]]).to(device)
        # self.type_embedding = torch.nn.Embedding(num_tags, sen_dim).to(device)
        # self.label_embedding = self.type_embedding(label_all)
        # self.label_embedding = torch.randn(1, num_tags, sen_dim).to(device)  # 1, num_tags, dim
        # self.label_embedding = label_embedding.unsqueeze(0)
        # print ("label_embedding shape ", self.label_embedding.shape)

    def get_att_dis(self, behaviored, target):
        batch_s, seq, dim = behaviored.shape
        batchs = []
        for batch_no in range(batch_s):
            behaviored_ = behaviored[batch_no]
            target_ = target[batch_no]
            attention_distribution = []
            for i in range(seq):
                attention_score = torch.nn.functional.cosine_similarity(target_, behaviored_[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
                attention_distribution.append(attention_score)

            attention_distribution = torch.cat(attention_distribution, dim=0).reshape(len(attention_distribution), -1)
            batchs.append(attention_distribution)
        # batchs = torch.Tensor(batchs)
        batchs = torch.cat(batchs, dim=0).reshape(len(batchs), len(attention_distribution), -1)
        return batchs

    def forward(self, features, labels):
        # features  batch, seq, dim
        # label batch, seq
        batch, seq_len, sen_dim = features.shape
        # label embedding
        # label_all = torch.LongTensor([[i for i in range(num_tags)]])
        # type_embedding = torch.nn.Embedding(num_tags, sen_dim)
        # label_dim = type_embedding(label_all) # batch, num_tags, dim
        # label_dim = self.label_embedding.transpose(1,2) # batch, dim, num_tags
        # batch, sen_len, 1
        label_features = labels.unsqueeze(2) # batch, seq, 1
        # print (label_features)
        # features  batch, seq, dim
        # label_features batch, num_tags, dim
        # label batch, seq

        label_mask = torch.zeros((batch, seq_len, self.num_tags)).to(features.device)
        label_mask.scatter_(2, label_features, 1)# batch, sen, num_tags
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        label_embs = torch.LongTensor([i for i in range(self.num_tags)]).to(features.device)
        prototype_embeddings = []
        for i in range(batch):
            prototype_embedding = self.label_embedding(label_embs).unsqueeze(0) # 1, num_tags, dim
            prototype_embeddings.append(prototype_embedding)
        prototype_embeddings = torch.cat(prototype_embeddings, dim=1).reshape(batch, self.num_tags, -1)

        # label_embedding = torch.nn.functional.normalize(prototype_embedding, p=2, dim=-1)
        # label_embedding = label_embedding.transpose(1,2) # batch, dim, num_tags
        # sen_label_matrix = torch.bmm(features, label_embedding) # batch, sen, num_tags

        sen_label_matrix = self.get_att_dis(features, prototype_embeddings)
        sen_label_pos = torch.exp(torch.div(sen_label_matrix*label_mask, self.tempture))
        sen_label_pos = sen_label_pos * label_mask
        sen_label_pos = torch.sum(sen_label_pos, axis=-1) # batch, seq
        label_mask_neg = torch.ones_like(label_mask).to(features.device)
        label_mask_neg = label_mask_neg - label_mask
        sen_label_neg = torch.exp(torch.div(sen_label_matrix*label_mask_neg, self.tempture))
        sen_label_neg = sen_label_neg * label_mask_neg
        sen_label_neg = torch.sum(sen_label_neg, axis=-1)  # batch, seq
        loss = -1 * torch.log(torch.div(sen_label_pos, sen_label_neg))
        return loss.sum()


class LabelContrastiveLoss(torch.nn.Module):
    def __init__(self, num_tags, sen_dim, device, tempture=0.5):
        super(LabelContrastiveLoss, self).__init__()
        self.tempture = tempture
        self.num_tags = num_tags
        self.label_embedding = torch.nn.Embedding(num_tags, sen_dim)
        # label_all = torch.LongTensor([[i for i in range(num_tags)]]).to(device)
        # self.type_embedding = torch.nn.Embedding(num_tags, sen_dim).to(device)
        # self.label_embedding = self.type_embedding(label_all)
        # self.label_embedding = torch.randn(1, num_tags, sen_dim).to(device)  # 1, num_tags, dim
        # self.label_embedding = label_embedding.unsqueeze(0)
        # print ("label_embedding shape ", self.label_embedding.shape)

    def get_att_dis(self, behaviored, target):
        batch_s, seq, dim = behaviored.shape
        batchs = []
        for batch_no in range(batch_s):
            behaviored_ = behaviored[batch_no]
            target_ = target[batch_no]
            attention_distribution = []
            for i in range(seq):
                attention_score = torch.nn.functional.cosine_similarity(target_, behaviored_[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
                attention_distribution.append(attention_score)

            attention_distribution = torch.cat(attention_distribution, dim=0).reshape(len(attention_distribution), -1)
            batchs.append(attention_distribution)
        # batchs = torch.Tensor(batchs)
        batchs = torch.cat(batchs, dim=0).reshape(len(batchs), len(attention_distribution), -1)
        return batchs

    def forward(self, features, labels):
        # features  batch, seq, dim
        # label batch, seq
        batch, seq_len, sen_dim = features.shape
        # label embedding
        # label_all = torch.LongTensor([[i for i in range(num_tags)]])
        # type_embedding = torch.nn.Embedding(num_tags, sen_dim)
        # label_dim = type_embedding(label_all) # batch, num_tags, dim
        # label_dim = self.label_embedding.transpose(1,2) # batch, dim, num_tags
        # batch, sen_len, 1
        label_features = labels.unsqueeze(2) # batch, seq, 1
        # print (label_features)
        # features  batch, seq, dim
        # label_features batch, num_tags, dim
        # label batch, seq

        label_mask = torch.zeros((batch, seq_len, self.num_tags)).to(features.device)
        label_mask.scatter_(2, label_features, 1)# batch, sen, num_tags
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        label_embs = torch.LongTensor([i for i in range(self.num_tags)]).to(features.device)
        prototype_embeddings = []
        for i in range(batch):
            prototype_embedding = self.label_embedding(label_embs).unsqueeze(0) # 1, num_tags, dim
            prototype_embeddings.append(prototype_embedding)
        prototype_embeddings = torch.cat(prototype_embeddings, dim=1).reshape(batch, self.num_tags, -1)

        # label_embedding = torch.nn.functional.normalize(prototype_embedding, p=2, dim=-1)
        # label_embedding = label_embedding.transpose(1,2) # batch, dim, num_tags
        # sen_label_matrix = torch.bmm(features, label_embedding) # batch, sen, num_tags

        sen_label_matrix = self.get_att_dis(features, prototype_embeddings)
        sen_label_pos = torch.exp(torch.div(sen_label_matrix*label_mask, self.tempture))
        sen_label_pos = sen_label_pos * label_mask
        sen_label_pos = torch.sum(sen_label_pos, axis=-1) # batch, seq
        label_mask_neg = torch.ones_like(label_mask).to(features.device)
        label_mask_neg = label_mask_neg - label_mask
        sen_label_neg = torch.exp(torch.div(sen_label_matrix*label_mask_neg, self.tempture))
        sen_label_neg = sen_label_neg * label_mask_neg
        sen_label_neg = torch.sum(sen_label_neg, axis=-1)  # batch, seq
        loss = -1 * torch.log(torch.div(sen_label_pos, sen_label_neg))
        return loss.sum()


class ContrastiveLoss(torch.nn.Module):
    """
    self-supervised contrastive loss with data augmentation
    """

    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, hidden1, hidden2, labels=None):
        """
        sentence_embeddings: one document in a batch，so here take the number of document sentence as batch size
        """
        hidden1 = torch.nn.functional.normalize(hidden1.squeeze(), p=2, dim=-1)
        hidden2 = torch.nn.functional.normalize(hidden2.squeeze(), p=2, dim=-1)
        batch_size, hidden_dim = hidden1.shape
        hidden1_large = hidden1
        hidden2_large = hidden2
        # labels = labels.squeeze()
        labels = torch.arange(0, batch_size).to(device=hidden1.device)

        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size). \
            to(device=hidden1.device, dtype=torch.float)
        LARGE_NUM = 1e9
        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss


class SupervisedContrastiveLoss(torch.nn.Module):
    """
    supervised contrastive loss with/without data augmentation
    """

    def __init__(self, temperature=0.5):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2=None, labels=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # features1 = features1.squeeze()
        features1 = features1.contiguous().view(-1, features1.shape[-1])
        labels = labels.squeeze()
        batch_size = features1.shape[0]
        features1 = torch.nn.functional.normalize(features1, p=2, dim=1)

        if features2 is not None:
            # features2 = features2.squeeze()
            features2 = features2.contiguous().view(-1, features2.shape[-1])
            features2 = torch.nn.functional.normalize(features2, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            print('features1: ', features1.shape)
            print('labels: ', labels.shape)
            print('batch_size: ', batch_size)
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 两个样本i,j的label相等时，mask_{i,j}=1

        logits = torch.div(torch.matmul(features1, features1.T), self.temperature)
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()

        if features2 is not None:
            logits_ab = torch.div(torch.matmul(features1, features2.T), self.temperature)
            # logits_ab_max, _ = torch.max(logits_ab, dim=1, keepdim=True)
            # logits_ab = logits_ab - logits_ab_max.detach()
            logits = torch.cat([logits_ab, logits], dim=1)

        exp_logits = torch.exp(logits)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        if features2 is not None:
            logits_mask_ab = torch.ones_like(mask)
            positives_mask_ab = mask
            negatives_mask_ab = 1. - mask
            logits_mask = torch.cat([logits_mask_ab, logits_mask], dim=1)
            positives_mask = torch.cat([positives_mask_ab, positives_mask], dim=1)
            negatives_mask = torch.cat([negatives_mask_ab, negatives_mask], dim=1)

        num_positives_per_row = torch.sum(positives_mask, axis=1)
        denominator = torch.sum(exp_logits * logits_mask, axis=1, keepdims=True)
        # denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True)
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / \
                    num_positives_per_row[num_positives_per_row > 0]
        loss = -log_probs
        loss = loss.mean()
        return loss


class DocumentSupervisedContrastiveLoss(torch.nn.Module):
    """
    supervised contrastive loss with/without data augmentation
    """

    def __init__(self, temperature=0.5):
        super(DocumentSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2=None, labels=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features1 = features1.squeeze()
        labels = labels.squeeze()
        batch_size = features1.shape[0]
        features1 = torch.nn.functional.normalize(features1, p=2, dim=1)

        if features2 is not None:
            features2 = features2.squeeze()
            features2 = torch.nn.functional.normalize(features2, p=2, dim=1)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 两个样本i,j的label相等时，mask_{i,j}=1

        anchor_dot_contrast = torch.div(torch.matmul(features1, features1.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if features2 is not None:
            logits_ab = torch.div(torch.matmul(features1, features2.T), self.temperature)
            logits_ab_max, _ = torch.max(logits_ab, dim=1, keepdim=True)
            logits_ab = logits_ab - logits_ab_max.detach()
            logits = torch.cat([logits_ab, logits], dim=1)

        exp_logits = torch.exp(logits)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        if features2 is not None:
            logits_mask_ab = torch.ones_like(mask)
            positives_mask_ab = mask
            negatives_mask_ab = 1. - mask
            logits_mask = torch.cat([logits_mask_ab, logits_mask], dim=1)
            positives_mask = torch.cat([positives_mask_ab, positives_mask], dim=1)
            negatives_mask = torch.cat([negatives_mask_ab, negatives_mask], dim=1)

        num_positives_per_row = torch.sum(positives_mask, axis=1)
        num_negatives_per_row = torch.sum(negatives_mask, axis=1)
        denominator = torch.sum(exp_logits * logits_mask, axis=1, keepdims=True)
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs_a = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / \
                      num_positives_per_row[num_positives_per_row > 0]
        log_probs_b = torch.sum(log_probs * negatives_mask, axis=1)[num_negatives_per_row > 0] / \
                      num_negatives_per_row[num_negatives_per_row > 0]
        loss = - (log_probs_a.mean() - log_probs_b.mean())
        return loss


class CrossEntropyFocalLoss(CrossEntropyLoss):
    """
    Examples::

        >>> loss = CrossEntropyFocalLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, alpha: int = 1, gamma: int = 0, weight: Optional[Tensor] = None, size_average=None,
                 ignore_index: int = -100, reduce=None, reduction: str = 'none') -> None:
        super(CrossEntropyFocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiLossLayer(torch.nn.Module):
    def __init__(self, num_losses=1):
        super(MultiLossLayer, self).__init__()
        self.num_losses = num_losses
        self.sigmas = torch.nn.Parameter(torch.FloatTensor(self.num_losses))
        self.init_sigmas()

    def init_sigmas(self, ):
        self.sigmas = torch.nn.init.uniform_(self.sigmas, a=0, b=1)

    def forward(self, loss_list):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.sigmas = torch.nn.functional.normalize(self.sigmas, p=2, dim=0)
        sigmas2 = torch.pow(self.sigmas, 2)
        factor = torch.div(torch.tensor(1.0).to(device), torch.multiply(torch.tensor(2.0).to(device), sigmas2[0]))
        loss = torch.add(torch.multiply(factor, loss_list[0]), torch.log(self.sigmas[0]))
        for i in range(1, len(self.sigmas)):
            factor = torch.div(torch.tensor(1.0).to(device), sigmas2[i])
            loss = torch.add(loss, torch.add(torch.multiply(factor, loss_list[i]), torch.log(self.sigmas[i])))
        return loss


class LabelGuidedEmbedder(torch.nn.Module):
    def __init__(self, config, bert=None, labels=None):
        super(LabelGuidedEmbedder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model"], do_lower_case=True)
        if bert is None:
            self.bert = BertModelC.from_pretrained(config["bert_model"])
            self.bert_trainable = config["bert_trainable"]
            self.bert_hidden_size = self.bert.config.hidden_size
            self.word_lstm_hs = config["word_lstm_hs"]
            self.cacheable_tasks = config["cacheable_tasks"]
            for param in self.bert.parameters():
                param.requires_grad = self.bert_trainable
        else:
            self.bert = bert
            self.bert_hidden_size = self.bert.bert_hidden_size
            self.word_lstm_hs = config["word_lstm_hs"]
        self.label_name_ = labels if labels else LEGALEVAL_LABELS_PRES
        # self.label_name = self.tokenizer(self.label_name_, padding='longest', add_special_tokens=True)
        # for k, l in self.label_name.items():
        #     self.label_name[k] = torch.tensor(l).unsqueeze(0)
        # self.labels_embeddings = torch.nn.Parameter(self.label_name["input_ids"], requires_grad=False)
        # self.self_attn = SelfAttention(self.bert_hidden_size, 8, 0.1, 1e-5, 0.1)
        # self.cross_attn = BartAttention(self.bert_hidden_size, 8, 0.1, True)
        # self.label_proj = torch.nn.Linear(self.word_lstm_hs * 4, self.bert_hidden_size)

    def forward(self, batch, classifier=None):
        documents, sentences, tokens = batch["input_ids"].shape
        labels = [self.label_name_[index] for index in batch["labels"].squeeze()]
        labels_batch = self.tokenizer(labels, padding='longest', add_special_tokens=True)
        for k, l in labels_batch.items():
            labels_batch[k] = torch.tensor(l).to(batch["input_ids"].device)
            # labels_batch[k] = labels_batch[k].unsqueeze(0)
        # labels_embeddings = self.bert(labels_batch)[:, 0, :]
        label_name_embeddings = self.bert(input_ids=labels_batch["input_ids"].squeeze(),
                                          attention_mask=labels_batch["attention_mask"].squeeze())[0].mean(dim=1)
        if classifier:
            labels_embeddings = torch.index_select(classifier.state_dict()['weight'], 0,
                                                   index=batch['labels'].squeeze())
            # labels_embeddings = self.label_proj(labels_embeddings) # OOM
            labels_embeddings = label_name_embeddings + labels_embeddings
        if "bert_embeddings" in batch:
            inputs_embeds = batch['bert_embeddings']
            # labels_embeddings = self.self_attn(labels_embeddings.unsqueeze(0))[0]
            # label_embeds = labels_embeddings.expand(inputs_embeds.size(0), -1, -1)
            # attention_mask = batch["attention_mask"].squeeze()
            # label_attn_mask = torch.ones(sentences).to(attention_mask.device)
            # cross_attn_mask = (attention_mask * 1.).unsqueeze(-1).bmm((label_attn_mask.unsqueeze(0) * 1.).repeat(attention_mask.size(0), 1, 1))
            # new_bert_embeddings, attn_weights_reshaped, _ = self.cross_attn(inputs_embeds, label_embeds, attention_mask=cross_attn_mask.unsqueeze(1), output_attentions=True)

            # contrast_mask = torch.rand_like(inputs_embeds.to(dtype=torch.float32)) # [sentences, tokens, hidden_size]
            # contrast_mask = torch.rand_like(batch["attention_mask"].squeeze().to(dtype=torch.float32)) # [sentences, tokens]
            contrast_mask = torch.stack(
                [torch.matmul(inputs_embeds[i], labels_embeddings[i].T).T for i in range(sentences)])
            new_bert_embeddings = self.bert(inputs_embeds=batch['bert_embeddings'], embedding_weight=contrast_mask)[0]
        return new_bert_embeddings


class SimpleOutputLayer(torch.nn.Module):
    def __init__(self, config, in_dim=768, num_labels=14):
        super(SimpleOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        if config.get("focalloss", False):
            self.loss_fct = CrossEntropyFocalLoss(gamma=2)
        else:
            self.loss_fct = CrossEntropyLoss()

    def forward(self, x, mask, labels=None):
        batch_size, max_sequence, in_dim = x.shape
        logits = self.classifier(x)
        outputs = {}
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs["loss"] = loss
        else:
            predicted_label = torch.argmax(logits, axis=-1)
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label
        return outputs


class CRFOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''

    def __init__(self, in_dim, num_labels):
        super(CRFOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        self.crf = ConditionalRandomField(self.num_labels)

    def forward(self, x, mask, labels=None):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''

        batch_size, max_sequence, in_dim = x.shape

        logits = self.classifier(x)
        outputs = {}
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask)
            loss = -log_likelihood
            outputs["loss"] = loss
        else:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_label = [x for x, y in best_paths]
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label

            #log_denominator = self.crf._input_likelihood(logits, mask)
            #log_numerator = self.crf._joint_likelihood(logits, predicted_label, mask)
            #log_likelihood = log_numerator - log_denominator
            #outputs["log_likelihood"] = log_likelihood

        return outputs

class CRFPerTaskOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks):
        super(CRFPerTaskOutputLayer, self).__init__()

        self.per_task_output = torch.nn.ModuleDict()
        for task in tasks:
            self.per_task_output[task.task_name] = CRFOutputLayer(in_dim=in_dim, num_labels=len(task.labels))


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.per_task_output[task](x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for t, task_output in self.per_task_output.items():
                task_result = task_output(x, mask)
                task_result["task"] = t
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, task in enumerate(self.per_task_output.keys()):
            if index % 2 == 0:
                self.task_to_device[task] = device1
                self.per_task_output[task].to(device1)
            else:
                self.task_to_device[task] = device2
                self.per_task_output[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]



class AttentionPooling(torch.nn.Module):
    def __init__(self, in_features, dimension_context_vector_u=200, number_context_vectors=5):
        super(AttentionPooling, self).__init__()
        self.dimension_context_vector_u = dimension_context_vector_u
        self.number_context_vectors = number_context_vectors
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=self.dimension_context_vector_u, bias=True)
        self.linear2 = torch.nn.Linear(in_features=self.dimension_context_vector_u,
                                       out_features=self.number_context_vectors, bias=False)

        self.output_dim = self.number_context_vectors * in_features

    def forward(self, tokens, mask):
        #shape tokens: (batch_size, tokens, in_features)

        # compute the weights
        # shape tokens: (batch_size, tokens, dimension_context_vector_u)
        a = self.linear1(tokens)
        a = torch.tanh(a)
        # shape (batch_size, tokens, number_context_vectors)
        a = self.linear2(a)
        # shape (batch_size, number_context_vectors, tokens)
        a = a.transpose(1, 2)
        a = masked_softmax(a, mask)

        # calculate weighted sum
        s = torch.bmm(a, tokens)
        s = s.view(tokens.shape[0], -1)
        return s


class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        # state_dict_1 = self.bert.state_dict()
        # state_dict_2 = torch.load('/home/astha_agarwal/model/pytorch_model.bin')
        # for name2 in state_dict_2.keys():
        #    for name1 in state_dict_1.keys():
        #        temp_name = copy.deepcopy(name2)
        #       if temp_name.replace("bert.", '') == name1:
        #            state_dict_1[name1] = state_dict_2[name2]

        #self.bert.load_state_dict(state_dict_1,strict=False)

        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]

        #### break the large judgements into sentences chunk of given size. Do this while inference
        # chunk_size = 1024
        # input_ids = batch["input_ids"].view(-1, tokens)
        # chunk_cnt = int(math.ceil(input_ids.shape[0]/chunk_size))
        # input_ids_chunk_list = torch.chunk(input_ids,chunk_cnt)
        #
        # attention_mask_chunk_list = torch.chunk(attention_mask,chunk_cnt)
        # outputs = []
        # for input_ids,attention_mask in zip(input_ids_chunk_list,attention_mask_chunk_list):
        #     with torch.no_grad():
        #         output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #         output = output[0]
        #         #output = output[0].to('cpu')
        #     outputs.append(copy.deepcopy(output))
        #     torch.cuda.empty_cache()
        #
        # bert_embeddings = torch.cat(tuple(outputs))  #.to('cuda')

        if not self.bert_trainable and "task" in batch and batch["task"] in self.cacheable_tasks:
            # cache the embeddings of BERT if it is not fine-tuned
            # to save GPU memory put the values on CPU
            batch["bert_embeddings"] = bert_embeddings.to("cpu")

        return bert_embeddings

class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, tasks):
        super(BertHSLN, self).__init__()

        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.generic_output_layer = config.get("generic_output_layer")

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.init_sentence_enriching(config, tasks)

        self.reinit_output_layer(tasks, config)


    def init_sentence_enriching(self, config, tasks):
        input_dim = self.attention_pooling.output_dim
        print(f"Attention pooling dim: {input_dim}")
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

    def reinit_output_layer(self, tasks, config):
        if config.get("without_context_enriching_transfer"):
            self.init_sentence_enriching(config, tasks)
        input_dim = self.lstm_hidden_size * 2

        if self.generic_output_layer:
            self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
        else:
            self.crf = CRFPerTaskOutputLayer(input_dim, tasks)

    def forward(self, batch, labels=None, output_all_tasks=False):

        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)


        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)


        sentence_mask = batch["sentence_mask"]

        # shape: (documents, sentence, 2*lstm_hidden_size)
        sentence_embeddings_encoded = self.sentence_lstm(sentence_embeddings, sentence_mask)
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        if self.generic_output_layer:
            output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        else:
            output = self.crf(batch["task"], sentence_embeddings_encoded, sentence_mask, labels, output_all_tasks)


        return output


class BertHSLNMultiSeparateLayers(torch.nn.Module):
    '''
    Model Multi-Task Learning, where only certail layers are shared.
    This class is necessary to separate the model on two GPUs.
    '''

    def __init__(self, config, tasks):
        super(BertHSLNMultiSeparateLayers, self).__init__()


        self.bert = BertTokenEmbedder(config)


        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                                             hidden_size=self.lstm_hidden_size,
                                                             num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = PerTaskGroupWrapper(
                                        task_groups=config["attention_groups"],
                                        create_module_func=lambda g:
                                            AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])
                                )

        attention_pooling_output_dim = next(iter(self.attention_pooling.per_task_mod.values())).output_dim
        self.sentence_lstm = PerTaskGroupWrapper(
                                    task_groups=config["context_enriching_groups"],
                                    create_module_func=lambda g:
                                    PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=attention_pooling_output_dim,
                                        hidden_size=self.lstm_hidden_size,
                                        num_layers=1, batch_first=True, bidirectional=True))
                                    )

        self.crf = CRFPerTaskGroupOutputLayer(self.lstm_hidden_size * 2, tasks, config["output_groups"])



    def to_device(self, device1, device2):
        self.bert.to(device1)
        self.word_lstm.to(device1)
        self.attention_pooling.to_device(device1, device2)
        self.sentence_lstm.to_device(device1, device2)
        self.crf.to_device(device1, device2)
        self.device1 = device1
        self.device2 = device2

    def forward(self, batch, labels=None, output_all_tasks=False):
        task_name = batch["task"]
        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)

        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        device = self.attention_pooling.get_device(task_name)
        sentence_embeddings = self.attention_pooling(task_name, bert_embeddings_encoded.to(device), tokens_mask.to(device))
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)

        sentence_mask = batch["sentence_mask"]
        # shape: (documents, sentence, 2*lstm_hidden_size)
        device = self.sentence_lstm.get_device(task_name)
        sentence_embeddings_encoded = self.sentence_lstm(task_name, sentence_embeddings.to(device), sentence_mask.to(device))
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        device = self.crf.get_device(task_name)
        if labels is not None:
            labels = labels.to(device)

        output = self.crf(task_name, sentence_embeddings_encoded.to(device), sentence_mask.to(device), labels, output_all_tasks)

        return output

class CRFPerTaskGroupOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, tasks, task_groups):
        super(CRFPerTaskGroupOutputLayer, self).__init__()

        def get_task(name):
            for t in tasks:
                if t.task_name == name:
                    return t

        self.crf = PerTaskGroupWrapper(
                                        task_groups=task_groups,
                                        create_module_func=lambda g:
                                            # we assume same labels per group
                                            CRFOutputLayer(in_dim=in_dim, num_labels=len(get_task(g[0]).labels))
        )
        self.all_tasks = [t for t in [g for g in task_groups]]


    def forward(self, task, x, mask, labels=None, output_all_tasks=False):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''
        output = self.crf(task, x, mask, labels)
        if output_all_tasks:
            output["task_outputs"] = []
            assert labels is None
            for task in self.self.all_tasks:
                task_result = self.crf(task, x, mask, labels)
                task_result["task"] = task
                output["task_outputs"].append(task_result)
        return output

    def to_device(self, device1, device2):
        self.crf.to_device(device1, device2)

    def get_device(self, task):
        return self.crf.get_device(task)


class PerTaskGroupWrapper(torch.nn.Module):
    def __init__(self, task_groups, create_module_func):
        super(PerTaskGroupWrapper, self).__init__()

        self.per_task_mod = torch.nn.ModuleDict()
        for g in task_groups:
            mod = create_module_func(g)
            for t in g:
                self.per_task_mod[t] = mod

        self.task_groups = task_groups

    def forward(self, task_name, *args):
        mod = self.per_task_mod[task_name]
        return mod(*args)

    def to_device(self, device1, device2):
        self.task_to_device = dict()
        for index, tasks in enumerate(self.task_groups):
            for task in tasks:
                if index % 2 == 0:
                    self.task_to_device[task] = device1
                    self.per_task_mod[task].to(device1)
                else:
                    self.task_to_device[task] = device2
                    self.per_task_mod[task].to(device2)

    def get_device(self, task):
        return self.task_to_device[task]



class BertHSLNWithAuxiliaryTask(torch.nn.Module):
    '''
    Model for BertHSLN with additional auxiliary task, for instance, label shift prediction(LSP) task or BIOE task;
    For combination: to use the component of RR and the component of the auxiliary task parallely
                     to get the emission scores and then they are fed into the CRF of RR task
                     to get the appropriate probabilities for each RR label.
    '''

    def __init__(self, config, tasks):
        super(BertHSLNWithAuxiliaryTask, self).__init__()
        self.config = config
        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.generic_output_layer = config.get("generic_output_layer")

        self.lstm_hidden_size = config["word_lstm_hs"]
        self.word_lstm_layer = config.get("word_lstm_layer", 1)

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                                             hidden_size=self.lstm_hidden_size,
                                                             num_layers=self.word_lstm_layer, batch_first=True, bidirectional=True))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.init_sentence_enriching(config, tasks)
        self.span_task = config.get('span_task', None)
        self.auxiliary_task = config.get('auxiliary_task', None)
        if self.auxiliary_task and self.auxiliary_task == 'lsp':
            self.reinit_auxiliary_output_layer(num_labels=2)
        if self.auxiliary_task and self.auxiliary_task == 'bioe':
            self.reinit_auxiliary_output_layer(num_labels=4)
        self.contrastive_loss = config.get('contrastive_loss', None)
        self.supervised_contrastive_loss = config.get('supervised_contrastive_loss', None)
        self.data_augmentation_strategy = config.get('data_augmentation_strategy', None)
        self.document_supervised_contrastive_loss = config.get('document_supervised_contrastive_loss', None)
        self.temperature = config.get('temperature', 1.0)
        if self.contrastive_loss:
            self.con_loss = ContrastiveLoss(temperature=self.temperature)
        if self.supervised_contrastive_loss:
            self.con_dropout = torch.nn.Dropout(config.get("con_dropout", config["dropout"]))
            self.con_loss = SupervisedContrastiveLoss(temperature=self.temperature)
        if self.document_supervised_contrastive_loss:
            self.con_loss = DocumentSupervisedContrastiveLoss(temperature=self.temperature)
        self.reinit_output_layer(tasks, config)

        self.use_multi_loss = config.get('use_multi_loss', 0)
        if self.use_multi_loss != 0:
            print("self.use_multi_loss: ", self.use_multi_loss)
            self.loss_layer = MultiLossLayer(self.use_multi_loss)
        if self.data_augmentation_strategy == 'label-guided':
            print("self.data_augmentation_strategy: ", self.data_augmentation_strategy)
            self.label_guided_embedder = LabelGuidedEmbedder(config)

        hidden_dim = self.lstm_hidden_size * 2
        if self.auxiliary_task is not None:
            hidden_dim *= 2
        self.label_contrastive_loss = None
        self.discount_label_loss = None
        num_tags = config.get("num_tags")
        if config.get("label_contrastive_loss", None) is not None:
            self.label_contrastive_loss = LabelContrastiveLoss(num_tags, hidden_dim, get_device(0))
        if config.get("discount_label_loss", None) is not None:
            self.discount_label_loss = DiscountedTypeLoss(num_tags, hidden_dim, get_device(0))
        if config.get("span_model", None) is not None:
            self.span_model = SpanModel(num_tags)

    def init_sentence_enriching(self, config, tasks):
        input_dim = self.attention_pooling.output_dim
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                                                 hidden_size=self.lstm_hidden_size,
                                                                 num_layers=1, batch_first=True, bidirectional=True))

    def reinit_output_layer(self, tasks, config):
        layer_type = config.get("output_layer", "crf")
        if self.auxiliary_task:
            input_dim = self.lstm_hidden_size * 4
        else:
            input_dim = self.lstm_hidden_size * 2
        self.output_layer = None
        if layer_type == "simple":
            self.output_layer = SimpleOutputLayer(config, in_dim=input_dim, num_labels=len(tasks[0].labels))
        else:
            if config.get("without_context_enriching_transfer"):
                self.init_sentence_enriching(config, tasks)
            if self.generic_output_layer:
                self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
            else:
                self.crf = CRFPerTaskOutputLayer(input_dim, tasks)


    def reinit_auxiliary_output_layer(self, num_labels):
        input_dim = self.attention_pooling.output_dim
        self.auxiliary_sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                                                           hidden_size=self.lstm_hidden_size,
                                                                           num_layers=1, batch_first=True,
                                                                           bidirectional=True))

        input_dim = self.lstm_hidden_size * 2
        self.auxiliary_crf = CRFOutputLayer(in_dim=input_dim, num_labels=num_labels)

    def forward(self, batch, labels=None, auxiliary_labels=None, output_all_tasks=False):

        documents, sentences, tokens = batch["input_ids"].shape

        start_ids, end_ids = None, None
        if "start_ids" in batch:
            start_ids = batch["start_ids"]
        if "end_ids" in batch:
            end_ids = batch["end_ids"]

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)

        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)
        sentence_mask = batch["sentence_mask"]

        # shape: (documents, sentence, 2*lstm_hidden_size)
        sentence_embeddings_encoded_ = self.sentence_lstm(sentence_embeddings, sentence_mask)

        span_output = {}
        if self.span_task is not None:
            if labels is not None:
                span_outputs = self.span_model(sentence_embeddings_encoded, sentence_mask, start_ids, end_ids, labels)
                span_output["loss"] = span_outputs["loss"]
            else:
                span_outputs = self.span_model(sentence_embeddings_encoded, sentence_mask, None, None, None)
                predict_labels = span_outputs["predict_labels"]
                span_predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predict_labels]

                predicted_label = torch.tensor(span_predicted_label)
                span_output["predicted_label"] = predicted_label

        if self.auxiliary_task is not None:
            auxiliary_sentence_embeddings_encoded = self.auxiliary_sentence_lstm(sentence_embeddings, sentence_mask)
            auxiliary_sentence_embeddings_encoded = self.dropout(auxiliary_sentence_embeddings_encoded)
            auxiliary_output = self.auxiliary_crf(auxiliary_sentence_embeddings_encoded, sentence_mask,
                                                  labels=auxiliary_labels)
            sentence_embeddings_encoded = torch.cat(
                [sentence_embeddings_encoded_, auxiliary_sentence_embeddings_encoded], axis=-1)
            sentence_embeddings_encoded_ = sentence_embeddings_encoded

        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded_)
        if self.output_layer is not None:
            output = self.output_layer(sentence_embeddings_encoded, sentence_mask, labels)
        else:
            if self.generic_output_layer:
                output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
            else:
                output = self.crf(batch["task"], sentence_embeddings_encoded, sentence_mask, labels, output_all_tasks)

        if labels is not None and self.data_augmentation_strategy is not None:
            f1 = sentence_embeddings_encoded_
            if self.data_augmentation_strategy == 'dropout':
                f2 = sentence_embeddings_encoded
            else:
                if self.data_augmentation_strategy == 'shuffle':
                    mask_zeros1 = torch.tensor([[1 if i == 0 else 0 for i in range(tokens)]] * sentences).to(
                        bert_embeddings.device)
                    mask_zeros2 = torch.tensor(
                        [[-1 if i == tokens_mask[j].sum() else 0 for i in range(tokens)] for j in range(sentences)]).to(
                        bert_embeddings.device)
                    mask_range = torch.tensor([[1] + list(range(1, tokens))] * sentences).to(bert_embeddings.device)
                    mask_range = (1 - (tokens_mask + mask_zeros1 + mask_zeros2)) * mask_range
                    rand = torch.rand_like(batch['input_ids'].view(-1, tokens).to(dtype=torch.float64))
                    indices = torch.argsort(rand + mask_range, dim=-1)
                    indices = indices.unsqueeze(2).expand(indices.shape[0], indices.shape[1], bert_embeddings.shape[-1])
                    new_bert_embeddings = torch.gather(bert_embeddings, dim=-1, index=indices)
                if self.data_augmentation_strategy == 'label-guided':
                    batch['bert_embeddings'] = bert_embeddings
                    batch['labels'] = labels
                    new_bert_embeddings = self.label_guided_embedder(batch,
                                                                     self.crf.per_task_output['legaleval'].classifier)
                bert_embeddings_encoded = self.word_lstm(new_bert_embeddings, tokens_mask)
                sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
                sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
                sentence_embeddings = self.dropout(sentence_embeddings)
                sentence_mask = batch["sentence_mask"]
                sentence_embeddings_encoded_ = self.sentence_lstm(sentence_embeddings, sentence_mask)
                if self.auxiliary_task is not None:
                    auxiliary_sentence_embeddings_encoded = self.auxiliary_sentence_lstm(sentence_embeddings,
                                                                                         sentence_mask)
                    auxiliary_sentence_embeddings_encoded = self.dropout(auxiliary_sentence_embeddings_encoded)
                    sentence_embeddings_encoded = torch.cat(
                        [sentence_embeddings_encoded_, auxiliary_sentence_embeddings_encoded], axis=-1)
                    sentence_embeddings_encoded_ = sentence_embeddings_encoded
                f2 = sentence_embeddings_encoded_

            if self.contrastive_loss:
                output['contrastive_loss'] = self.con_loss(f1, f2)
            if self.supervised_contrastive_loss or self.document_supervised_contrastive_loss:
                output['contrastive_loss'] = self.con_loss(f1, f2, labels=labels)
            if self.auxiliary_task is not None:
                output["auxiliary_loss"] = auxiliary_output['loss']
            if self.use_multi_loss > 0 and 'contrastive_loss' in output and 'auxiliary_loss' in output:
                loss_list = [output['loss'], output['contrastive_loss'], output['auxiliary_loss']]
                assert len(loss_list) == self.loss_layer.num_losses, 'the number of losses is not correct'
                loss = self.loss_layer(loss_list)
                output['loss'] = loss
            if self.label_contrastive_loss:
                label_contrastive_loss = self.label_contrastive_loss(sentence_embeddings_encoded_, labels)
                output["label_contrastive_loss"] = label_contrastive_loss
            if self.discount_label_loss:
                discount_label_loss = self.discount_label_loss(sentence_embeddings_encoded_, labels)
                output["discount_label_loss"] = discount_label_loss
            if self.span_task is not None:
                output["span_loss"] = span_output['loss']
        if self.config.get("span_labels", None):
            output["predicted_label"] = span_output["predicted_label"]
        return output



class DeBertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(DeBertTokenEmbedder, self).__init__()
        self.bert = DebertaV2Model.from_pretrained(config["bert_model"])
        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape
        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]
        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = outputs[0]
        if not self.bert_trainable and "task" in batch and batch["task"] in self.cacheable_tasks:
            batch["bert_embeddings"] = bert_embeddings.to("cpu")
        return bert_embeddings


class Decoder(torch.nn.Module):
    def __init__(self,
                 output_size=2,
                 embedding_size=128,
                 hidden_size=256,
                 n_layers=4,
                 dropout=0.5,
                 labels=None):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = torch.nn.Linear(embedding_size, embedding_size)
        self.rnn = torch.nn.LSTM(input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        x = x.unsqueeze(1)

        # embedded = [1, batch size, embedding size]
        embedded = self.dropout(F.relu(self.embedding(x)))
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hidden size]
        # hidden = [n layers, batch size, hidden size]
        # cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        return output, hidden, cell


class BertHSLNDec(torch.nn.Module):
    def __init__(self, config, tasks):
        super(BertHSLNDec, self).__init__()
        self.label_name = LEGALEVAL_LABELS_DETAILS
        self.num_labels = len(self.label_name)
        self.teacher_forcing_ratio = 0.5
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model"], do_lower_case=True)
        self.bert = BertTokenEmbedder(config)
        self.dropout = torch.nn.Dropout(config["dropout"])
        self.lstm_hidden_size = config["word_lstm_hs"]
        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                                             hidden_size=self.lstm_hidden_size,
                                                             num_layers=1, batch_first=True, bidirectional=True))
        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])
        self.init_sentence_enriching(config, tasks)
        self.decoder = Decoder(output_size=14, embedding_size=self.lstm_hidden_size,
                               hidden_size=self.lstm_hidden_size,
                               n_layers=1, dropout=0.1)
        self.output_layer = SimpleOutputLayer(config, in_dim=self.lstm_hidden_size, num_labels=self.num_labels)

    def init_sentence_enriching(self, config, tasks):
        input_dim = self.attention_pooling.output_dim
        self.sentence_lstm = torch.nn.LSTM(input_size=input_dim,
                                           hidden_size=int(self.lstm_hidden_size/2),
                                           num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, batch, labels=None):
        documents, sentences, tokens = batch["input_ids"].shape
        target_len = sentences
        # if labels is not None:
        #     labels_embeddings = []
        #     batch_size = labels.shape[0]
        #     for i in range(batch_size):
        #         label_names = [self.label_name[i] for i in labels[-1, :]]
        #         labels_batch = self.tokenizer(label_names, padding='longest', add_special_tokens=True)
        #         for k, l in labels_batch.items():
        #             labels_batch[k] = torch.tensor(l).unsqueeze(0).to(labels.device)
        #         labels_embedding = self.bert(labels_batch)[:, 0, :]
        #         labels_embeddings.append(labels_embedding.unsqueeze(0))
        #     labels_embeddings = torch.concat(labels_embeddings, dim=0)

        bert_embeddings = self.bert(batch)
        bert_embeddings = self.dropout(bert_embeddings)
        tokens_mask = batch["attention_mask"].view(-1, tokens)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        sentence_embeddings = self.dropout(sentence_embeddings)
        sentence_mask = batch["sentence_mask"]

        sentence_embeddings_encoded, (hidden, cell) = self.sentence_lstm(sentence_embeddings)
        # sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded_)
        # output = self.output_layer(sentence_embeddings_encoded, sentence_mask, labels)
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros([sentences, documents, self.lstm_hidden_size]).to(bert_embeddings.device)

        # # first input to decoder is last coordinates of x
        # decoder_input = sentence_embeddings_encoded[:, -1, :]
        # for i in range(target_len):
        #     # run decode for one time step
        #     output, hidden, cell = self.decoder(decoder_input, hidden, cell)
        #     # place predictions in a tensor holding predictions for each time step
        #     outputs[i] = output.transpose(0, 1)
        #     # decide if we are going to use teacher forcing or not
        #     teacher_forcing = random.random() < self.teacher_forcing_ratio
        #     # output is the same shape as input, [batch_size, feature size]
        #     # so we can use output directly as input or use true lable depending on
        #     # teacher_forcing is true or not
        #     decoder_input = labels_embeddings[:, i, :] if labels is not None and teacher_forcing else output[:, -1, :]
        for i in range(target_len):
            decoder_input = sentence_embeddings_encoded[:, i, :]
            hidden = hidden.view(1, documents, -1)
            cell = cell.view(1, documents, -1)
            # output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            output, _, _ = self.decoder(decoder_input, hidden, cell)
            outputs[i] = output.transpose(0, 1)
            # teacher_forcing = random.random() < self.teacher_forcing_ratio
            # teacher_forcing = True
            # decoder_input = labels_embeddings[:, i, :] if labels is not None and teacher_forcing else output[:, -1, :]
        output = self.output_layer(outputs.transpose(0, 1), sentence_mask, labels)
        return output

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(torch.nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = torch.nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(torch.nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.1):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerEncoder(torch.nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = torch.nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class BertHSLNMultiHeads(torch.nn.Module):
    '''
    '''
    def __init__(self, config, tasks):
        super(BertHSLNMultiHeads, self).__init__()

        self.bert = BertTokenEmbedder(config)
        self.config = config
        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.generic_output_layer = config.get("generic_output_layer")

        self.lstm_hidden_size = config["word_lstm_hs"]
        if self.config.get("word_lstm", False):
            self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                    hidden_size=self.lstm_hidden_size,
                                    num_layers=1, batch_first=True, bidirectional=True))

            self.attention_pooling = AttentionPooling(self.lstm_hidden_size * 2,
                                                    dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                    number_context_vectors=config["att_pooling_num_ctx"])
        self.init_sentence_enriching(config, tasks)
        self.reinit_output_layer(tasks, config)


    def init_sentence_enriching(self, config, tasks):
        if config.get("word_lstm", False):
            input_dim = self.attention_pooling.output_dim
        else:
            input_dim = self.lstm_hidden_size
        self.output_dim = config.get("model_dim", self.lstm_hidden_size)
        num_heads = config.get("num_heads", 1)

        # Input dim -> Model dim
        self.input_net = torch.nn.Sequential(
            torch.nn.Dropout(config.get("dropout", 0.1)),
            torch.nn.Linear(input_dim, self.output_dim)
        )

        # self.output_dim = self.lstm_hidden_size
        # self.multihead_attn = torch.nn.MultiheadAttention(input_dim, 1, batch_first=True) # why so bad?
        # self.multihead_attn = EncoderBlock(input_dim, num_heads,self. output_dim)

        # implemetation of uvadlc-notebooks
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.output_dim)
        self.transformer = TransformerEncoder(num_layers=config.get("num_layers", 1),
                                              input_dim=self.output_dim,
                                              dim_feedforward=2*self.output_dim,
                                              num_heads=num_heads,
                                              dropout=config.get("dropout", 0.1))

    def reinit_output_layer(self, tasks, config):
        input_dim = self.output_dim
        if self.generic_output_layer:
            self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
        else:
            self.crf = CRFPerTaskOutputLayer(input_dim, tasks)

    def forward(self, batch, labels=None, output_all_tasks=False):

        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)

        if self.config.get("word_lstm", False):
            # shape (documents*sentences, tokens, 2*lstm_hidden_size)
            bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)
            bert_embeddings_encoded = self.dropout(bert_embeddings_encoded)
            # shape (documents*sentences, pooling_out)
            sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        else:
            if self.config.get("bert_embedding_type", "CLS") == "CLS":
                sentence_embeddings = bert_embeddings[:, 0, :]
            else:
                sentence_embeddings = torch.mean(bert_embeddings, dim=1)

        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)
        sentence_mask = batch["sentence_mask"]

        attn_mask = torch.tensor(sentence_mask, dtype=torch.float32)
        attn_mask = attn_mask.T * attn_mask
        # attn_output, attn_output_weights = self.multihead_attn(sentence_embeddings, sentence_embeddings, sentence_embeddings)
        # sentence_embeddings_encoded = self.multihead_attn(sentence_embeddings, mask=attn_mask)
        sentence_embeddings = self.input_net(sentence_embeddings)
        sentence_embeddings = self.positional_encoding(sentence_embeddings)
        sentence_embeddings_encoded = self.transformer(sentence_embeddings, mask=attn_mask)

        if self.generic_output_layer:
            output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        else:
            output = self.crf(batch["task"], sentence_embeddings_encoded, sentence_mask, labels, output_all_tasks)
        return output



class BertHSLNGRU(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, tasks):
        super(BertHSLNGRU, self).__init__()

        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.generic_output_layer = config.get("generic_output_layer")

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.GRU(input_size=self.bert.bert_hidden_size,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.init_sentence_enriching(config, tasks)

        self.reinit_output_layer(tasks, config)


    def init_sentence_enriching(self, config, tasks):
        input_dim = self.attention_pooling.output_dim
        print(f"Attention pooling dim: {input_dim}")
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.GRU(input_size=input_dim,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

    def reinit_output_layer(self, tasks, config):
        if config.get("without_context_enriching_transfer"):
            self.init_sentence_enriching(config, tasks)
        input_dim = self.lstm_hidden_size * 2

        if self.generic_output_layer:
            self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=len(tasks[0].labels))
        else:
            self.crf = CRFPerTaskOutputLayer(input_dim, tasks)

    def forward(self, batch, labels=None, output_all_tasks=False):

        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)


        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)


        sentence_mask = batch["sentence_mask"]

        # shape: (documents, sentence, 2*lstm_hidden_size)
        sentence_embeddings_encoded = self.sentence_lstm(sentence_embeddings, sentence_mask)
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)

        if self.generic_output_layer:
            output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        else:
            output = self.crf(batch["task"], sentence_embeddings_encoded, sentence_mask, labels, output_all_tasks)


        return output
