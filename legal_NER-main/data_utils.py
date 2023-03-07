import os
import json
from tqdm import trange
from sklearn.model_selection import train_test_split, KFold
import collections
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import bisect
import gc
import glob
import random


BERT_VOCAB = os.path.join("/ossfs/workspace/legal_rr/ptm/checkpoint-48000/")
MAX_SEQ_LENGTH = 512

tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

def get_tokenizer():
    return tokenizer


class InputExample:
    def __init__(self,
                 text_type, # preamble or judgement 
                 text_id,
                 text,
                 labels=None):
        self.text_type = text_type
        self.text_id = text_id
        self.text = text
        self.pseudo = pseudo
        self.distant_labels = distant_labels
        self.text_id = text_id
        self.doc_labels = doc_labels
        self.level_info = level_info

def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _stat_class(dataset):
    convert_y = list()
    dataset_y = [example['labels'] for example in dataset]
    for item in dataset_y:
        convert_y.extend([i['type'] for i in item])
    dataset_res = collections.Counter(convert_y).most_common(8)

    stat_result = dict()

    for item in dataset_res:
        stat_result.update({item[0]: [item[1], item[1] / len(convert_y)]})
    return stat_result

def _compute_kl_divergence(vector_1, vector_2):
    kl_value = np.sum(np.multiply(vector_1, np.log2(
        np.multiply(vector_1, 1 / vector_2))))

    entropy_value = np.sum(np.multiply(vector_1, np.log2(1 / vector_1)))  # 交叉熵
    
    ratio = kl_value / entropy_value  # 信息量损失比例
    return kl_value, ratio


def convert_preamble_data_to_json(max_len, base_path, crf_dict, span_dict, target_path):
    # judgement_train_dir = os.path.join(train_base_dir, 'NER_TRAIN_JUDGEMENT.json')
    # judgement_dev_dir = os.path.join(dev_base_dir, 'NER_DEV_JUDGEMENT.json')
    # preamble_train_dir = os.path.join(base_path, 'NER_TRAIN_PREAMBLE.json')
    datas = json.loads(open(base_path).read())
    examples = []
    
    for data in datas:
        _id = data["id"]
        text = data["data"]["text"]
        sen_token_strs = tokenizer.tokenize(text)
        
        length = max_len - 2 
        sen_token_strs_list = []
        sen_tokens_list = []
        while len(sen_token_strs) > length:
            sen_token_strs_list.append(sen_token_strs[:length])
            sen_tokens = tokenizer.convert_tokens_to_ids(sen_token_strs[:length])
            sen_tokens.insert(0, 101)
            sen_tokens.append(102)
            if len(sen_tokens) < max_len:
                sen_tokens += [0] * (max_len - len(sen_tokens))
            sen_tokens_list.append(sen_tokens)
            sen_token_strs = sen_token_strs[length:]
            
        if len(sen_token_strs) > 0 and len(sen_token_strs) <= length:
            sen_token_strs_list.append(sen_token_strs)
            sen_tokens = tokenizer.convert_tokens_to_ids(sen_token_strs)
            sen_tokens.insert(0, 101) # CLS
            sen_tokens.append(102) #SEP
            if len(sen_tokens) < max_len:
                sen_tokens += [0] * (max_len - len(sen_tokens))
            sen_tokens_list.append(sen_tokens)
        
        for idx, sen_token_strs in enumerate(sen_token_strs_list):
            sen_tokens = sen_tokens_list[idx]
            sen_label_new = [''] * len(sen_token_strs)
            attention_mask = [0] *   max_len  
            token_types = [1] *  max_len 
            
            annotates = data["annotations"]
            has_valid_entity = False
            for annotate in annotates:
                for result in annotate["result"]:
                    start = result["value"]["start"]
                    end = result["value"]["end"]
                    labels = result["value"]["labels"]
                    
                    entity = result["value"]["text"]
                    entity_token_strs = tokenizer.tokenize(entity)
                    if len(entity_token_strs) < 1:
                        continue
                    for idx, item in enumerate(sen_token_strs):
                        if item == entity_token_strs[0]:
                            flag = idx
                            for k in range(1, len(entity_token_strs)):
                                if idx + k >= len(sen_token_strs) or sen_token_strs[idx+k] != entity_token_strs[k]:
                                    flag = -1 
                                    break
                            if flag!=-1:
                                has_valid_entity = True
                                for j in range(len(entity_token_strs)):
                                    sen_label_new[flag+j] = labels[0]
            if not has_valid_entity:
                continue
            sen_token_strs.insert(0, "[CLS]") 
            sen_token_strs.append("[SEP]")
            sen_label_new.insert(0, '')
            sen_label_new.append('')

            for i in range(len(sen_token_strs)):
                attention_mask[i] = 1 
                token_types[i] = 0
            
            start_ids = [0] * max_len
            end_ids = [0] * max_len 
            text_labels = [0] * max_len

            prev = ''
            for idx, tag in enumerate(sen_label_new):
                if tag != '':
                    if prev != tag:
                        if idx+1 >= len(sen_label_new) or sen_label_new[idx] != sen_label_new[idx+1]: # single
                            tt = crf_dict["S-"+tag]
                            text_labels[idx] = tt 
                            tt = span_dict[tag]
                            start_ids[idx] = tt 
                            end_ids[idx] = tt 
                        # begin 
                        else:
                            tt = crf_dict["B-"+tag]
                            text_labels[idx] = tt 
                            tt = span_dict[tag]
                            start_ids[idx] = tt 
                    elif prev == tag:
                        #end
                        if idx+1 >= len(sen_label_new) or sen_label_new[idx] != sen_label_new[idx+1]:
                            tt = crf_dict["E-"+tag]
                            text_labels[idx] = tt 
                            tt = span_dict[tag]
                            end_ids[idx] = tt 
                        #mid
                        else:
                            tt = crf_dict["I-"+tag]
                            text_labels[idx] = tt 
                prev = tag        
                                            
            # print ("content", sen_token_strs)
            # print ("sentence_ids", sen_tokens)
            # print ("labels", text_labels)
            # print ("length", len(sen_tokens), len(text_labels), len(start_ids), len(end_ids))
            # print ("start_ids", start_ids)
            # print ("end_ids", end_ids)
            # print ("attention_mask", attention_mask)
            input_exmple = {"id":_id,
                            "text":sen_token_strs,
                            "input_ids":sen_tokens,
                            "attention_mask":attention_mask,
                            "token_type_ids": token_types,
                            "labels": text_labels,
                            "start_ids": start_ids,
                            "end_ids": end_ids
                            }
            examples.append(input_exmple)
    return examples


def convert_judgement_data_to_json(max_len, base_path, crf_dict, span_dict, target_path):
    datas = json.loads(open(base_path).read())
    examples = []
    for data in datas:
        text = data["data"]["text"]
        sen_token_strs = tokenizer.tokenize(text)
        sent_tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_len,
                                               truncation=True, pad_to_max_length=True)
        _id = data["id"]
        # 最长510
        sen_token_strs = sen_token_strs if len(sen_token_strs)<=max_len - 2 else sen_token_strs[:max_len - 2]
        sent_tokens = sen_token_strs if len(sent_tokens)<=max_len - 2 else sent_tokens[:max_len - 2]
        
        sen_label_new = [''] * len(sen_token_strs)
        attention_mask = [0] *   max_len  
        token_types = [1] *   max_len 
        
        ents = []
        annotates = data["annotations"]

        has_valid_entity = False
        for annotate in annotates:
            for result in annotate["result"]:
                start = result["value"]["start"]
                end = result["value"]["end"]
                labels = result["value"]["labels"]
                
                entity = result["value"]["text"]
                ents.append([entity, labels[0]])
                entity_token_strs = tokenizer.tokenize(entity)
                if len(entity_token_strs) < 1:
                    continue
                for idx, item in enumerate(sen_token_strs):
                    if item == entity_token_strs[0]:
                        flag = idx
                        for k in range(1, len(entity_token_strs)):
                            if idx + k >= len(sen_token_strs) or sen_token_strs[idx+k] != entity_token_strs[k]:
                                flag = -1 
                                break
                        if flag!=-1:
                            has_valid_entity = True
                            for j in range(len(entity_token_strs)):
                                sen_label_new[flag+j] = labels[0]
        if not has_valid_entity:
            continue
        sen_token_strs.insert(0, "[CLS]") 
        sen_token_strs.append("[SEP]")
        sen_label_new.insert(0, '')
        sen_label_new.append('')

        for i in range(len(sen_token_strs)):
            attention_mask[i] = 1 
            token_types[i] = 0
        
        start_ids = [0] * max_len
        end_ids = [0] * max_len 
        text_labels = [0] * max_len

        prev = ''
        for idx, tag in enumerate(sen_label_new):
            if tag != '':
                if prev != tag:
                    if idx+1 >= len(sen_label_new) or sen_label_new[idx] != sen_label_new[idx+1]: # single
                        tt = crf_dict["S-"+tag]
                        text_labels[idx] = tt 
                        tt = span_dict[tag]
                        start_ids[idx] = tt 
                        end_ids[idx] = tt 
                    # begin 
                    else:
                        tt = crf_dict["B-"+tag]
                        text_labels[idx] = tt 
                        tt = span_dict[tag]
                        start_ids[idx] = tt 
                elif prev == tag:
                    #end
                    if idx+1 >= len(sen_label_new) or sen_label_new[idx] != sen_label_new[idx+1]:
                        tt = crf_dict["E-"+tag]
                        text_labels[idx] = tt 
                        tt = span_dict[tag]
                        end_ids[idx] = tt 
                    #mid
                    else:
                        tt = crf_dict["I-"+tag]
                        text_labels[idx] = tt 
            prev = tag        
                                          
        # print ("content", sen_token_strs)
        # print ("sentence_ids", sent_tokens_txt)
        # print ("labels", text_labels)
        # print ("length", len(sent_tokens_txt), len(text_labels), len(start_ids), len(end_ids))
        # print ("start_ids", start_ids)
        # print ("end_ids", end_ids)
        # print ("attention_mask", attention_mask)
        input_exmple = {"id":_id,
                        "text":sen_token_strs,
                        "input_ids":sent_tokens,
                        "attention_mask":attention_mask,
                        "token_type_ids": token_types,
                        "labels": text_labels,
                        "start_ids": start_ids,
                        "end_ids": end_ids,
                        "entities": ents
                        }
        examples.append(input_exmple)
    return examples     
    # break        
    # print (max_len, total, count)              
            
         


                # for label in labels:

                #     tag = span_dict[label]
                #     start_ids[start] = tag
                #     end_ids[end-1] = tag
                #     if end -1 == start:
                #         tag = crf_dict["S-"+label]
                #         text_labels[start] = tag
                #     else:
                #         tag = crf_dict["B-"+label]
                #         text_labels[start] = tag

                #         tag = crf_dict["E-"+label]
                #         text_labels[end-1] = tag

                #         for i in range(start+1, end-1):
                #             tag = crf_dict["I-"+label]
                #             text_labels[i] = tag
                               
    

    # with open(stack_dir,'r',encoding='utf8') as f:
    #     data = {}
    #     for line in f:
    #         unit = json.loads(line)
    #         stack_examples.append({'text_id':unit['text_id'] ,
    #                             'text': unit['text'],
    #                             'labels': unit['attributes'],
    #                             'level':[unit['level1'],unit['level2'],unit['level3']],
    #                             'pseudo': 0})


    # # 构建实体知识库
    # kf = KFold(10)
    # entities = set()
    # ent_types = set()
    # for _now_id, _candidate_id in kf.split(stack_examples):
    #     now = [stack_examples[_id] for _id in _now_id]
    #     candidate = [stack_examples[_id] for _id in _candidate_id]
    #     now_entities = set()

    #     for _ex in now:
    #         for _label in _ex['labels']:
    #             ent_types.add(_label['type'])
    #             if len(_label['entity']) > 1:
    #                 now_entities.add(_label['entity'])
    #                 entities.add(_label['entity'])

    #     # print(len(now_entities))
    #     for _ex in candidate:
    #         text = _ex['text']
    #         candidate_entities = []

    #         for _ent in now_entities:
    #             if _ent in text:
    #                 candidate_entities.append(_ent)

    #         _ex['candidate_entities'] = candidate_entities
    # # assert len(ent_types) == 13

    # # process test examples predicted by the preliminary model

    # # process test examples
    # with open(test_dir,'r',encoding='utf8') as f:
    #     data = {}
    #     for line in f:
    #         unit = json.loads(line)
    #         candidate_entities = []
    #         for _ent in entities:
    #             if _ent in unit['text']:
    #                 candidate_entities.append(_ent)
    #         test_examples.append({'text_id':unit['text_id'] ,
    #                             'text': unit['text'],
    #                             'level':[unit['level1'],unit['level2'],unit['level3']],
    #                             'pseudo': 0,
    #                             'candidate_entities': candidate_entities})

    
    # for i in range(10):
    #     train, dev = train_test_split(stack_examples, shuffle=True, test_size=0.2)
    #     dataset_stat = _stat_class(stack_examples)
    #     train_stat = _stat_class(train)
    #     dev_stat = _stat_class(dev)

    #     train_kl_value, train_ratio = _compute_kl_divergence(
    #         np.array([item[1][1] for item in sorted(dataset_stat.items())]),
    #         np.array([item[1][1] for item in sorted(train_stat.items())]))
    #     valid_kl_value, valid_ratio = _compute_kl_divergence(
    #         np.array([item[1][1] for item in sorted(dataset_stat.items())]),
    #         np.array([item[1][1] for item in sorted(dev_stat.items())]))
    #     print(train_ratio, valid_ratio)
    #     if train_ratio < 0.05 and valid_ratio<0.05:
    #         break


    # if save_data:
    #     save_info(base_dir, stack_examples, 'stack')
    #     save_info(base_dir, train, 'train')
    #     save_info(base_dir, dev, 'dev')
    #     save_info(base_dir, test_examples, 'test')

    # if save_dict:
    #     ent_types = list(ent_types)
    #     span_ent2id = {_type: i+1 for i, _type in enumerate(ent_types)}

    #     ent_types = ['O'] + [p + '-' + _type for p in ['B', 'I', 'E', 'S'] for _type in list(ent_types)]
    #     crf_ent2id = {ent: i for i, ent in enumerate(ent_types)}

    #     mid_data_dir = os.path.join(os.path.split(base_dir)[0], 'mid_data')
    #     if not os.path.exists(mid_data_dir):
    #         os.mkdir(mid_data_dir)

    #     save_info(mid_data_dir, span_ent2id, 'span_ent2id')
    #     save_info(mid_data_dir, crf_ent2id, 'crf_ent2id')


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        one_data = {
            "input_ids": torch.tensor(item['input_ids']).long(),
            "labels": torch.tensor(item['labels']).long(),
            "input_mask": torch.tensor(item['attention_mask']).long(),
            "token_type_ids": torch.tensor(item['token_type_ids']).long(),
            "start_ids": torch.tensor(item['start_ids']).long(),
            "end_ids": torch.tensor(item['end_ids']).long(),
            # "token_segs": item["text"],
            # "entities": item["entities"]
        }
        return one_data


def yield_data(data, batch_data):
    tmp = MyDataset(data)
    return DataLoader(dataset=tmp, batch_size=batch_data, shuffle=True)



# if __name__ == '__main__':
#     train_base_dir = '../data/'
#     dev_base_dir =  '../data/NER_DEV/'
#     preamble_train_dir = os.path.join(train_base_dir, 'NER_TRAIN_PREAMBLE.json')

#     span_id_dir = "../data/label2id/judgement/span_ent2id.json"
#     span_dict = json.loads(open(span_id_dir).read())

#     crf_id_dir = "../data/label2id/judgement/crf_ent2id.json"
#     crf_dict = json.loads(open(crf_id_dir).read())

#     crf_id_dir_preamble = "../data/label2id/preamble/crf_ent2id.json"
#     crf_dict_preamble = json.loads(open(crf_id_dir_preamble).read())


#     span_id_dir_preamble = "../data/label2id/preamble/span_ent2id.json"
#     span_dict_preamble = json.loads(open(span_id_dir_preamble).read())

#     convert_preamble_data_to_json(512, preamble_train_dir, crf_dict_preamble, span_dict_preamble, "")
    

#     train_base_dir = '../data/'
#     dev_base_dir =  '../data/NER_DEV/'

#     judgement_train_dir = os.path.join(train_base_dir, 'NER_TRAIN_JUDGEMENT.json')
#     judgement_dev_dir = os.path.join(dev_base_dir, 'NER_DEV_JUDGEMENT.json')
    
#     # preamble_train_dir = os.path.join(base_path, 'NER_TRAIN_PREAMBLE.json')
    
#     batch_size = 256

#     train_examples = convert_judgement_data_to_json(512, judgement_train_dir, "")
#     yield_data(train_examples, 256)

#     print ("train example number", len(train_examples))
#     dev_examples = convert_judgement_data_to_json(512, judgement_dev_dir, "")
#     print ("dev example number", len(dev_examples))

class Dataloader(object):
    def __init__(self, datasets,  batch_size,
                 device, shuffle=True, is_test=False):
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)
