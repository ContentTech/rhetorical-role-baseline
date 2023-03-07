#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import os
import json
import argparse
import pandas as pd
from termcolor import colored
from copy import deepcopy
import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-i', '--input', type=str, help='input file')
    parser.add_argument('-o', '--output', type=str, help='output file')
    args = parser.parse_args()
    print("argparse.args=", args, type(args))
    d = args.__dict__
    for key, value in d.items():
        print('%s = %s'%(key, value))
    return args

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch_device)
def translation(tokenizer, model, input):
    input_ids = tokenizer.encode(input, return_tensors="pt").to(torch_device)
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

def en_de(tokenizer, model, input):
    inputs = tokenizer(input, return_tensors="pt").to(torch_device)
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("de"))
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return decoded

def data_reader(input_json):
    # mname_en_de = "/mnt/fengyao.hjj/pretrained_models/wmt19-en-de"
    mname_en_de = "/mnt/fengyao.hjj/pretrained_models/wmt21-dense-24-wide-en-x"
    tokenizer_en_de = AutoTokenizer.from_pretrained(mname_en_de)
    model_en_de = AutoModelForSeq2SeqLM.from_pretrained(mname_en_de).to(torch_device)
    mname_de_en = "/mnt/fengyao.hjj/pretrained_models/wmt19-de-en"
    tokenizer_de_en = FSMTTokenizer.from_pretrained(mname_de_en)
    model_de_en = FSMTForConditionalGeneration.from_pretrained(mname_de_en).to(torch_device)

    json_format = json.load(open(input_json, 'r'))
    print('train: ', len(json_format), type(json_format))
    files_de = []
    files_en = []
    for file_ori_en in json_format:
        file = deepcopy(file_ori_en)
        file_en = deepcopy(file_ori_en)
        for sent_id, annotation in enumerate(file['annotations'][0]['result']):
            sentence_txt = annotation['value']['text']
            sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").split())
            sentence_txt_de = translation(tokenizer_en_de, model_en_de, sentence_txt)
            sentence_txt_en = translation(tokenizer_de_en, model_de_en, sentence_txt_de)
            # print(colored(sentence_txt, 'green'))
            # print(colored(sentence_txt_de, 'red'))
            # print(colored(sentence_txt_en, 'blue'))
            file['annotations'][0]['result'][sent_id]['value']['text'] = sentence_txt_de
            file_en['annotations'][0]['result'][sent_id]['value']['text'] = sentence_txt_en
        files_de.append(file)
        files_en.append(file_en)
    fw_de = open('.'.join(input_json.split('.')[:-1] + ['de.wmt21', 'json']), 'w')
    json.dump(files_de, fw_de, ensure_ascii=True)
    fw_en = open('.'.join(input_json.split('.')[:-1] + ['en.wmt19', 'json']), 'w')
    json.dump(files_en, fw_en, ensure_ascii=True)
    # return output_file


if __name__ == "__main__":
    args = cmd()
    print('BASEDIR: ', BASEDIR)
    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/pubmed-20k/train.json"
    data_reader(train_file)
