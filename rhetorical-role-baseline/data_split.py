#!/usr/bin/python
# -*- coding:utf-8 -*-
#****************************************************************#
# ScriptName: generate_labels.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-03-29 10:40
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-03-29 10:40
# Function:
#***************************************************************#

import sys
import os
import argparse
import logging
import json
import pandas as pd
from tabulate import tabulate
from termcolor import colored
import random
import math
import os
from transformers import BertTokenizer
import pandas as pd
from termcolor import colored

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MAX_SEQ_LENGTH = 256
BERT_VOCAB = os.path.join(BASEDIR, "results/bert_continue_bz64/checkpoint-48000")

#显示所有列
# pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
FORMAT = '%(asctime)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'user': 'fengyao'}
logger = logging.getLogger()
logger.warning('Test: %s', '0', extra=d)


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-i', '--input', nargs='+', type=str, help='input file')
    parser.add_argument('-o', '--output', type=str, help='output file')
    parser.add_argument('-d', '--dev', type=str, default=None, help='dev file')

    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


def write_in_hsln_format(input_dir, output_dir, type='train', auxiliary_task="bioe"):
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    # tokenizer = DebertaV2Tokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    hsln_format_txt_dirpath = output_dir
    json_format = json.load(open(input_dir))
    final_string = ''
    filename_sent_boundries = {}
    for file in json_format:
        file_name = file['id']
        previous_label = "mask"
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for ani, annotation in enumerate(file['annotations'][0]['result']):
            if ani < len(file['annotations'][0]['result']) - 1:
                next_label = file['annotations'][0]['result'][ani+1]['value']['labels'][0]
            else:
                next_label = "mask"
            filename_sent_boundries[file_name]['sentence_span'].append(
                [annotation['value']['start'], annotation['value']['end']])

            sentence_txt = annotation['value']['text']
            # sentence_txt = sentence_txt.replace("\r", "")
            sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=MAX_SEQ_LENGTH,
                                               truncation=True)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                if len(annotation['value']['labels']) != 1:
                    print(colored(annotation['value']['labels'], 'red'))
                if auxiliary_task == 'lsp':
                    if previous_label == annotation['value']['labels'][0]:
                        ls = "0"
                    else:
                        ls = "1"
                if auxiliary_task == 'bioe':
                    if annotation['value']['labels'][0] != next_label:
                        ls = "E"
                    if annotation['value']['labels'][0] != previous_label:
                        ls = "B"
                    if annotation['value']['labels'][0] == previous_label and annotation['value']['labels'][0] == next_label:
                        ls = "I"
                    if annotation['value']['labels'][0] == "NONE":
                        ls = "O"
                final_string = final_string + annotation['value']['labels'][
                    0] + "\t" + ls + "\t" + sent_tokens_txt + "\n"
                previous_label = annotation['value']['labels'][0]
        final_string = final_string + "\n"

    with open(hsln_format_txt_dirpath + f'/{type}_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + f'/{type}_sentece_boundries.json', 'w+') as json_file:
        json.dump(filename_sent_boundries, json_file)
    return filename_sent_boundries


def write_in_csv_format(input_json, output_file=None):
    json_format = json.load(open(input_json, 'r'))
    sentences = []
    labels = []
    doc_ids = []
    sent_ids = []
    sent_spans = []
    datas = []
    groups = []
    files = []
    unique_ids = []
    for file in json_format:
        for sent_id, annotation in enumerate(file['annotations'][0]['result']):
            sentence_txt = annotation['value']['text']
            sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
            if sentence_txt.strip() != "":
                sentences.append(sentence_txt)
                labels.append(annotation['value']['labels'][0])
                sent_ids.append(sent_id)
                datas.append(file['data'])
                groups.append(file['meta']['group'])
                doc_ids.append(file['id'])
                files.append(file['file'])
                unique_ids.append(file['unique_id'])
                sent_spans.append([annotation['value']['start'], annotation['value']['end']])

    data = {
        "context": sentences,
        "label": labels,
        "doc_id": doc_ids,
        "sent_id": sent_ids,
        "sent_spans": sent_spans,
        "datas": datas,
        "groups": groups,
        "file": files,
        "unique_id": unique_ids,
    }
    df = pd.DataFrame(data)
    if output_file:
        df.to_csv(output_file, index=False)
    return df



def train_dev_split(docs, must_in_dev=[], n=10, dn=30, data_dir=''):
    random.seed(1)
    random.shuffle(docs)
    print("docs: ", len(docs))

    for i, doc in enumerate(docs):
        doc["unique_id"] = f"{i}"

    if must_in_dev:
        for i, doc in enumerate(must_in_dev):
            doc["unique_id"] = f"dev.{i}"

    num = math.floor(len(docs)/n)
    docs_list = []
    train_list = []
    dev_list = []
    for i, index in enumerate(range(0, len(docs)-1, num)):
        if len(docs_list) == n - 1:
            print(i, index, index+num, len(docs[index:]), len(docs[index: len(docs)-dn]), len(docs[len(docs)-dn:]))
            docs_list.append(docs[index:])
            train_list.append(docs[index: len(docs)-dn])
            dev_list.append(docs[len(docs)-dn:])
            break
        else:
            print(i, index, index+num, len(docs[index: index+num]), len(docs[index: index+num-dn]), len(docs[index+num-dn: index+num]))
            docs_list.append(docs[index: index+num])
            train_list.append(docs[index: index+num-dn])
            dev_list.append(docs[index+num-dn: index+num])

    for i, docs in enumerate(docs_list):
        print(i, len(docs))
        d = docs_list[0:i] + train_list[i:i+1] + docs_list[i+1:]
        d = sum(d, [])
        # print("train_ids: ", [doc["unique_id"] + "=" + str(doc["id"]) for doc in d])
        # print("dev_ids: ", [doc["unique_id"] + "=" + str(doc["id"]) for doc in dev_list[i]])
        print(len(d), len(train_list[i:i+1][0]), len(dev_list[i] + must_in_dev))
        data_subdir = os.path.join(data_dir, f"{i}")
        try:
            os.mkdir(data_subdir)
        except Exception as e:
            print(e.__str__())
        train_json_file = os.path.join(data_subdir, f"train.{i}.json")
        dev_json_file = os.path.join(data_subdir, f"dev.{i}.json")
        print(train_json_file)
        print(dev_json_file)
        with open(train_json_file, 'w') as file:
            json.dump(d, file)
        with open(dev_json_file, 'w') as file:
            dev = must_in_dev + dev_list[i]
            json.dump(dev, file)

        write_in_csv_format(train_json_file, os.path.join(data_subdir, f"train.{i}.csv"))
        write_in_csv_format(dev_json_file, os.path.join(data_subdir, f"dev.{i}.csv"))
        # write_in_hsln_format(train_json_file, output_dir=data_dir, type='train')
        # write_in_hsln_format(dev_json_file, output_dir=data_dir, type='dev')

if __name__=="__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_split.py \
    -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/train.all.json \
    -i /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/train.json \
    /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en/train.en.json \
    /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/Y2021M8/data_annotated/y2021m8.json \
    /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/Y2019B7/y2019b7.json 

    /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D9.1/ILDC_multi_555_pred.json

    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_split.py \
    -o  /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/train.all.json \
    -d  /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/pubmed-20k/dev.json

    # /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/pubmed-20k/train.json \
    """
    args = cmd()
    print(args.input)

    if args.input and len(args.input) > 0:
        # 合并多个json files，同时把空句子,空文档以及长度大于512的文档过滤掉；
        # 将使用D3 split 10模型预测的训练数据里非常不好的几个doc去掉；
        all_docs = []
        # REMOVE_DOCS_LIST = [
        #     '0.186.4206', 
        #     '1.186.4206', 
        #     '2.17.COM_India_Trade_Promotion_Organisation_vs_Competition_TA2016080716152316141COM296986.txt', 
        #     '2.26.HC_Mahindra_Electric_Mobility_Limited_and_Ors_vs_CompDE201901051916442526COM475775.txt', 
        #     '2.36.SC_Excel Crop Care Limited vs Competition Commission of India and Ors 08052017 SC(1).txt', 
        #     '2.37.SC_Rajasthan Cylinders and Containers Limited vs Union of India UOI and Ors 01102018 SC(1).txt', 
        #     '2.41.SC_Competition Commission of India vs Bharti Airtel Limited and Ors 05122018 SC(1).txt', 
        #     '2.46.SC_Competition Commission of India vs Bharti Airtel Limited and Ors 05122018 SC(1).txt', 
        #     '2.75.SC_2007_560.txt', 
        #     '2.80.SC_2010_1296.txt', 
        #     '3.3.1963_S_59']
        REMOVE_DOCS_LIST = []

        for i, file in enumerate(args.input):
            docs = json.load(open(file, 'r'))
            new_docs = []
            for j, doc in enumerate(docs):
                annos = []
                for sent_id, annotation in enumerate(doc['annotations'][0]['result']):
                    sentence_txt = annotation['value']['text']
                    sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
                    if sentence_txt.strip() != "":
                        annos.append(annotation)
                doc['annotations'][0]['result'] = annos
                doc["file"] = os.path.split(file)[1]
                doc["type"] = "train"
                doc["id"] = f"{i}.{j}.{doc['id']}"
                doc["unique_id"] = f"{i}.{j}.{doc['id']}"
                if 512 > len(annos) > 0 and doc["id"] not in REMOVE_DOCS_LIST:
                    new_docs.append(doc)
                else:
                    print(f"remove doc id {doc['id']}.")
            all_docs += new_docs
        print(len(all_docs))
        with open(args.output, 'w') as file:
            json.dump(all_docs, file)

        print(os.path.join(os.path.dirname(args.output), f"train.all.csv"))
        write_in_csv_format(args.output, os.path.join(os.path.dirname(args.output), f"train.all.csv"))

    if args.dev:
        all_docs = json.load(open(args.output, 'r')) 
        dev_docs = json.load(open(args.dev, 'r')) 
        for i, doc in enumerate(dev_docs):
            doc["file"] = os.path.split(args.dev)[1]
            doc["type"] = "dev"
        docs = all_docs + dev_docs
        train_dev_split(docs, n=5, dn=50, data_dir=os.path.dirname(args.output))
        # docs = all_docs
        # train_dev_split(docs, must_in_dev=dev_docs, n=10, dn=30, data_dir=os.path.dirname(args.output))

        nums = []
        for doc in all_docs:
            nums.append(len(doc["annotations"][0]["result"]))
        print(nums)
        print(max(nums))