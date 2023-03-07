#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
from termcolor import colored
import pandas as pd
import argparse
from tabulate import tabulate
# 显示所有列
# pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth', 100)


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-t', '--train', type=str, default=None, help='train.csv')
    parser.add_argument('-d', '--dev', type=str, default=None, help='dev.csv')
    args = parser.parse_args()
    print("argparse.args=", args, type(args))
    d = args.__dict__
    for key, value in d.items():
        print('%s = %s' % (key, value))
    return args


if __name__ == "__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_check_duplication.py \
    -t /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D8/train_scibert.txt \
    -d /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D8/test_scibert.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_check_duplication.py \
    -t /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D3/8/train_scibert.txt \
    -d /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D3/8/dev_scibert.txt

    cd /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D3/

    cd /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D9.1/all_dev143


    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_check_duplication.py \
    -t /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/4/train_scibert.txt \
    -d /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/4/dev_scibert.txt

    mv 0/train_scibert.txt 0/train_scibert_bp.txt 
    mv 0/train_scibert.dedup.txt 0/train_scibert.txt 
    mv 1/train_scibert.txt 1/train_scibert_bp.txt 
    mv 1/train_scibert.dedup.txt 1/train_scibert.txt 
    mv 2/train_scibert.txt 2/train_scibert_bp.txt 
    mv 2/train_scibert.dedup.txt 2/train_scibert.txt 
    mv 3/train_scibert.txt 3/train_scibert_bp.txt 
    mv 3/train_scibert.dedup.txt 3/train_scibert.txt 
    mv 4/train_scibert.txt 4/train_scibert_bp.txt 
    mv 4/train_scibert.dedup.txt 4/train_scibert.txt 
    mv 5/train_scibert.txt 5/train_scibert_bp.txt 
    mv 5/train_scibert.dedup.txt 5/train_scibert.txt 
    mv 6/train_scibert.txt 6/train_scibert_bp.txt 
    mv 6/train_scibert.dedup.txt 6/train_scibert.txt 
    mv 7/train_scibert.txt 7/train_scibert_bp.txt 
    mv 7/train_scibert.dedup.txt 7/train_scibert.txt 
    mv 8/train_scibert.txt 8/train_scibert_bp.txt 
    mv 8/train_scibert.dedup.txt 8/train_scibert.txt 
    mv 9/train_scibert.txt 9/train_scibert_bp.txt 
    mv 9/train_scibert.dedup.txt 9/train_scibert.txt 

    """
    args = cmd()
    train_file = args.train
    dev_file = args.dev

    if args.train.endswith('csv'):
        train_file = pd.read_csv(train_file, encoding='utf-8')
        dev_file = pd.read_csv(dev_file, encoding='utf-8')

        print(train_file.shape[0], dev_file.shape[0])
        print(train_file["unique_id"].unique())
        print(dev_file["unique_id"].unique())

        sentences = [str(s).strip() for s in list(train_file['context'])]
        doc_ids = [str(s).strip() for s in list(train_file['doc_id'])]
        unique_ids = [str(s).strip() for s in list(train_file['unique_id'])]
        files = [str(s).strip() for s in list(train_file['file'])]
        sent_ids = [str(s).strip() for s in list(train_file['sent_id'])]
        labels = [str(s).strip() for s in list(train_file['label'])]

        num = 0
        for s in dev_file.iterrows():
            if str(s[1]['context']).strip() in sentences:
                num += 1  
                print(colored('dev: ' + str(s[1]['unique_id']) + '\t' + str(s[1]['file']) + '\t' + str(s[1]['doc_id']) + '\t' + s[1]['label'], 'red'))
                index = sentences.index(s[1]['context'].strip())
                print(colored('train: ' + unique_ids[index] + '\t' + files[index] + '\t' + doc_ids[index] + '\t' + labels[index], 'green'))
        print(num)

    if args.train.endswith('txt'):
        train_lines = open(train_file, 'r').readlines()
        dev_lines = open(dev_file, 'r').readlines()
        sentences = [s.strip().split('\t')[-1] for s in train_lines if len(s.strip()) > 0 and not s.startswith("###")]
        dev_sentences = [s.strip().split('\t')[-1] for s in dev_lines if len(s.strip()) > 0 and not s.startswith("###")]
        print("sentences: ", len(sentences))
        print("dev_sentences: ", len(dev_sentences))
        num = 0
        for s in dev_sentences:
            if s in sentences:
                num += 1 
        print(num)

        dev_files = [s.strip() for s in dev_lines if s.startswith("###")]
        train_files = [s.strip() for s in train_lines if s.startswith("###")]
        print("train_files: ", train_files, len(train_files))
        print("\n\n")
        print("dev_files: ", dev_files, len(dev_files))
        duplicates = []
        for df in dev_files:
            for tf in train_files:
                if df.replace(".txt", "").split(".")[-1] == tf.replace(".txt", "").split(".")[-1]:
                    print(df, tf)
                    duplicates.append(tf)
        print(duplicates, len(duplicates))
        with open(".".join(args.train.split(".")[0:-1] + ['dedup', 'txt']), "w") as fw:
            flag = True
            for s in train_lines:
                if s.startswith("###") and s.strip() in duplicates:
                    flag = False
                elif s.startswith("###"):
                    flag = True
                if flag:
                    fw.write(s)

        train_lines = open(".".join(args.train.split(".")[0:-1] + ['dedup', 'txt']), 'r').readlines()
        train_files = [s.strip() for s in train_lines if s.startswith("###")]
        duplicates = []
        for df in dev_files:
            for tf in train_files:
                if df.replace(".txt", "").split(".")[-1] == tf.replace(".txt", "").split(".")[-1]:
                    print(df, tf)
                    duplicates.append(tf)
        print(duplicates, len(duplicates))