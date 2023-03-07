#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import argparse
import pandas as pd
from termcolor import colored


def write_in_csv_format(input, output_file):
    sentences = []
    labels = []
    doc_ids = []
    with open(input, 'r') as f:
        for line in f:
            if line.startswith("###"):
                doc_id = line.strip()
            elif len(line.strip()) > 0:
                label = line.strip().split("\t")[0]
                sent = line.strip().split("\t")[-1]
                doc_ids.append(doc_id)
                sentences.append(sent)
                labels.append(label)
                print(doc_id, label, sent)
    data = {
        "context": sentences,
        "label": labels,
        "doc_id": doc_ids
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    return output_file


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-f', '--file', type = str, help='input file')
    parser.add_argument('-o', '--output', type = str, help='input file')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


if __name__=="__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_to_csv.py \
        -f /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/train_scibert.txt \
        -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/train.csv
    """
    args = cmd()
    print(args.file)
    # write_in_csv_format(args.file, args.output)
    df1 = pd.read_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/pubmed-20k/train_enc_dec.csv")
    df2 = pd.read_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/Y2019B7/train_enc_dec.csv")
    df3 = pd.read_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/Y2021M8/data_annotated/train_enc_dec.csv")
    df4 = pd.read_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en/train_enc_dec.csv")
    dev = pd.read_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/pubmed-20k/dev_enc_dec.csv")
    test = pd.read_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/pubmed-20k/test_enc_dec.csv")
    df = pd.concat([df1, df2, df3, df4])
    df = df[df["context"].notna()]
    for item in df.iterrows():
        if len(item[1]['context'].strip()) == 0:
            print(item[1]['context'], item[1]['label'])

    df.rename(columns={"label": "label_detail"}, inplace=True)
    df.rename(columns={"labels_abbr": "label"}, inplace=True)
    df["doc_id"] = df["doc_id"].map(lambda x: "###" + str(x))
    dev.rename(columns={"label": "label_detail"}, inplace=True)
    dev.rename(columns={"labels_abbr": "label"}, inplace=True)
    dev["doc_id"] = dev["doc_id"].map(lambda x: "###" + str(x))
    test.rename(columns={"label": "label_detail"}, inplace=True)
    test.rename(columns={"labels_abbr": "label"}, inplace=True)
    test["doc_id"] = test["doc_id"].map(lambda x: "###" + str(x))
    print(df.head)
    df[["context", "label", "label_detail", "doc_id", "sent_id"]].to_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/train.csv", index=False)
    dev[["context", "label", "label_detail", "doc_id", "sent_id"]].to_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/dev.csv", index=False)
    test[["context", "label", "label_detail", "doc_id", "sent_id"]].to_csv("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/test.csv", index=False)