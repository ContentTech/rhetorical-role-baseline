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
    parser.add_argument('-f', '--file', type=str, default=None, help='ugc.csv')
    parser.add_argument('-l', '--label_file', type=str, default=None, help='ugc.csv')
    args = parser.parse_args()
    print("argparse.args=", args, type(args))
    d = args.__dict__
    for key, value in d.items():
        print('%s = %s' % (key, value))
    return args

import sklearn.metrics as metrics
def calc_metric(df):
    """
    :param df:
    :param threshold
    :return:
    """
    print("-" * 100)
    pred, label = list(df['pred']), list(df['label'])
    macro_f1 = round(metrics.f1_score(y_pred=pred, y_true=label, average="macro"), 4)
    micro_f1 = round(metrics.f1_score(y_pred=pred, y_true=label, average="micro"), 4)
    weighted_f1 = round(metrics.f1_score(y_pred=pred, y_true=label, average="weighted"), 4)
    accuracy_score = round(metrics.accuracy_score(y_pred=pred, y_true=label), 4)
    print("weighted_f1: ", weighted_f1)
    print("accuracy_score: ", accuracy_score)
    print("macro_f1: ", macro_f1)
    print("micro_f1: ", micro_f1)
    print("-"*100)



def confusion(y_true, y_pred):
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score
    print('------Weighted------')
    print('Weighted precision', round(precision_score(y_true, y_pred, average='weighted'), 4))
    print('Weighted recall', round(recall_score(y_true, y_pred, average='weighted'), 4))
    print('Weighted f1-score', round(f1_score(y_true, y_pred, average='weighted'), 4))
    print('------Macro------')
    print('Macro precision', round(precision_score(y_true, y_pred, average='macro'), 4))
    print('Macro recall', round(recall_score(y_true, y_pred, average='macro'), 4))
    print('Macro f1-score', round(f1_score(y_true, y_pred, average='macro'), 4))
    print('------Micro------')
    print('Micro precision', round(precision_score(y_true, y_pred, average='micro'), 4))
    print('Micro recall', round(recall_score(y_true, y_pred, average='micro'), 4))
    print('Micro f1-score', round(f1_score(y_true, y_pred, average='micro'), 4))
    t = classification_report(y_true, y_pred)
    print(t)


if __name__ == "__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/show_result.py \
    -f /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D3/final_result.csv
    -f /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_bz64_baseline_wo_context_D0/test_predict.csv
    -f /mnt/fengyao.hjj/rhetorical-role-baseline/results/zlucia_legalbert_baseline_wo_context_D0/test_predict.csv

    -f /mnt/fengyao.hjj/rhetorical-role-baseline/results/zlucia_legalbert_baseline_wo_context_D4/dev_predict.csv
    -f /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_bz64_baseline_wo_context_D4/dev_predict.csv
    -f /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_bz64_baseline_wo_context_D0/dev_predict.csv
    -f /mnt/fengyao.hjj/rhetorical-role-baseline/results/zlucia_legalbert_baseline_wo_context_D0/dev_predict.csv
    -f /mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_baseline_wo_context/test_predict.csv
    
    python /Users/fengyao/PycharmProjects/rhetorical-role-baseline/experiments/show_result.py \
    -f /Users/fengyao/PycharmProjects/rhetorical-role-baseline/results/baseline/prediction.json \
    -l /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/legaleval/dev.json
    
    python /Users/fengyao/PycharmProjects/rhetorical-role-baseline/experiments/show_result.py \
    -l /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/legaleval/train.json
    """
    args = cmd()
    input_file = args.file
    ground = args.label_file
    if input_file and input_file.split('.')[-1] == 'csv':
        df = pd.read_csv(args.file, encoding='utf-8')
        df['pred'] = df['predicted_labels']
        calc_metric(df)
        confusion(list(df['label']), list(df['pred']))
        for doc in df['doc_id'].unique():
            for item in df[df['doc_id'] == doc].iterrows():
                example = item[1]         
                # if example['label'] == example['predicted_labels']:
                #     print(colored(example['label'], 'green'), colored(example['predicted_labels'], 'green'), example['context'])
                if example['label'] != example['predicted_labels']:
                    print(colored(example['label'], 'green'), colored(example['predicted_labels'], 'red'), example['context'])
        label_dfu = df.label.value_counts()
        pred_dfu = df.predicted_labels.value_counts()
        print(label_dfu)
        print(pred_dfu)


    if input_file is None and ground.split('.')[-1] == 'json':
        ground = json.load(open(ground, 'r'))
        mlabels = []
        for gro in ground:
            print(f"\n{gro['id']}")
            # if gro['id'] in [4282, 4254, 4227, ]:
            labels = []
            for g in gro['annotations'][0]['result']:
                # print(' '.join(g['value']['labels']),
                #       g['value']['text'].strip().lstrip().replace('\n', '').replace('\t', ''))
                if (len(labels) == 0) or (len(labels) > 0 and g['value']['labels'][0] != labels[-1]):
                    if g['value']['labels'][0] != 'NONE':
                        labels.append(g['value']['labels'][0])
            if labels not in mlabels:
                mlabels.append(labels)
                print(colored(labels, 'green'))

    if input_file and input_file.split('.')[-1] == 'json' and ground:
        prediction = json.load(open(input_file, 'r'))
        ground = json.load(open(ground, 'r'))
        mlabels = []
        mpreditions = []
        for pred, gro in zip(prediction, ground):
            print(f"\n{gro['id']}")
            labels = []
            preditions = []
            for p, g in zip(pred['annotations'][0]['result'], gro['annotations'][0]['result']):
                if p['value']['labels'] != g['value']['labels']:
                    color = 'red'
                else:
                    color = 'green'
                print(colored(p['value']['labels'][0], color),
                      ' '.join(g['value']['labels']),
                      g['value']['text'].strip().lstrip().replace('\n', '').replace('\t', ''))
                if (len(labels) == 0) or (len(labels) > 0 and g['value']['labels'][0] != labels[-1]):
                    if g['value']['labels'][0] != 'NONE':
                        labels.append(g['value']['labels'][0])
                if (len(preditions) == 0) or (len(preditions) > 0 and p['value']['labels'][0] != preditions[-1]):
                    if p['value']['labels'][0] != 'NONE':
                        preditions.append(p['value']['labels'][0])
            if labels not in mlabels:
                mlabels.append(labels)
            if preditions not in mpreditions:
                mpreditions.append(preditions)
        for labels, preditions in zip(mlabels, mpreditions):
            print(colored(labels, 'green'))
            if preditions not in mlabels:
                print(colored(preditions, 'red'))
            else:
                print(colored(preditions, 'blue'))
